package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"net/http"
	"os"
	"syscall"
	"unsafe"
)

// GPU sensors, as a GB200 tray's BMC reads them from the GPUs out of band.
// Readings are computed exactly like the tray's fake NVML (fakegpu/nvml_stub.c,
// "telemetry model"), from the same shared state (the occupancy file, mounted
// read-only), so the BMC, nvidia-smi and the GPU exporter report the same
// values at the same moment. Keep the constants and formulas in sync.

const (
	idleMW       = 140000.0
	maxMW        = 1000000.0
	idleMC       = 32000.0 // milli-degrees C
	fullMC       = 75000.0
	tempTauNs    = 20e9 // thermal time constant
	utilWindowNs = 1000000000
	powerLimitW  = 1200 // nvmlDeviceGetPowerManagementLimit
)

// gpuState mirrors struct fg_gpu_state in fakegpu/occupancy.h.
type gpuState struct {
	BusyNs, SampleNs, SampleBusy uint64
	Util, TempMc                 uint32
	TempNs, NVLinkTx, NVLinkRx   uint64
}

const maxTrayGPUs = 8 // FG_MAX_GPUS

// gpuReset mirrors struct fg_gpu_reset in fakegpu/occupancy.h. The array
// follows gpu[FG_MAX_GPUS], proc[FG_MAX_PROCS] (16 bytes each) and
// pidns[FG_MAX_PROCS] (8 bytes each).
const (
	maxProcs   = 128 // FG_MAX_PROCS
	resetEntry = 16  // sizeof(struct fg_gpu_reset)
)

// gpuClique is the clique the GPU reports through NVML: the one it took at
// its last reset (a partition change waits for a GPU reset), or before any
// reset the one its partition has.
func (s *Server) gpuClique(g int) int {
	off := maxTrayGPUs*binary.Size(gpuState{}) + maxProcs*16 + maxProcs*8 + g*resetEntry
	if data, err := os.ReadFile(s.cfg.TelemetryPath); err == nil && len(data) >= off+4 {
		if c := binary.LittleEndian.Uint32(data[off:]); c != 0 {
			return int(c) - 1
		}
	}
	return s.state.Clique(g, s.cfg.CliqueID)
}

// gpuReading is one GPU's sensors at one instant.
type gpuReading struct {
	TempC  int     // as nvmlDeviceGetTemperature: whole degrees, rounded
	PowerW float64 // as nvmlDeviceGetPowerUsage: milliwatt resolution
}

// monotonicNs is fakegpu's clock (CLOCK_MONOTONIC), shared by all
// containers on the host; the jitter and the thermal lag depend on it.
func monotonicNs() uint64 {
	var ts syscall.Timespec
	syscall.Syscall(syscall.SYS_CLOCK_GETTIME, 1, uintptr(unsafe.Pointer(&ts)), 0)
	return uint64(ts.Sec)*1000000000 + uint64(ts.Nsec)
}

// readGPUStates returns the tray's GPU states; a missing file reads as the
// zero state the tray's NVML would create on first use.
func readGPUStates(path string) ([]gpuState, error) {
	states := make([]gpuState, maxTrayGPUs)
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return states, nil
	} else if err != nil {
		return nil, err
	}
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, states); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return states, nil
}

// utilization is NVML's utilization(), without storing the new sample: the
// value NVML would return now.
func utilization(g *gpuState, now uint64) uint32 {
	if g.SampleNs == 0 {
		return 0
	}
	if now-g.SampleNs >= utilWindowNs {
		u := float64(g.BusyNs-g.SampleBusy) / float64(now-g.SampleNs) * 100.0
		if u > 100 {
			u = 100
		}
		return uint32(u)
	}
	return g.Util
}

func jitter(idx int, hz float64, now uint64) float64 {
	t := float64(now) / 1e9
	return math.Sin(t*hz*6.283+float64(idx)*1.7)*0.6 + math.Sin(t*hz*2.7*6.283+float64(idx))*0.4
}

func powerMW(idx int, util uint32, now uint64) uint32 {
	u := float64(util) / 100.0
	p := idleMW + u*(maxMW-idleMW)
	return uint32(p * (1.0 + 0.02*jitter(idx, 0.5, now)))
}

func temperatureMC(idx int, g *gpuState, util uint32, now uint64) uint32 {
	target := idleMC + float64(util)/100.0*(fullMC-idleMC)
	cur := float64(g.TempMc)
	if g.TempNs == 0 || cur == 0 {
		cur = idleMC
	} else {
		cur += (target - cur) * (1.0 - math.Exp(-float64(now-g.TempNs)/tempTauNs))
	}
	return uint32(cur + 400.0*jitter(idx, 0.2, now))
}

// gpuReadings returns the sensors of every GPU now, or nil with the tray
// powered off (no readings, as from GPUs without power).
func (s *Server) gpuReadings() ([]gpuReading, error) {
	if st, _ := s.power.PowerState(); st != "On" {
		return nil, nil
	}
	states, err := readGPUStates(s.cfg.TelemetryPath)
	if err != nil {
		return nil, err
	}
	now := monotonicNs()
	out := make([]gpuReading, s.cfg.GPUCount)
	for i := range out {
		g := &states[i]
		u := utilization(g, now)
		out[i] = gpuReading{
			TempC:  int((temperatureMC(i, g, u, now) + 500) / 1000),
			PowerW: float64(powerMW(i, u, now)) / 1000,
		}
	}
	return out, nil
}

// ---- Redfish resources -------------------------------------------------------
//
// Each GPU's sensors live on its own chassis (HGX_GPU_<n>/Sensors, as on the
// real tray); Chassis_0 carries the tray's total GPU power and the thermal
// summary of all GPUs, pointing at the GPU sensors.

func gpuSensorIDs(g int) (temp, power string) {
	return fmt.Sprintf("HGX_GPU_%d_TEMP_0", g), fmt.Sprintf("HGX_GPU_%d_Power_0", g)
}

const totalPowerSensor = "Total_GPU_Power_0"

func sensorURI(chassis, sensor string) string {
	return root + "/Chassis/" + chassis + "/Sensors/" + sensor
}

// excerpt is a SensorExcerpt: the reading inline, the sensor linked.
func excerpt(uri string, reading any) obj { return obj{"Reading": reading, "DataSourceUri": uri} }

// readings returns the GPU readings for a handler, or writes the error.
func (s *Server) readings(w http.ResponseWriter) ([]gpuReading, bool) {
	rs, err := s.gpuReadings()
	if err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return nil, false
	}
	return rs, true
}

// value is a reading or null when the tray is off.
func value[T any](rs []gpuReading, g int, f func(gpuReading) T) any {
	if rs == nil {
		return nil
	}
	return f(rs[g])
}

func totalPower(rs []gpuReading) any {
	if rs == nil {
		return nil
	}
	var sum float64
	for _, r := range rs {
		sum += r.PowerW
	}
	return math.Round(sum*1000) / 1000
}

func sensorStatus(rs []gpuReading) obj {
	if rs == nil {
		return obj{"State": "UnavailableOffline", "Health": "OK"}
	}
	return obj{"State": "Enabled", "Health": "OK"}
}

// sensorChassis resolves the chassis of a sensor request: Chassis_0 (gpu -1)
// or HGX_GPU_<n>; anything else is written as 404.
func (s *Server) sensorChassis(w http.ResponseWriter, r *http.Request) (gpu int, ok bool) {
	ch := r.PathValue("ch")
	if ch == chassisID {
		return -1, true
	}
	if g, isGPU := s.gpuChassisIndex(ch); isGPU {
		return g, true
	}
	redfishError(w, http.StatusNotFound, "ResourceNotFound", "Chassis "+ch+" has no sensors.")
	return 0, false
}

func (s *Server) sensorList(w http.ResponseWriter, r *http.Request) {
	gpu, ok := s.sensorChassis(w, r)
	if !ok {
		return
	}
	ids := []string{totalPowerSensor}
	if gpu >= 0 {
		t, p := gpuSensorIDs(gpu)
		ids = []string{t, p}
	}
	writeJSON(w, 200, collection(root+"/Chassis/"+r.PathValue("ch")+"/Sensors", "SensorCollection", "Sensor Collection", ids))
}

func (s *Server) sensor(w http.ResponseWriter, r *http.Request) {
	gpu, ok := s.sensorChassis(w, r)
	if !ok {
		return
	}
	id := r.PathValue("sensor")
	tempID, powerID := gpuSensorIDs(gpu)
	if (gpu < 0 && id != totalPowerSensor) || (gpu >= 0 && id != tempID && id != powerID) {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Sensor "+id+" was not found.")
		return
	}
	rs, ok := s.readings(w)
	if !ok {
		return
	}
	body := obj{
		"@odata.id":   sensorURI(r.PathValue("ch"), id),
		"@odata.type": "#Sensor.v1_9_0.Sensor",
		"Id":          id,
		"Status":      sensorStatus(rs),
	}
	switch id {
	case totalPowerSensor:
		body["Name"] = "Total GPU Power"
		body["ReadingType"], body["ReadingUnits"] = "Power", "W"
		body["PhysicalContext"] = "GPUSubsystem"
		body["Reading"] = totalPower(rs)
		body["RelatedItem"] = []obj{link(root + "/Chassis/" + chassisID)}
	case tempID:
		body["Name"] = fmt.Sprintf("GPU %d Temperature", gpu)
		body["ReadingType"], body["ReadingUnits"] = "Temperature", "Cel"
		body["PhysicalContext"] = "GPU"
		body["Reading"] = value(rs, gpu, func(r gpuReading) int { return r.TempC })
		body["RelatedItem"] = []obj{link(gpuPath(gpu))}
	default:
		body["Name"] = fmt.Sprintf("GPU %d Power", gpu)
		body["ReadingType"], body["ReadingUnits"] = "Power", "W"
		body["PhysicalContext"] = "GPU"
		body["Reading"] = value(rs, gpu, func(r gpuReading) float64 { return r.PowerW })
		body["RelatedItem"] = []obj{link(gpuPath(gpu))}
	}
	writeJSON(w, 200, body)
}

// gpuMetrics is a GPU's EnvironmentMetrics, served on its processor (where
// nv-redfish looks) and on its HGX_GPU_<n> chassis.
func (s *Server) gpuMetrics(path string, g int, rs []gpuReading) obj {
	temp, power := gpuSensorIDs(g)
	ch := gpuChassisID(g)
	return obj{
		"@odata.id":          path,
		"@odata.type":        "#EnvironmentMetrics.v1_3_0.EnvironmentMetrics",
		"Id":                 "EnvironmentMetrics",
		"Name":               fmt.Sprintf("GPU %d Environment Metrics", g),
		"TemperatureCelsius": excerpt(sensorURI(ch, temp), value(rs, g, func(r gpuReading) int { return r.TempC })),
		"PowerWatts":         excerpt(sensorURI(ch, power), value(rs, g, func(r gpuReading) float64 { return r.PowerW })),
		"PowerLimitWatts":    obj{"SetPoint": powerLimitW, "AllowableMax": powerLimitW, "ControlMode": "Automatic"},
	}
}

func (s *Server) gpuEnvironment(w http.ResponseWriter, r *http.Request) {
	g, ok := s.gpuIndex(w, r)
	if !ok {
		return
	}
	if rs, ok := s.readings(w); ok {
		writeJSON(w, 200, s.gpuMetrics(gpuPath(g)+"/EnvironmentMetrics", g, rs))
	}
}

// Chassis EnvironmentMetrics: the tray's GPU power on Chassis_0, the GPU's
// metrics on HGX_GPU_<n>.
func (s *Server) chassisEnvironment(w http.ResponseWriter, r *http.Request) {
	gpu, ok := s.sensorChassis(w, r)
	if !ok {
		return
	}
	rs, ok := s.readings(w)
	if !ok {
		return
	}
	path := root + "/Chassis/" + r.PathValue("ch") + "/EnvironmentMetrics"
	if gpu >= 0 {
		writeJSON(w, 200, s.gpuMetrics(path, gpu, rs))
		return
	}
	writeJSON(w, 200, obj{
		"@odata.id":   path,
		"@odata.type": "#EnvironmentMetrics.v1_3_0.EnvironmentMetrics",
		"Id":          "EnvironmentMetrics",
		"Name":        "Chassis Environment Metrics",
		"PowerWatts":  excerpt(sensorURI(chassisID, totalPowerSensor), totalPower(rs)),
	})
}

// ThermalSubsystem of Chassis_0: liquid-cooled tray, so temperatures only, no
// fans; one reading per GPU, from the GPU's own sensor.
func (s *Server) thermalSubsystem(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("ch") != chassisID {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Chassis "+r.PathValue("ch")+" has no thermal subsystem.")
		return
	}
	path := root + "/Chassis/" + chassisID + "/ThermalSubsystem"
	writeJSON(w, 200, obj{
		"@odata.id":      path,
		"@odata.type":    "#ThermalSubsystem.v1_3_0.ThermalSubsystem",
		"Id":             "ThermalSubsystem",
		"Name":           "Thermal Subsystem",
		"ThermalMetrics": link(path + "/ThermalMetrics"),
		"Status":         obj{"State": "Enabled", "Health": "OK"},
	})
}

func (s *Server) thermalMetrics(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("ch") != chassisID {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Chassis "+r.PathValue("ch")+" has no thermal subsystem.")
		return
	}
	rs, ok := s.readings(w)
	if !ok {
		return
	}
	temps := make([]obj, s.cfg.GPUCount)
	for g := range temps {
		id, _ := gpuSensorIDs(g)
		temps[g] = excerpt(sensorURI(gpuChassisID(g), id), value(rs, g, func(r gpuReading) int { return r.TempC }))
		temps[g]["DeviceName"] = fmt.Sprintf("GPU_%d", g)
		temps[g]["PhysicalContext"] = "GPU"
	}
	writeJSON(w, 200, obj{
		"@odata.id":                  root + "/Chassis/" + chassisID + "/ThermalSubsystem/ThermalMetrics",
		"@odata.type":                "#ThermalMetrics.v1_3_0.ThermalMetrics",
		"Id":                         "ThermalMetrics",
		"Name":                       "Chassis Thermal Metrics",
		"TemperatureReadingsCelsius": temps,
	})
}
