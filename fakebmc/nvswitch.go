package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// NVLink switch tray BMC: the NVLink fabric, its NVSwitch chips and their
// ports, modelled on NVIDIA's switch tray BMC (NVIDIA/bmcweb fabric routes).
//
// Cabling: GPU g (numbered across trays) link l lands on switch l % S, port
// g*(L/S) + l/S, so every GPU spreads its L links evenly over the S switches.
// Disabling a switch port takes the GPU link down at once; the BMC publishes
// the affected links to that tray's sideband (nvlink-disabled-switch).

const fabricID = "NVLinkFabric_0"

type switchState struct {
	mu   sync.Mutex
	path string

	DisabledPorts map[string]bool   `json:"disabled_ports"` // "<switch>:<port>"
	Isolation     map[int]string    `json:"isolation"`      // switch -> SwitchIsolationMode
	PPCIe         map[int]bool      `json:"ppcie"`          // switch -> PPCIeModeEnabled
	Config        *switchConfigFile `json:"config,omitempty"`
}

type switchConfigFile struct {
	FileName   string    `json:"file_name"`
	Size       int64     `json:"size"`
	UploadedAt time.Time `json:"uploaded_at"`
}

type portRef struct{ Switch, Port int }

type gpuLink struct {
	Tray      int // index into cfg.Trays
	GPU, Link int // GPU index within the tray
}

func (s *Server) linksPerSwitch() int { return s.cfg.NVLinks / s.cfg.Switches }
func (s *Server) totalGPUs() int      { return len(s.cfg.Trays) * s.cfg.GPUCount }

// cabledTo returns the GPU link behind a switch port, if the port is cabled.
func (s *Server) cabledTo(p portRef) (gpuLink, bool) {
	per := s.linksPerSwitch()
	g := p.Port / per
	if g >= s.totalGPUs() {
		return gpuLink{}, false
	}
	return gpuLink{Tray: g / s.cfg.GPUCount, GPU: g % s.cfg.GPUCount, Link: (p.Port%per)*s.cfg.Switches + p.Switch}, true
}

func loadSwitchState(dir string) (*switchState, error) {
	st := &switchState{path: filepath.Join(dir, "switch-state.json"), DisabledPorts: map[string]bool{},
		Isolation: map[int]string{}, PPCIe: map[int]bool{}}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	if data, err := os.ReadFile(st.path); err == nil {
		if err := json.Unmarshal(data, st); err != nil {
			return nil, fmt.Errorf("%s: %w", st.path, err)
		}
	}
	return st, nil
}

func (st *switchState) save() error {
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	tmp := st.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, st.path)
}

// publish writes, for every tray, the GPU links whose switch port is disabled.
func (s *Server) publishSwitchLinks() error {
	perTray := make([][]string, len(s.cfg.Trays))
	for key := range s.sw.DisabledPorts {
		var p portRef
		if _, err := fmt.Sscanf(key, "%d:%d", &p.Switch, &p.Port); err != nil {
			continue
		}
		if gl, ok := s.cabledTo(p); ok {
			perTray[gl.Tray] = append(perTray[gl.Tray], fmt.Sprintf("%d %d", gl.GPU, gl.Link))
		}
	}
	for i, tray := range s.cfg.Trays {
		lines := perTray[i]
		sort.Strings(lines)
		body := "# GPU links whose NVSwitch port is disabled: <gpu> <link>\n" + strings.Join(lines, "\n")
		if len(lines) > 0 {
			body += "\n"
		}
		path := filepath.Join(tray.SidebandDir, "nvlink-disabled-switch")
		if err := os.WriteFile(path+".tmp", []byte(body), 0o644); err != nil {
			return err
		}
		if err := os.Rename(path+".tmp", path); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) nvswitchRoutes(auth func(string, http.HandlerFunc)) {
	f := root + "/Fabrics"
	auth("GET "+f, func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, collection(f, "FabricCollection", "Fabric Collection", []string{fabricID}))
	})
	auth("GET "+f+"/{fab}", s.fabric)
	auth("GET "+f+"/{fab}/Switches", s.switches)
	auth("GET "+f+"/{fab}/Switches/{sw}", s.switchGet)
	auth("PATCH "+f+"/{fab}/Switches/{sw}", s.switchPatch)
	auth("GET "+f+"/{fab}/Switches/{sw}/Ports", s.switchPorts)
	auth("GET "+f+"/{fab}/Switches/{sw}/Ports/{port}", s.switchPort)
	auth("PATCH "+f+"/{fab}/Switches/{sw}/Ports/{port}", s.switchPortPatch)
	auth("POST "+f+"/{fab}/upload-switch-config", s.uploadSwitchConfig)
	auth("DELETE "+f+"/{fab}/upload-switch-config", s.deleteSwitchConfig)
}

func (s *Server) checkFabric(w http.ResponseWriter, r *http.Request) bool {
	if r.PathValue("fab") != fabricID {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Fabric "+r.PathValue("fab")+" was not found.")
		return false
	}
	return true
}

func (s *Server) switchIndex(w http.ResponseWriter, r *http.Request) (int, bool) {
	if !s.checkFabric(w, r) {
		return 0, false
	}
	id := r.PathValue("sw")
	n, err := strconv.Atoi(strings.TrimPrefix(id, "NVSwitch_"))
	if !strings.HasPrefix(id, "NVSwitch_") || err != nil || n < 0 || n >= s.cfg.Switches {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Switch "+id+" was not found.")
		return 0, false
	}
	return n, true
}

func (s *Server) portIndex(w http.ResponseWriter, r *http.Request) (portRef, bool) {
	sw, ok := s.switchIndex(w, r)
	if !ok {
		return portRef{}, false
	}
	id := r.PathValue("port")
	n, err := strconv.Atoi(strings.TrimPrefix(id, "NVLink_"))
	if !strings.HasPrefix(id, "NVLink_") || err != nil || n < 0 || n >= s.cfg.SwitchPorts {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Port "+id+" was not found.")
		return portRef{}, false
	}
	return portRef{sw, n}, true
}

func (s *Server) fabric(w http.ResponseWriter, r *http.Request) {
	if !s.checkFabric(w, r) {
		return
	}
	path := root + "/Fabrics/" + fabricID
	nvidia := obj{
		"@odata.type":         "#NvidiaFabric.v1_0_0.NvidiaFabric",
		"SwitchConfigPushURI": path + "/upload-switch-config",
	}
	s.sw.mu.Lock()
	if c := s.sw.Config; c != nil {
		nvidia["SwitchConfig"] = obj{"FileName": c.FileName, "SizeBytes": c.Size, "UploadedAt": c.UploadedAt.Format(time.RFC3339)}
	}
	s.sw.mu.Unlock()
	writeJSON(w, 200, obj{
		"@odata.id":   path,
		"@odata.type": "#Fabric.v1_3_0.Fabric",
		"Id":          fabricID,
		"Name":        "NVLink Fabric",
		"FabricType":  "NVLink",
		"Status":      obj{"State": "Enabled", "Health": "OK"},
		"Switches":    link(path + "/Switches"),
		"Oem":         obj{"Nvidia": nvidia},
	})
}

func (s *Server) switches(w http.ResponseWriter, r *http.Request) {
	if !s.checkFabric(w, r) {
		return
	}
	ids := make([]string, s.cfg.Switches)
	for i := range ids {
		ids[i] = fmt.Sprintf("NVSwitch_%d", i)
	}
	writeJSON(w, 200, collection(root+"/Fabrics/"+fabricID+"/Switches", "SwitchCollection", "NVSwitch Collection", ids))
}

func (s *Server) switchGet(w http.ResponseWriter, r *http.Request) {
	n, ok := s.switchIndex(w, r)
	if !ok {
		return
	}
	s.sw.mu.Lock()
	iso := s.sw.Isolation[n]
	ppcie := s.sw.PPCIe[n]
	down := 0
	for key := range s.sw.DisabledPorts {
		if strings.HasPrefix(key, fmt.Sprintf("%d:", n)) {
			down++
		}
	}
	s.sw.mu.Unlock()
	if iso == "" {
		iso = "SwitchCommunicationEnabled"
	}
	health := "OK"
	if down > 0 {
		health = "Warning"
	}
	path := fmt.Sprintf("%s/Fabrics/%s/Switches/NVSwitch_%d", root, fabricID, n)
	writeJSON(w, 200, obj{
		"@odata.id":        path,
		"@odata.type":      "#Switch.v1_9_0.Switch",
		"Id":               fmt.Sprintf("NVSwitch_%d", n),
		"Name":             fmt.Sprintf("NVSwitch %d", n),
		"SwitchType":       "NVLink",
		"Manufacturer":     "NVIDIA",
		"Model":            "NVLink5 Switch",
		"TotalSwitchWidth": s.cfg.SwitchPorts,
		"Status":           obj{"State": "Enabled", "Health": health},
		"Ports":            link(path + "/Ports"),
		"Oem": obj{"Nvidia": obj{
			"@odata.type":         "#NvidiaSwitch.v1_4_0.NvidiaNVSwitch",
			"SwitchIsolationMode": iso,
			"PPCIeModeEnabled":    ppcie,
			"DisabledPortCount":   down,
		}},
	})
}

func (s *Server) switchPatch(w http.ResponseWriter, r *http.Request) {
	n, ok := s.switchIndex(w, r)
	if !ok {
		return
	}
	var req struct {
		Oem struct {
			Nvidia struct {
				SwitchIsolationMode *string
				PPCIeModeEnabled    *bool
			}
		}
	}
	if !decode(w, r, &req) {
		return
	}
	nv := req.Oem.Nvidia
	if nv.SwitchIsolationMode == nil && nv.PPCIeModeEnabled == nil {
		redfishError(w, http.StatusBadRequest, "PropertyNotWritable",
			"Writable here: Oem.Nvidia.SwitchIsolationMode, Oem.Nvidia.PPCIeModeEnabled.")
		return
	}
	if m := nv.SwitchIsolationMode; m != nil && *m != "SwitchCommunicationEnabled" && *m != "SwitchCommunicationDisabled" {
		redfishError(w, http.StatusBadRequest, "PropertyValueNotInList",
			fmt.Sprintf("The value %q for SwitchIsolationMode is not in [SwitchCommunicationEnabled SwitchCommunicationDisabled].", *m))
		return
	}
	s.sw.mu.Lock()
	if nv.SwitchIsolationMode != nil {
		s.sw.Isolation[n] = *nv.SwitchIsolationMode
	}
	if nv.PPCIeModeEnabled != nil {
		s.sw.PPCIe[n] = *nv.PPCIeModeEnabled
	}
	err := s.sw.save()
	s.sw.mu.Unlock()
	if err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) switchPorts(w http.ResponseWriter, r *http.Request) {
	n, ok := s.switchIndex(w, r)
	if !ok {
		return
	}
	ids := make([]string, s.cfg.SwitchPorts)
	for i := range ids {
		ids[i] = fmt.Sprintf("NVLink_%d", i)
	}
	writeJSON(w, 200, collection(fmt.Sprintf("%s/Fabrics/%s/Switches/NVSwitch_%d/Ports", root, fabricID, n),
		"PortCollection", "NVSwitch Port Collection", ids))
}

func (s *Server) switchPort(w http.ResponseWriter, r *http.Request) {
	p, ok := s.portIndex(w, r)
	if !ok {
		return
	}
	s.sw.mu.Lock()
	disabled := s.sw.DisabledPorts[fmt.Sprintf("%d:%d", p.Switch, p.Port)]
	s.sw.mu.Unlock()
	gl, cabled := s.cabledTo(p)
	state, status, health := "Enabled", "LinkUp", "OK"
	switch {
	case disabled:
		state, status, health = "Disabled", "LinkDown", "Warning"
	case !cabled:
		status = "NoLink"
	}
	nvidia := obj{"@odata.type": "#NvidiaPort.v1_4_0.NvidiaNVLinkPort", "TXWidth": 2, "RXWidth": 2}
	if cabled {
		tray := s.cfg.Trays[gl.Tray].Name
		nvidia["RemoteEndpoint"] = obj{
			"Host":    tray,
			"BMC":     tray + "-bmc",
			"GPU":     fmt.Sprintf("GPU_%d", gl.GPU),
			"GPUUUID": gpuUUID(tray, gl.GPU),
			"Port":    fmt.Sprintf("NVLink_%d", gl.Link),
		}
	}
	writeJSON(w, 200, obj{
		"@odata.id":        r.URL.Path,
		"@odata.type":      "#Port.v1_11_0.Port",
		"Id":               fmt.Sprintf("NVLink_%d", p.Port),
		"Name":             fmt.Sprintf("NVSwitch %d NVLink %d", p.Switch, p.Port),
		"PortProtocol":     "NVLink",
		"PortType":         "DownstreamPort",
		"LinkState":        state,
		"LinkStatus":       status,
		"Width":            2,
		"CurrentSpeedGbps": 200,
		"MaxSpeedGbps":     200,
		"Status":           obj{"State": map[bool]string{true: "Disabled", false: "Enabled"}[disabled], "Health": health},
		"Oem":              obj{"Nvidia": nvidia},
	})
}

// Switch ports change state immediately (unlike GPU ports, which wait for
// the tray to reset).
func (s *Server) switchPortPatch(w http.ResponseWriter, r *http.Request) {
	p, ok := s.portIndex(w, r)
	if !ok {
		return
	}
	var req struct{ LinkState string }
	if !decode(w, r, &req) {
		return
	}
	if req.LinkState != "Enabled" && req.LinkState != "Disabled" {
		redfishError(w, http.StatusBadRequest, "PropertyValueNotInList",
			fmt.Sprintf("The value %q for the property LinkState is not in the list of acceptable values [Enabled Disabled].", req.LinkState))
		return
	}
	key := fmt.Sprintf("%d:%d", p.Switch, p.Port)
	s.sw.mu.Lock()
	if req.LinkState == "Disabled" {
		s.sw.DisabledPorts[key] = true
	} else {
		delete(s.sw.DisabledPorts, key)
	}
	err := s.sw.save()
	if err == nil {
		err = s.publishSwitchLinks()
	}
	s.sw.mu.Unlock()
	if err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// upload-switch-config: multipart/form-data with exactly one part named
// ImportFile, as in NVIDIA's bmcweb. The file is kept; it is not interpreted.
func (s *Server) uploadSwitchConfig(w http.ResponseWriter, r *http.Request) {
	if !s.checkFabric(w, r) {
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 64<<20)
	if err := r.ParseMultipartForm(16 << 20); err != nil {
		redfishError(w, http.StatusBadRequest, "MalformedJSON", "Expected multipart/form-data with an ImportFile part: "+err.Error())
		return
	}
	f, hdr, err := r.FormFile("ImportFile")
	if err != nil {
		redfishError(w, http.StatusBadRequest, "ActionParameterMissing", "The ImportFile part is required.")
		return
	}
	defer f.Close()
	dst := filepath.Join(s.cfg.StateDir, "switch-config.bin")
	out, err := os.Create(dst)
	if err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return
	}
	n, err := io.Copy(out, f)
	out.Close()
	if err != nil || n == 0 {
		os.Remove(dst)
		redfishError(w, http.StatusBadRequest, "ActionParameterValueError", "The ImportFile part is empty or unreadable.")
		return
	}
	s.sw.mu.Lock()
	s.sw.Config = &switchConfigFile{FileName: filepath.Base(hdr.Filename), Size: n, UploadedAt: time.Now().UTC()}
	err = s.sw.save()
	s.sw.mu.Unlock()
	if err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) deleteSwitchConfig(w http.ResponseWriter, r *http.Request) {
	if !s.checkFabric(w, r) {
		return
	}
	s.sw.mu.Lock()
	defer s.sw.mu.Unlock()
	if s.sw.Config == nil {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "No switch configuration file is present.")
		return
	}
	os.Remove(filepath.Join(s.cfg.StateDir, "switch-config.bin"))
	s.sw.Config = nil
	if err := s.sw.save(); err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
