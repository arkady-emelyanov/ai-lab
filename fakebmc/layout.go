package main

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// Inventory layout of NVIDIA's GB200 BMCs, from NVIDIA's own BMC models
// (infra-controller crates/bmc-mock/src/hw: wiwynn_gb200_nvl.rs,
// nvidia_gb200.rs, nvidia_gbx00.rs, nvidia_switch_nd5200_ld.rs) and the
// NVIDIA Switch BMC user manual:
//
//   compute tray  Systems    System_0 (Grace host: power, reset),
//                            HGX_Baseboard_0 (the GPUs, behind the HMC)
//                 Chassis    Chassis_0 (the tray), BMC_0, CBC_0..3 (NVLink cable
//                            cartridges, carrying the tray's rack slot),
//                            HGX_Chassis_0, HGX_GPU_<n> (one per GPU, its UUID)
//                 Managers   BMC_0 (host BMC), HGX_BMC_0 (the HMC)
//   switch tray   Systems    System_0 (the NVOS CPU)
//                 Chassis    BMC_eeprom, CPLD_0, MGX_BMC_0, MGX_NVSwitch_<n>
//                 Managers   BMC_0
//                 Fabrics    MGX_NVLinkFabric_0
//
// Chassis_0 comes first in the chassis collection: generic Redfish clients
// (the lab's exporter) take the first chassis as the primary one.

const (
	cbcCount        = 4  // NVLink cable cartridges per compute tray
	cbcSlotOffset   = 10 // ChassisPhysicalSlotNumber = ComputeTrayIndex + 10
	cbcRevisionID   = 2
	nvl72TopologyID = 128 // TopologyId of NVL72 racks' compute trays
	switchSysChass  = "BMC_eeprom"
)

func gpuChassisID(g int) string { return fmt.Sprintf("HGX_GPU_%d", g) }

func gpuPath(g int) string {
	return fmt.Sprintf("%s/Systems/%s/Processors/GPU_%d", root, hgxSystemID, g)
}

// ---- systems ---------------------------------------------------------------------

func (s *Server) systemIDs() []string {
	if s.isSwitch() {
		return []string{systemID}
	}
	return []string{systemID, hgxSystemID}
}

func (s *Server) systems(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, collection(root+"/Systems", "ComputerSystemCollection", "Computer System Collection", s.systemIDs()))
}

func (s *Server) system(w http.ResponseWriter, r *http.Request) {
	path := root + "/Systems/" + r.PathValue("sys")
	reset := obj{"#ComputerSystem.Reset": obj{
		"target":                            path + "/Actions/ComputerSystem.Reset",
		"ResetType@Redfish.AllowableValues": resetTypes,
	}}
	body := obj{
		"@odata.id":    path,
		"@odata.type":  "#ComputerSystem.v1_20_0.ComputerSystem",
		"Id":           r.PathValue("sys"),
		"SystemType":   "Physical",
		"Manufacturer": "NVIDIA",
	}
	switch {
	case s.isSwitch() && r.PathValue("sys") == systemID:
		body["Name"] = "NVLink switch tray CPU"
		body["Model"] = "NVLink5 switch tray"
		body["SerialNumber"] = traySerial("nvswitch")
		body["PowerState"] = "On"
		body["Status"] = obj{"State": "Enabled", "Health": "OK"}
		body["Actions"] = reset
		body["Links"] = obj{"Chassis": []obj{link(root + "/Chassis/" + switchSysChass)},
			"ManagedBy": []obj{link(root + "/Managers/" + managerID)}}
	case !s.isSwitch() && r.PathValue("sys") == systemID:
		power, status := s.powerState()
		body["Name"] = "System"
		body["HostName"] = s.cfg.Tray
		body["Model"] = s.cfg.ProductName
		body["SerialNumber"] = traySerial(s.cfg.Tray)
		body["PowerState"] = power
		body["Status"] = status
		body["Actions"] = reset
		body["Links"] = obj{"Chassis": []obj{link(root + "/Chassis/" + chassisID)},
			"ManagedBy": []obj{link(root + "/Managers/" + managerID)}}
	case !s.isSwitch() && r.PathValue("sys") == hgxSystemID:
		power, status := s.powerState()
		body["Name"] = "HGX Baseboard"
		body["Model"] = s.cfg.ProductName
		body["PowerState"] = power
		body["Status"] = status
		body["ProcessorSummary"] = obj{"Count": s.cfg.GPUCount}
		body["Processors"] = link(path + "/Processors")
		body["Links"] = obj{"Chassis": []obj{link(root + "/Chassis/" + hgxChassisID)},
			"ManagedBy": []obj{link(root + "/Managers/" + hmcID)}}
	default:
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "System "+r.PathValue("sys")+" was not found.")
		return
	}
	writeJSON(w, 200, body)
}

// ---- chassis ---------------------------------------------------------------------

func (s *Server) chassisIDs() []string {
	if s.isSwitch() {
		ids := []string{switchSysChass, "CPLD_0", "MGX_BMC_0"}
		for n := 0; n < s.cfg.Switches; n++ {
			ids = append(ids, fmt.Sprintf("MGX_NVSwitch_%d", n))
		}
		return ids
	}
	ids := []string{chassisID, managerID}
	for n := 0; n < cbcCount; n++ {
		ids = append(ids, fmt.Sprintf("CBC_%d", n))
	}
	ids = append(ids, hgxChassisID)
	for g := 0; g < s.cfg.GPUCount; g++ {
		ids = append(ids, gpuChassisID(g))
	}
	return ids
}

func (s *Server) chassisList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, collection(root+"/Chassis", "ChassisCollection", "Chassis Collection", s.chassisIDs()))
}

// gpuChassisIndex returns n for HGX_GPU_<n>.
func (s *Server) gpuChassisIndex(id string) (int, bool) {
	n, err := strconv.Atoi(strings.TrimPrefix(id, "HGX_GPU_"))
	return n, strings.HasPrefix(id, "HGX_GPU_") && err == nil && n >= 0 && n < s.cfg.GPUCount
}

func (s *Server) chassis(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("ch")
	known := false
	for _, c := range s.chassisIDs() {
		known = known || c == id
	}
	if !known {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Chassis "+id+" was not found.")
		return
	}
	path := root + "/Chassis/" + id
	body := obj{
		"@odata.id":    path,
		"@odata.type":  "#Chassis.v1_23_0.Chassis",
		"Id":           id,
		"Name":         id,
		"ChassisType":  "Component",
		"Manufacturer": "NVIDIA",
		"Status":       obj{"State": "Enabled", "Health": "OK"},
	}
	managedBy := func(m string) obj { return obj{"ManagedBy": []obj{link(root + "/Managers/" + m)}} }

	if s.isSwitch() {
		body["PowerState"] = "On"
		body["Links"] = managedBy(managerID)
		switch {
		case id == switchSysChass:
			body["ChassisType"], body["Model"] = "Module", "NVLink5 switch tray"
			body["Links"] = obj{"ComputerSystems": []obj{link(root + "/Systems/" + systemID)},
				"ManagedBy": []obj{link(root + "/Managers/" + managerID)}}
		case id == "CPLD_0":
			body["ChassisType"], body["Model"] = "Module", "Switch tray CPLD"
		case id == "MGX_BMC_0":
			body["Model"] = "Switch tray BMC"
		default: // MGX_NVSwitch_<n>
			n := strings.TrimPrefix(id, "MGX_NVSwitch_")
			body["Model"] = "NVLink5 NVSwitch"
			body["Links"] = obj{"ManagedBy": []obj{link(root + "/Managers/" + managerID)},
				"Switches": []obj{link(root + "/Fabrics/" + fabricID + "/Switches/NVSwitch_" + n)}}
		}
		writeJSON(w, 200, body)
		return
	}

	power, status := s.powerState()
	body["PowerState"], body["Status"] = power, status
	switch {
	case id == chassisID:
		body["Name"] = s.cfg.Tray
		body["ChassisType"] = "RackMount"
		body["Model"] = s.cfg.ProductName
		body["SerialNumber"] = traySerial(s.cfg.Tray)
		body["Assembly"] = link(path + "/Assembly")
		body["Sensors"] = link(path + "/Sensors")
		body["EnvironmentMetrics"] = link(path + "/EnvironmentMetrics")
		body["ThermalSubsystem"] = link(path + "/ThermalSubsystem")
		body["Links"] = obj{"ComputerSystems": []obj{link(root + "/Systems/" + systemID)},
			"ManagedBy": []obj{link(root + "/Managers/" + managerID)}}
	case id == managerID:
		body["ChassisType"], body["Model"] = "Module", s.cfg.ProductName
		body["Links"] = managedBy(managerID)
	case strings.HasPrefix(id, "CBC_"):
		// NVLink cable cartridge: topology tools place the tray in the rack
		// (and so in the NVLink domain) from these.
		body["Manufacturer"], body["Model"] = "Nvidia", "18x1RU CBL Cartridge"
		body["Links"] = managedBy(managerID)
		body["Oem"] = obj{"Nvidia": obj{
			"@odata.type":               "#NvidiaChassis.v1_4_0.NvidiaCBCChassis",
			"ChassisPhysicalSlotNumber": s.cfg.TrayIndex + cbcSlotOffset,
			"ComputeTrayIndex":          s.cfg.TrayIndex,
			"RevisionId":                cbcRevisionID,
			"TopologyId":                nvl72TopologyID,
		}}
	case id == hgxChassisID:
		body["Model"] = s.cfg.ProductName
		contains := make([]obj, s.cfg.GPUCount)
		for g := range contains {
			contains[g] = link(root + "/Chassis/" + gpuChassisID(g))
		}
		body["Links"] = obj{"ComputerSystems": []obj{link(root + "/Systems/" + hgxSystemID)},
			"ManagedBy": []obj{link(root + "/Managers/" + hmcID)}, "Contains": contains}
	default: // HGX_GPU_<n>
		g, _ := s.gpuChassisIndex(id)
		body["Model"] = s.cfg.GPUName
		body["SerialNumber"] = gpuSerial(s.cfg.Tray, g)
		body["UUID"] = gpuUUID(s.cfg.Tray, g)
		body["Sensors"] = link(path + "/Sensors")
		body["EnvironmentMetrics"] = link(path + "/EnvironmentMetrics")
		body["Links"] = obj{"ManagedBy": []obj{link(root + "/Managers/" + hmcID)},
			"ContainedBy": link(root + "/Chassis/" + hgxChassisID),
			"Processors":  []obj{link(gpuPath(g))}}
	}
	writeJSON(w, 200, body)
}

// Chassis_0/Assembly: the tray's serial, where inventory tools read it.
func (s *Server) assembly(w http.ResponseWriter, r *http.Request) {
	if s.isSwitch() || r.PathValue("ch") != chassisID {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "No assembly for chassis "+r.PathValue("ch")+".")
		return
	}
	path := root + "/Chassis/" + chassisID + "/Assembly"
	writeJSON(w, 200, obj{
		"@odata.id":   path,
		"@odata.type": "#Assembly.v1_5_0.Assembly",
		"Id":          "Assembly",
		"Name":        "Assembly data",
		"Assemblies": []obj{{
			"@odata.id":    path + "#/Assemblies/0",
			"MemberId":     "0",
			"Name":         s.cfg.Tray,
			"Model":        s.cfg.ProductName,
			"Vendor":       "NVIDIA",
			"SerialNumber": traySerial(s.cfg.Tray),
		}},
	})
}

// ---- managers --------------------------------------------------------------------

func (s *Server) managerIDs() []string {
	if s.isSwitch() {
		return []string{managerID}
	}
	return []string{managerID, hmcID}
}

func (s *Server) managers(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, collection(root+"/Managers", "ManagerCollection", "Manager Collection", s.managerIDs()))
}

func (s *Server) manager(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("mgr")
	path := root + "/Managers/" + id
	body := obj{
		"@odata.id":   path,
		"@odata.type": "#Manager.v1_19_0.Manager",
		"Id":          id,
		"ManagerType": "BMC",
		"Status":      obj{"State": "Enabled", "Health": "OK"},
	}
	switch {
	case id == managerID:
		body["Name"] = "OpenBMC Manager"
		body["FirmwareVersion"] = "fakebmc-" + version
		chassis := switchSysChass
		if !s.isSwitch() {
			chassis = chassisID
		}
		body["Links"] = obj{
			"ManagerForChassis": []obj{link(root + "/Chassis/" + chassis)},
			"ManagerForServers": []obj{link(root + "/Systems/" + systemID)},
		}
		body["Actions"] = obj{"#Manager.Reset": obj{
			"target":                            path + "/Actions/Manager.Reset",
			"ResetType@Redfish.AllowableValues": []string{"GracefulRestart", "ForceRestart"},
		}}
	case id == hmcID && !s.isSwitch():
		body["Name"] = "HGX Management Controller"
		body["FirmwareVersion"] = "fakebmc-hmc-" + version
		body["Links"] = obj{
			"ManagerForChassis": []obj{link(root + "/Chassis/" + hgxChassisID)},
			"ManagerForServers": []obj{link(root + "/Systems/" + hgxSystemID)},
		}
	default:
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Manager "+id+" was not found.")
		return
	}
	writeJSON(w, 200, body)
}

// A BMC reset drops all sessions; the tray keeps running.
func (s *Server) managerReset(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("mgr") != managerID {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Manager "+r.PathValue("mgr")+" has no reset action.")
		return
	}
	s.mu.Lock()
	s.sessions = map[string]session{}
	s.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

// ---- firmware inventory ------------------------------------------------------------

// vbiosVersion is what the fake NVML reports (nvmlDeviceGetVbiosVersion).
const vbiosVersion = "97.00.82.00.0F"

func (s *Server) firmware() [][2]string {
	bmc := "fakebmc-" + version
	if s.isSwitch() {
		fw := [][2]string{{"MGX_FW_BMC_0", bmc}, {"MGX_FW_CPLD_0", "fakebmc-cpld-1"}}
		for n := 0; n < s.cfg.Switches; n++ {
			fw = append(fw, [2]string{fmt.Sprintf("MGX_FW_NVSwitch_%d", n), "fakebmc-nvswitch-1"})
		}
		return fw
	}
	fw := [][2]string{{"FW_BMC_0", bmc}, {"HGX_FW_BMC_0", "fakebmc-hmc-" + version}}
	for g := 0; g < s.cfg.GPUCount; g++ {
		fw = append(fw, [2]string{fmt.Sprintf("HGX_FW_GPU_%d", g), vbiosVersion})
	}
	return fw
}

func (s *Server) updateService(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, obj{
		"@odata.id":         root + "/UpdateService",
		"@odata.type":       "#UpdateService.v1_11_0.UpdateService",
		"Id":                "UpdateService",
		"Name":              "Update Service",
		"ServiceEnabled":    false,
		"FirmwareInventory": link(root + "/UpdateService/FirmwareInventory"),
	})
}

func (s *Server) firmwareInventory(w http.ResponseWriter, r *http.Request) {
	var ids []string
	for _, fw := range s.firmware() {
		ids = append(ids, fw[0])
	}
	writeJSON(w, 200, collection(root+"/UpdateService/FirmwareInventory", "SoftwareInventoryCollection",
		"Firmware Inventory Collection", ids))
}

func (s *Server) firmwareItem(w http.ResponseWriter, r *http.Request) {
	for _, fw := range s.firmware() {
		if fw[0] == r.PathValue("fw") {
			writeJSON(w, 200, obj{
				"@odata.id":   root + "/UpdateService/FirmwareInventory/" + fw[0],
				"@odata.type": "#SoftwareInventory.v1_10_0.SoftwareInventory",
				"Id":          fw[0],
				"Name":        fw[0],
				"Version":     fw[1],
				"Updateable":  false,
				"Status":      obj{"State": "Enabled", "Health": "OK"},
			})
			return
		}
	}
	redfishError(w, http.StatusNotFound, "ResourceNotFound", "Firmware "+r.PathValue("fw")+" was not found.")
}
