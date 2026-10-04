package main

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Resource names follow NVIDIA's GB200 BMC (NVIDIA/bmcweb): System_0 is the
// host, GPUs are Processors GPU_<n> with NVLink ports NVLink_<n>.
const (
	systemID  = "System_0"
	chassisID = "Chassis_0"
	managerID = "BMC_0"
	root      = "/redfish/v1"
)

type obj = map[string]any

type Server struct {
	cfg   *Config
	state *State       // tray role
	power *Power       // tray role
	sw    *switchState // nvswitch role

	mu       sync.Mutex
	sessions map[string]session // by token
}

type session struct {
	ID      string
	User    string
	Created time.Time
}

func newServer(cfg *Config, state *State, power *Power, sw *switchState) *Server {
	return &Server{cfg: cfg, state: state, power: power, sw: sw, sessions: map[string]session{}}
}

func (s *Server) isSwitch() bool { return s.cfg.Role == "nvswitch" }

func (s *Server) chassisID() string {
	if s.isSwitch() {
		return "NVSwitchTray_0"
	}
	return chassisID
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	// Unauthenticated, as required by the Redfish specification.
	mux.HandleFunc("GET /redfish", s.versions)
	mux.HandleFunc("GET "+root, s.serviceRoot)
	mux.HandleFunc("POST "+root+"/SessionService/Sessions", s.login)

	auth := func(pattern string, h http.HandlerFunc) { mux.Handle(pattern, s.authenticated(h)) }
	auth("GET "+root+"/SessionService", s.sessionService)
	auth("GET "+root+"/SessionService/Sessions", s.sessionList)
	auth("GET "+root+"/SessionService/Sessions/{id}", s.sessionGet)
	auth("DELETE "+root+"/SessionService/Sessions/{id}", s.sessionDelete)

	if s.isSwitch() {
		s.nvswitchRoutes(auth)
	} else {
		auth("GET "+root+"/Systems", s.systems)
		auth("GET "+root+"/Systems/{sys}", s.system)
		auth("POST "+root+"/Systems/{sys}/Actions/ComputerSystem.Reset", s.reset)
		auth("GET "+root+"/Systems/{sys}/Processors", s.processors)
		auth("GET "+root+"/Systems/{sys}/Processors/{gpu}", s.processor)
		auth("GET "+root+"/Systems/{sys}/Processors/{gpu}/Ports", s.ports)
		auth("GET "+root+"/Systems/{sys}/Processors/{gpu}/Ports/{port}", s.port)
		auth("PATCH "+root+"/Systems/{sys}/Processors/{gpu}/Ports/{port}", s.patchPort)
		auth("GET "+root+"/Systems/{sys}/Processors/{gpu}/Ports/{port}/Settings", s.portSettings)
		auth("PATCH "+root+"/Systems/{sys}/Processors/{gpu}/Ports/{port}/Settings", s.patchPortSettings)
	}

	auth("GET "+root+"/Chassis", s.chassisList)
	auth("GET "+root+"/Chassis/{ch}", s.chassis)
	auth("GET "+root+"/Managers", s.managers)
	auth("GET "+root+"/Managers/{mgr}", s.manager)
	auth("POST "+root+"/Managers/{mgr}/Actions/Manager.Reset", s.managerReset)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "The requested resource "+r.URL.Path+" was not found.")
	})
	return s.logged(stripSlash(mux))
}

// ---- plumbing ---------------------------------------------------------------

func (s *Server) logged(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: 200}
		next.ServeHTTP(rec, r)
		log.Printf("%s %s %s -> %d", r.RemoteAddr, r.Method, r.URL.Path, rec.status)
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) { r.status = code; r.ResponseWriter.WriteHeader(code) }

// stripSlash accepts both /redfish/v1/Systems and /redfish/v1/Systems/.
func stripSlash(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(r.URL.Path) > 1 && strings.HasSuffix(r.URL.Path, "/") {
			r.URL.Path = strings.TrimRight(r.URL.Path, "/")
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) authenticated(next http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if tok := r.Header.Get("X-Auth-Token"); tok != "" {
			s.mu.Lock()
			_, ok := s.sessions[tok]
			s.mu.Unlock()
			if ok {
				next(w, r)
				return
			}
		} else if u, p, ok := r.BasicAuth(); ok && s.validCredentials(u, p) {
			next(w, r)
			return
		}
		w.Header().Set("WWW-Authenticate", `Basic realm="Redfish"`)
		redfishError(w, http.StatusUnauthorized, "NoValidSession", "There is no valid session established with the implementation.")
	})
}

func (s *Server) validCredentials(user, pass string) bool {
	return subtle.ConstantTimeCompare([]byte(user), []byte(s.cfg.Username)) == 1 &&
		subtle.ConstantTimeCompare([]byte(pass), []byte(s.cfg.Password)) == 1
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("OData-Version", "4.0")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func redfishError(w http.ResponseWriter, status int, id, msg string) {
	writeJSON(w, status, obj{"error": obj{
		"code":    "Base.1.18.GeneralError",
		"message": "A general error has occurred. See ExtendedInfo for more information.",
		"@Message.ExtendedInfo": []obj{{
			"@odata.type": "#Message.v1_1_1.Message",
			"MessageId":   "Base.1.18." + id,
			"Message":     msg,
			"Severity":    "Warning",
		}},
	}})
}

func link(path string) obj { return obj{"@odata.id": path} }

func collection(path, typ, name string, members []string) obj {
	ms := make([]obj, len(members))
	for i, m := range members {
		ms[i] = link(path + "/" + m)
	}
	return obj{
		"@odata.id":           path,
		"@odata.type":         "#" + typ + "." + typ,
		"Name":                name,
		"Members":             ms,
		"Members@odata.count": len(ms),
	}
}

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		redfishError(w, http.StatusBadRequest, "MalformedJSON", "The request body submitted was malformed JSON: "+err.Error())
		return false
	}
	return true
}

// ---- service root and sessions ----------------------------------------------

func (s *Server) versions(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, obj{"v1": root + "/"})
}

func (s *Server) serviceRoot(w http.ResponseWriter, r *http.Request) {
	body := obj{
		"@odata.id":      root,
		"@odata.type":    "#ServiceRoot.v1_15_0.ServiceRoot",
		"Id":             "RootService",
		"Name":           "Root Service",
		"RedfishVersion": "1.17.0",
		"Vendor":         "NVIDIA",
		"Chassis":        link(root + "/Chassis"),
		"Managers":       link(root + "/Managers"),
		"SessionService": link(root + "/SessionService"),
		"Links":          obj{"Sessions": link(root + "/SessionService/Sessions")},
	}
	if s.isSwitch() {
		body["Product"] = "NVLink switch tray (fakebmc)"
		body["Fabrics"] = link(root + "/Fabrics")
	} else {
		body["Product"] = s.cfg.GPUName + " compute tray (fakebmc)"
		body["Systems"] = link(root + "/Systems")
	}
	writeJSON(w, 200, body)
}

func (s *Server) sessionService(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, obj{
		"@odata.id":      root + "/SessionService",
		"@odata.type":    "#SessionService.v1_1_9.SessionService",
		"Id":             "SessionService",
		"Name":           "Session Service",
		"ServiceEnabled": true,
		"SessionTimeout": 3600,
		"Sessions":       link(root + "/SessionService/Sessions"),
	})
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func (s *Server) login(w http.ResponseWriter, r *http.Request) {
	var req struct{ UserName, Password string }
	if !decode(w, r, &req) {
		return
	}
	if !s.validCredentials(req.UserName, req.Password) {
		redfishError(w, http.StatusUnauthorized, "ResourceAtUriUnauthorized", "Invalid username or password.")
		return
	}
	token, id := randomHex(16), randomHex(5)
	s.mu.Lock()
	s.sessions[token] = session{ID: id, User: req.UserName, Created: time.Now()}
	s.mu.Unlock()
	path := root + "/SessionService/Sessions/" + id
	w.Header().Set("X-Auth-Token", token)
	w.Header().Set("Location", path)
	writeJSON(w, http.StatusCreated, sessionObj(path, id, req.UserName))
}

func sessionObj(path, id, user string) obj {
	return obj{"@odata.id": path, "@odata.type": "#Session.v1_7_0.Session", "Id": id, "Name": "User Session", "UserName": user}
}

func (s *Server) findSession(id string) (string, session, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for tok, sess := range s.sessions {
		if sess.ID == id {
			return tok, sess, true
		}
	}
	return "", session{}, false
}

func (s *Server) sessionList(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	ids := make([]string, 0, len(s.sessions))
	for _, sess := range s.sessions {
		ids = append(ids, sess.ID)
	}
	s.mu.Unlock()
	writeJSON(w, 200, collection(root+"/SessionService/Sessions", "SessionCollection", "Session Collection", ids))
}

func (s *Server) sessionGet(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, sess, ok := s.findSession(id); ok {
		writeJSON(w, 200, sessionObj(r.URL.Path, id, sess.User))
		return
	}
	redfishError(w, http.StatusNotFound, "ResourceNotFound", "Session "+id+" was not found.")
}

func (s *Server) sessionDelete(w http.ResponseWriter, r *http.Request) {
	tok, _, ok := s.findSession(r.PathValue("id"))
	if !ok {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Session was not found.")
		return
	}
	s.mu.Lock()
	delete(s.sessions, tok)
	s.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

// ---- system and power ---------------------------------------------------------

func (s *Server) checkSystem(w http.ResponseWriter, r *http.Request) bool {
	if r.PathValue("sys") != systemID {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "System "+r.PathValue("sys")+" was not found.")
		return false
	}
	return true
}

func (s *Server) systems(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, collection(root+"/Systems", "ComputerSystemCollection", "Computer System Collection", []string{systemID}))
}

var resetTypes = []string{"On", "ForceOff", "GracefulShutdown", "GracefulRestart", "ForceRestart", "PowerCycle"}

func (s *Server) powerState() (string, obj) {
	state, err := s.power.PowerState()
	if err != nil {
		log.Printf("power state: %v", err)
		return "Unknown", obj{"State": "Enabled", "Health": "Warning"}
	}
	if state == "Off" {
		return state, obj{"State": "StandbyOffline", "Health": "OK"}
	}
	return state, obj{"State": "Enabled", "Health": "OK"}
}

func (s *Server) system(w http.ResponseWriter, r *http.Request) {
	if !s.checkSystem(w, r) {
		return
	}
	power, status := s.powerState()
	path := root + "/Systems/" + systemID
	writeJSON(w, 200, obj{
		"@odata.id":        path,
		"@odata.type":      "#ComputerSystem.v1_20_0.ComputerSystem",
		"Id":               systemID,
		"Name":             "System",
		"HostName":         s.cfg.Tray,
		"SystemType":       "Physical",
		"Manufacturer":     "NVIDIA",
		"Model":            s.cfg.GPUName + " compute tray",
		"PowerState":       power,
		"Status":           status,
		"ProcessorSummary": obj{"Count": s.cfg.GPUCount},
		"Processors":       link(path + "/Processors"),
		"Links": obj{
			"Chassis":   []obj{link(root + "/Chassis/" + chassisID)},
			"ManagedBy": []obj{link(root + "/Managers/" + managerID)},
		},
		"Actions": obj{"#ComputerSystem.Reset": obj{
			"target":                            path + "/Actions/ComputerSystem.Reset",
			"ResetType@Redfish.AllowableValues": resetTypes,
		}},
	})
}

func (s *Server) reset(w http.ResponseWriter, r *http.Request) {
	if !s.checkSystem(w, r) {
		return
	}
	var req struct{ ResetType string }
	if !decode(w, r, &req) {
		return
	}
	if req.ResetType == "" {
		req.ResetType = "GracefulRestart"
	}
	// Pending NVLink changes take effect whenever the tray (re)starts.
	powersOn := map[string]bool{"On": true, "GracefulRestart": true, "ForceRestart": true, "PowerCycle": true}
	if powersOn[req.ResetType] {
		if err := s.state.ApplyPending(); err != nil {
			redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
			return
		}
	}
	var err error
	switch req.ResetType {
	case "On":
		if st, _ := s.power.PowerState(); st != "On" {
			err = s.power.Action("start", false, 0)
		}
	case "ForceOff":
		err = s.power.Action("stop", true, 0)
	case "GracefulShutdown":
		err = s.power.Action("stop", false, 60)
	case "GracefulRestart":
		err = s.power.Action("restart", false, 60)
	case "ForceRestart", "PowerCycle":
		if st, _ := s.power.PowerState(); st == "On" {
			err = s.power.Action("restart", true, 0)
		} else {
			err = s.power.Action("start", false, 0)
		}
	default:
		redfishError(w, http.StatusBadRequest, "ActionParameterValueNotInList",
			fmt.Sprintf("The value %q for ResetType is not in the list of acceptable values %v.", req.ResetType, resetTypes))
		return
	}
	if err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ---- GPUs and NVLink ports ------------------------------------------------------

func (s *Server) gpuIndex(w http.ResponseWriter, r *http.Request) (int, bool) {
	if !s.checkSystem(w, r) {
		return 0, false
	}
	id := r.PathValue("gpu")
	n, err := strconv.Atoi(strings.TrimPrefix(id, "GPU_"))
	if !strings.HasPrefix(id, "GPU_") || err != nil || n < 0 || n >= s.cfg.GPUCount {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Processor "+id+" was not found.")
		return 0, false
	}
	return n, true
}

func (s *Server) linkKey(w http.ResponseWriter, r *http.Request) (LinkKey, bool) {
	g, ok := s.gpuIndex(w, r)
	if !ok {
		return LinkKey{}, false
	}
	id := r.PathValue("port")
	n, err := strconv.Atoi(strings.TrimPrefix(id, "NVLink_"))
	if !strings.HasPrefix(id, "NVLink_") || err != nil || n < 0 || n >= s.cfg.NVLinks {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Port "+id+" was not found.")
		return LinkKey{}, false
	}
	return LinkKey{g, n}, true
}

func (s *Server) processors(w http.ResponseWriter, r *http.Request) {
	if !s.checkSystem(w, r) {
		return
	}
	ids := make([]string, s.cfg.GPUCount)
	for i := range ids {
		ids[i] = fmt.Sprintf("GPU_%d", i)
	}
	writeJSON(w, 200, collection(root+"/Systems/"+systemID+"/Processors", "ProcessorCollection", "Processor Collection", ids))
}

func (s *Server) processor(w http.ResponseWriter, r *http.Request) {
	g, ok := s.gpuIndex(w, r)
	if !ok {
		return
	}
	path := fmt.Sprintf("%s/Systems/%s/Processors/GPU_%d", root, systemID, g)
	writeJSON(w, 200, obj{
		"@odata.id":     path,
		"@odata.type":   "#Processor.v1_20_0.Processor",
		"Id":            fmt.Sprintf("GPU_%d", g),
		"Name":          fmt.Sprintf("GPU %d", g),
		"ProcessorType": "GPU",
		"Manufacturer":  "NVIDIA",
		"Model":         s.cfg.GPUName,
		"UUID":          gpuUUID(s.cfg.Tray, g),
		"SerialNumber":  gpuSerial(s.cfg.Tray, g),
		"Status":        obj{"State": "Enabled", "Health": "OK"},
		"MemorySummary": obj{"TotalMemoryGiB": s.cfg.GPUMemMB / 1024},
		"Location":      obj{"PartLocation": obj{"LocationType": "Slot", "LocationOrdinalValue": g}},
		"Ports":         link(path + "/Ports"),
		"Links":         obj{"Chassis": link(root + "/Chassis/" + chassisID)},
		"Oem": obj{"Nvidia": obj{
			"@odata.type":  "#NvidiaProcessor.v1_4_0.NvidiaGPU",
			"PCIeBusId":    gpuPCIBusID(g),
			"FabricClique": obj{"ClusterUUID": s.cfg.ClusterUUID, "CliqueId": s.cfg.CliqueID},
		}},
	})
}

func (s *Server) ports(w http.ResponseWriter, r *http.Request) {
	g, ok := s.gpuIndex(w, r)
	if !ok {
		return
	}
	ids := make([]string, s.cfg.NVLinks)
	for i := range ids {
		ids[i] = fmt.Sprintf("NVLink_%d", i)
	}
	writeJSON(w, 200, collection(fmt.Sprintf("%s/Systems/%s/Processors/GPU_%d/Ports", root, systemID, g), "PortCollection", "NVLink Port Collection", ids))
}

func linkState(disabled bool) string {
	if disabled {
		return "Disabled"
	}
	return "Enabled"
}

func (s *Server) port(w http.ResponseWriter, r *http.Request) {
	k, ok := s.linkKey(w, r)
	if !ok {
		return
	}
	disabled, _ := s.state.LinkDisabled(k)
	power, _ := s.powerState()
	status, health := "LinkUp", "OK"
	if disabled || power != "On" {
		status = "LinkDown"
	}
	if disabled {
		health = "Warning"
	}
	path := fmt.Sprintf("%s/Systems/%s/Processors/GPU_%d/Ports/NVLink_%d", root, systemID, k.GPU, k.Link)
	writeJSON(w, 200, obj{
		"@odata.id":         path,
		"@odata.type":       "#Port.v1_11_0.Port",
		"Id":                fmt.Sprintf("NVLink_%d", k.Link),
		"Name":              fmt.Sprintf("NVLink %d", k.Link),
		"PortProtocol":      "NVLink",
		"PortType":          "BidirectionalPort",
		"LinkState":         linkState(disabled),
		"LinkStatus":        status,
		"Width":             2,
		"CurrentSpeedGbps":  200,
		"MaxSpeedGbps":      200,
		"Status":            obj{"State": map[bool]string{true: "Disabled", false: "Enabled"}[disabled], "Health": health},
		"@Redfish.Settings": obj{"SettingsObject": link(path + "/Settings")},
		"Oem": obj{"Nvidia": obj{
			"@odata.type":       "#NvidiaPort.v1_4_0.NvidiaNVLinkPort",
			"LinkDisableSticky": s.state.IsSticky(k),
			"TXWidth":           2,
			"RXWidth":           2,
		}},
	})
}

// Settings resource: the LinkState requested for the next tray reset.
func (s *Server) portSettings(w http.ResponseWriter, r *http.Request) {
	k, ok := s.linkKey(w, r)
	if !ok {
		return
	}
	current, pending := s.state.LinkDisabled(k)
	if pending != nil {
		current = *pending
	}
	writeJSON(w, 200, obj{
		"@odata.id":   r.URL.Path,
		"@odata.type": "#Port.v1_11_0.Port",
		"Id":          "Settings",
		"Name":        fmt.Sprintf("NVLink %d pending settings", k.Link),
		"LinkState":   linkState(current),
	})
}

func (s *Server) patchPortSettings(w http.ResponseWriter, r *http.Request) {
	k, ok := s.linkKey(w, r)
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
	if err := s.state.SetPending(k, req.LinkState == "Disabled"); err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// PATCH on the port itself: only Oem.Nvidia.LinkDisableSticky is writable.
func (s *Server) patchPort(w http.ResponseWriter, r *http.Request) {
	k, ok := s.linkKey(w, r)
	if !ok {
		return
	}
	var req struct {
		Oem struct {
			Nvidia struct {
				LinkDisableSticky *bool
			}
		}
	}
	if !decode(w, r, &req) {
		return
	}
	if req.Oem.Nvidia.LinkDisableSticky == nil {
		redfishError(w, http.StatusBadRequest, "PropertyNotWritable",
			"Only Oem.Nvidia.LinkDisableSticky is writable here; use the Settings resource for LinkState.")
		return
	}
	if err := s.state.SetSticky(k, *req.Oem.Nvidia.LinkDisableSticky); err != nil {
		redfishError(w, http.StatusInternalServerError, "InternalError", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ---- chassis and manager ---------------------------------------------------------

func (s *Server) chassisList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, collection(root+"/Chassis", "ChassisCollection", "Chassis Collection", []string{s.chassisID()}))
}

func (s *Server) chassis(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("ch") != s.chassisID() {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Chassis "+r.PathValue("ch")+" was not found.")
		return
	}
	if s.isSwitch() {
		writeJSON(w, 200, obj{
			"@odata.id":    root + "/Chassis/" + s.chassisID(),
			"@odata.type":  "#Chassis.v1_23_0.Chassis",
			"Id":           s.chassisID(),
			"Name":         "NVLink switch tray",
			"ChassisType":  "Sled",
			"Manufacturer": "NVIDIA",
			"Model":        "NVLink5 switch tray",
			"PowerState":   "On",
			"Status":       obj{"State": "Enabled", "Health": "OK"},
			"Links":        obj{"ManagedBy": []obj{link(root + "/Managers/" + managerID)}},
		})
		return
	}
	power, status := s.powerState()
	writeJSON(w, 200, obj{
		"@odata.id":    root + "/Chassis/" + chassisID,
		"@odata.type":  "#Chassis.v1_23_0.Chassis",
		"Id":           chassisID,
		"Name":         s.cfg.Tray,
		"ChassisType":  "Sled",
		"Manufacturer": "NVIDIA",
		"Model":        s.cfg.GPUName + " compute tray",
		"PowerState":   power,
		"Status":       status,
		"Links": obj{
			"ComputerSystems": []obj{link(root + "/Systems/" + systemID)},
			"ManagedBy":       []obj{link(root + "/Managers/" + managerID)},
		},
	})
}

func (s *Server) managers(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, collection(root+"/Managers", "ManagerCollection", "Manager Collection", []string{managerID}))
}

func (s *Server) manager(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("mgr") != managerID {
		redfishError(w, http.StatusNotFound, "ResourceNotFound", "Manager "+r.PathValue("mgr")+" was not found.")
		return
	}
	path := root + "/Managers/" + managerID
	writeJSON(w, 200, obj{
		"@odata.id":       path,
		"@odata.type":     "#Manager.v1_19_0.Manager",
		"Id":              managerID,
		"Name":            "OpenBMC Manager",
		"ManagerType":     "BMC",
		"FirmwareVersion": "fakebmc-" + version,
		"Status":          obj{"State": "Enabled", "Health": "OK"},
		"Links":           s.managerLinks(),
		"Actions": obj{"#Manager.Reset": obj{
			"target":                            path + "/Actions/Manager.Reset",
			"ResetType@Redfish.AllowableValues": []string{"GracefulRestart", "ForceRestart"},
		}},
	})
}

func (s *Server) managerLinks() obj {
	l := obj{"ManagerForChassis": []obj{link(root + "/Chassis/" + s.chassisID())}}
	if !s.isSwitch() {
		l["ManagerForServers"] = []obj{link(root + "/Systems/" + systemID)}
	}
	return l
}

// A BMC reset drops all sessions; the tray keeps running.
func (s *Server) managerReset(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.sessions = map[string]session{}
	s.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}
