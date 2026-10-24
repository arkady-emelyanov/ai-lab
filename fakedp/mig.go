package main

// MIG devices for the device plugin (main.go): the tray's MIG layout as
// nvidia-smi shows it, and a plugin per MIG profile.

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

// migDevice is a MIG device: its UUID (the device ID kubelet sees, as with
// NVIDIA's plugin), its parent GPU's UUID and its profile (1g.23gb, ...).
type migDevice struct {
	UUID, Parent, Profile string
}

type inventory struct {
	enabled map[string]bool // GPU UUID: MIG on
	devices []migDevice
}

func (inv inventory) profiles() []string {
	seen := map[string]bool{}
	var out []string
	for _, d := range inv.devices {
		if !seen[d.Profile] {
			seen[d.Profile] = true
			out = append(out, d.Profile)
		}
	}
	sort.Strings(out)
	return out
}

var (
	gpuLine = regexp.MustCompile(`^GPU \d+: .* \(UUID: (GPU-[0-9a-f-]+)\)$`)
	migLine = regexp.MustCompile(`^\s+MIG (\S+)\s+Device\s+\d+: \(UUID: (MIG-[0-9a-f-]+)\)$`)
	// A compute instance smaller than its GPU instance (1c.3g.93gb, ...).
	partialCI = regexp.MustCompile(`^\d+c\.`)
)

// parseInventory reads nvidia-smi's MIG mode query and -L listing. MIG
// devices whose compute instance does not fill its GPU instance are left
// out: NVIDIA's plugin has no resource for them (its mixed strategy names
// only the GPU instance profiles) and refuses to start with one; here they
// are only not offered.
func parseInventory(modes, list []byte) (inventory, error) {
	inv := inventory{enabled: map[string]bool{}}
	for _, line := range strings.Split(strings.TrimSpace(string(modes)), "\n") {
		f := strings.Split(line, ", ")
		if len(f) != 2 {
			return inv, fmt.Errorf("unexpected MIG mode line %q", line)
		}
		inv.enabled[f[0]] = f[1] == "Enabled"
	}
	parent := ""
	sc := bufio.NewScanner(bytes.NewReader(list))
	for sc.Scan() {
		if m := gpuLine.FindStringSubmatch(sc.Text()); m != nil {
			parent = m[1]
		} else if m := migLine.FindStringSubmatch(sc.Text()); m != nil && parent != "" && !partialCI.MatchString(m[1]) {
			inv.devices = append(inv.devices, migDevice{UUID: m[2], Parent: parent, Profile: m[1]})
		}
	}
	return inv, nil
}

// inventoryCache keeps the last MIG layout nvidia-smi gave (a failing run
// keeps the previous one).
type inventoryCache struct {
	mu  sync.Mutex
	inv inventory
}

func (c *inventoryCache) get() inventory {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.inv
}

// update reads the MIG layout and reports whether it could.
func (c *inventoryCache) update() bool {
	modes, err := exec.Command("nvidia-smi", "--query-gpu=uuid,mig.mode.current", "--format=csv,noheader").Output()
	if err != nil {
		log.Printf("nvidia-smi: %v", err)
		return false
	}
	list, err := exec.Command("nvidia-smi", "-L").Output()
	if err != nil {
		log.Printf("nvidia-smi -L: %v", err)
		return false
	}
	inv, err := parseInventory(modes, list)
	if err != nil {
		log.Printf("%v", err)
		return false
	}
	c.mu.Lock()
	c.inv = inv
	c.mu.Unlock()
	return true
}

// migResource is the resource name of a MIG profile, as NVIDIA's plugin
// and GPU Feature Discovery name it.
func migResource(profile string) string {
	return "nvidia.com/mig-" + strings.ReplaceAll(profile, "+", ".")
}

// deviceListDir is where NVIDIA's plugin, with its volume-mounts device
// list strategy, mounts /dev/null under each allocated device's ID.
const deviceListDir = "/var/run/nvidia-container-devices"

// migPlugin serves nvidia.com/mig-<profile>: the tray's MIG devices of a
// profile. A container gets their parent GPUs (CDI) and their UUIDs as
// NVIDIA's volume-mounts strategy gives them: a file per device in
// deviceListDir, NVIDIA_VISIBLE_DEVICES naming the directory. Unlike a
// device list in NVIDIA_VISIBLE_DEVICES (kubelet keeps one value when
// several resources set it), the files add up: a container can have MIG
// devices of several profiles and whole GPUs.
func migPlugin(profile string, gpus []gpu, inv *inventoryCache) *plugin {
	name := strings.ReplaceAll(profile, "+", ".")
	return &plugin{
		resource: migResource(profile),
		socket:   "fakegpu-mig-" + name + ".sock",
		devices: func() []*pluginapi.Device {
			var devs []*pluginapi.Device
			for _, d := range inv.get().devices {
				if d.Profile == profile {
					devs = append(devs, &pluginapi.Device{ID: d.UUID, Health: pluginapi.Healthy})
				}
			}
			return devs
		},
		allocate: func(ids []string) (*pluginapi.ContainerAllocateResponse, error) {
			parents := map[string]string{}
			for _, d := range inv.get().devices {
				if d.Profile == profile {
					parents[d.UUID] = d.Parent
				}
			}
			cr := &pluginapi.ContainerAllocateResponse{
				Envs: map[string]string{"NVIDIA_VISIBLE_DEVICES": deviceListDir},
			}
			added := map[string]bool{}
			for _, id := range ids {
				parent, ok := parents[id]
				if !ok {
					return nil, fmt.Errorf("unknown MIG device %q", id)
				}
				cr.Mounts = append(cr.Mounts, &pluginapi.Mount{
					ContainerPath: filepath.Join(deviceListDir, id), HostPath: "/dev/null", ReadOnly: true})
				if !added[parent] {
					added[parent] = true
					cr.CdiDevices = append(cr.CdiDevices, &pluginapi.CDIDevice{Name: cdiKind + "=" + parent})
				}
			}
			return cr, nil
		},
	}
}

// profilePlugins runs a plugin per MIG profile the tray has, and stops it
// once the profile is gone: kubelet then drops the resource from the node
// after its grace period (5 minutes), as when NVIDIA's plugin restarts
// without it.
type profilePlugins struct {
	start   func(profile string, stop <-chan struct{}) // returns when stopped
	running map[string]profileRun
}

type profileRun struct{ stop, done chan struct{} }

// sync starts the plugins of new profiles and stops those of profiles gone.
func (pp *profilePlugins) sync(profiles []string) {
	if pp.running == nil {
		pp.running = map[string]profileRun{}
	}
	current := map[string]bool{}
	for _, profile := range profiles {
		current[profile] = true
		if _, ok := pp.running[profile]; !ok {
			r := profileRun{make(chan struct{}), make(chan struct{})}
			pp.running[profile] = r
			go func() {
				defer close(r.done)
				pp.start(profile, r.stop)
			}()
		}
	}
	for profile, r := range pp.running {
		if !current[profile] {
			close(r.stop)
			<-r.done // its socket is gone before a new one may take the name
			delete(pp.running, profile)
			log.Printf("no %s devices left; stopped its plugin", migResource(profile))
		}
	}
}
