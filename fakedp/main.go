// fakedp is the GPU device plugin for the lab's Kubernetes mode: it
// advertises a tray's fake GPUs to kubelet as nvidia.com/gpu and hands each
// container the GPUs it was allocated as CDI devices.
//
// MIG, as NVIDIA's plugin does with its mixed strategy: a GPU with MIG on is
// not advertised as nvidia.com/gpu; its MIG devices are, as
// nvidia.com/mig-<profile> (nvidia.com/mig-1g.23gb, ...; + becomes .), the
// names GPU Feature Discovery labels nodes with. A container given MIG
// devices gets their parent GPU's CDI device and NVIDIA_VISIBLE_DEVICES set
// to their MIG- UUIDs, which the fake driver shows it only (in place of the
// MIG capability nodes the real driver checks). The MIG layout is read from
// nvidia-smi every 10 seconds (mig.go).
//
// NVIDIA's k8s-device-plugin cannot do this here: in CDI mode it generates
// its own CDI specification from a real driver installation (versioned
// libraries, nvidia-cdi-hook). The tray's specification is written by
// Ansible instead (/etc/cdi/fakegpu.json: one device per GPU UUID, plus the
// fake driver's libraries and configuration) and this plugin reads the GPUs
// from it. The rest of NVIDIA's stack (GPU Feature Discovery, node labels)
// runs unchanged on top of the fake NVML.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	resourceName = "nvidia.com/gpu"
	cdiKind      = "nvidia.com/gpu"
	socketName   = "fakegpu.sock"
)

var refresh = 10 * time.Second

// gpu is one device from the CDI specification: its UUID (the device ID
// kubelet sees, as with NVIDIA's plugin) and its device node.
type gpu struct {
	UUID, Node string
}

// readSpec returns the GPUs of a CDI specification: devices named by UUID,
// each with the /dev/nvidia<n> node CDI injects for it.
func readSpec(path string) ([]gpu, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var spec struct {
		Kind    string `json:"kind"`
		Devices []struct {
			Name           string `json:"name"`
			ContainerEdits struct {
				DeviceNodes []struct{ Path string } `json:"deviceNodes"`
			} `json:"containerEdits"`
		} `json:"devices"`
	}
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if spec.Kind != cdiKind {
		return nil, fmt.Errorf("%s: kind %q, want %q", path, spec.Kind, cdiKind)
	}
	var gpus []gpu
	for _, d := range spec.Devices {
		if strings.HasPrefix(d.Name, "GPU-") && len(d.ContainerEdits.DeviceNodes) == 1 {
			gpus = append(gpus, gpu{UUID: d.Name, Node: d.ContainerEdits.DeviceNodes[0].Path})
		}
	}
	sort.Slice(gpus, func(i, j int) bool { return gpus[i].Node < gpus[j].Node })
	if len(gpus) == 0 {
		return nil, fmt.Errorf("%s: no GPU devices", path)
	}
	return gpus, nil
}

// plugin serves one resource: its devices, and what a container
// allocated some of them gets.
type plugin struct {
	pluginapi.UnimplementedDevicePluginServer
	resource, socket string
	devices          func() []*pluginapi.Device
	allocate         func(ids []string) (*pluginapi.ContainerAllocateResponse, error)
	// watching counts kubelet's open ListAndWatch streams: without one,
	// kubelet reports no devices on the node.
	watching atomic.Int32
}

func (p *plugin) GetDevicePluginOptions(context.Context, *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{}, nil
}

// ListAndWatch sends the device list, then again every 10 seconds: health
// changes (a device node appearing or disappearing) and MIG changes reach
// kubelet, and so does the full list if kubelet lost it.
func (p *plugin) ListAndWatch(_ *pluginapi.Empty, stream pluginapi.DevicePlugin_ListAndWatchServer) error {
	p.watching.Add(1)
	defer p.watching.Add(-1)
	for {
		if err := stream.Send(&pluginapi.ListAndWatchResponse{Devices: p.devices()}); err != nil {
			return err
		}
		select {
		case <-stream.Context().Done():
			return nil
		case <-time.After(refresh):
		}
	}
}

func (p *plugin) Allocate(_ context.Context, req *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	resp := &pluginapi.AllocateResponse{}
	for _, c := range req.ContainerRequests {
		cr, err := p.allocate(c.DevicesIds)
		if err != nil {
			return nil, err
		}
		resp.ContainerResponses = append(resp.ContainerResponses, cr)
	}
	return resp, nil
}

func (p *plugin) GetPreferredAllocation(context.Context, *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	return &pluginapi.PreferredAllocationResponse{}, nil
}

func (p *plugin) PreStartContainer(context.Context, *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	return &pluginapi.PreStartContainerResponse{}, nil
}

// gpuPlugin serves nvidia.com/gpu: the GPUs without MIG.
func gpuPlugin(gpus []gpu, inv *inventoryCache) *plugin {
	return &plugin{
		resource: resourceName,
		socket:   socketName,
		devices: func() []*pluginapi.Device {
			var devs []*pluginapi.Device
			mig := inv.get()
			for _, g := range gpus {
				if mig.enabled[g.UUID] {
					continue // its MIG devices are advertised instead
				}
				health := pluginapi.Healthy
				if _, err := os.Stat(g.Node); err != nil {
					health = pluginapi.Unhealthy
				}
				devs = append(devs, &pluginapi.Device{ID: g.UUID, Health: health})
			}
			return devs
		},
		allocate: func(ids []string) (*pluginapi.ContainerAllocateResponse, error) {
			known := map[string]bool{}
			for _, g := range gpus {
				known[g.UUID] = true
			}
			cr := &pluginapi.ContainerAllocateResponse{
				Envs: map[string]string{"NVIDIA_VISIBLE_DEVICES": strings.Join(ids, ",")},
			}
			for _, id := range ids {
				if !known[id] {
					return nil, fmt.Errorf("unknown device %q", id)
				}
				cr.CdiDevices = append(cr.CdiDevices, &pluginapi.CDIDevice{Name: cdiKind + "=" + id})
			}
			return cr, nil
		},
	}
}

// inode identifies a socket file, to notice it being recreated.
func inode(path string) uint64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		return st.Ino
	}
	return 0
}

// serve runs the plugin's gRPC server and registers it with kubelet. It
// returns when kubelet restarts (it recreates its socket, and may remove
// ours) or stops watching the GPUs, so the caller can register again, as
// kubelet requires. A kubelet that accepts the registration while it is
// still starting may never open ListAndWatch, and its stream can end
// without a restart; either way the node shows no GPUs until the plugin
// registers again.
func serve(dir string, p *plugin) error {
	kubeletSock := filepath.Join(dir, "kubelet.sock")
	sock := filepath.Join(dir, p.socket)
	_ = os.Remove(sock)
	lis, err := net.Listen("unix", sock)
	if err != nil {
		return err
	}
	srv := grpc.NewServer()
	pluginapi.RegisterDevicePluginServer(srv, p)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()

	kubelet := inode(kubeletSock)
	conn, err := grpc.NewClient("unix://"+kubeletSock,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := pluginapi.NewRegistrationClient(conn).Register(ctx, &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     p.socket,
		ResourceName: p.resource,
		Options:      &pluginapi.DevicePluginOptions{},
	}); err != nil {
		return fmt.Errorf("register with kubelet: %w", err)
	}
	log.Printf("registered %d devices as %s", len(p.devices()), p.resource)

	unwatched := time.Now()
	for {
		time.Sleep(5 * time.Second)
		if _, err := os.Stat(sock); err != nil || inode(kubeletSock) != kubelet {
			log.Printf("kubelet restarted; registering again")
			return nil
		}
		if p.watching.Load() > 0 {
			unwatched = time.Now()
		} else if time.Since(unwatched) > 30*time.Second {
			log.Printf("kubelet is not watching %s; registering again", p.resource)
			return nil
		}
	}
}

func main() {
	specPath := flag.String("cdi-spec", "/etc/cdi/fakegpu.json", "CDI specification of the tray's GPUs")
	dir := flag.String("plugin-dir", pluginapi.DevicePluginPath, "kubelet device plugin directory")
	nri := flag.Bool("nri", false, "also add management CDI devices requested by pod annotations (NRI plugin, nri.go)")
	mgmt := flag.String("management-namespaces", "", "comma-separated namespaces allowed management devices (-nri)")
	flag.Parse()

	if *nri {
		go runNRI(strings.Split(*mgmt, ","))
	}

	gpus, err := readSpec(*specPath)
	if err != nil {
		log.Fatal(err)
	}
	inv := &inventoryCache{}
	inv.update()
	go run(*dir, gpuPlugin(gpus, inv))
	// A plugin per MIG profile, from when the tray first has one.
	started := map[string]bool{}
	for {
		for _, profile := range inv.get().profiles() {
			if !started[profile] {
				started[profile] = true
				go run(*dir, migPlugin(profile, gpus, inv))
			}
		}
		time.Sleep(refresh)
		inv.update()
	}
}

// run serves a plugin, registering it again whenever serve returns.
func run(dir string, p *plugin) {
	for {
		if err := serve(dir, p); err != nil {
			log.Printf("%s: %v; retrying", p.resource, err)
			time.Sleep(5 * time.Second)
		}
	}
}
