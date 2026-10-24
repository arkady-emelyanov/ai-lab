package main

import (
	"sort"
	"sync"
	"testing"
)

func TestParseInventory(t *testing.T) {
	modes := []byte("GPU-aaa, Disabled\nGPU-bbb, Enabled\n")
	list := []byte(`GPU 0: NVIDIA GB200 (UUID: GPU-aaa)
GPU 1: NVIDIA GB200 (UUID: GPU-bbb)
  MIG 3g.93gb     Device  0: (UUID: MIG-111)
  MIG 1g.23gb+me  Device  1: (UUID: MIG-222)
  MIG 1c.3g.93gb  Device  2: (UUID: MIG-333)
`)
	inv, err := parseInventory(modes, list)
	if err != nil {
		t.Fatal(err)
	}
	if inv.enabled["GPU-aaa"] || !inv.enabled["GPU-bbb"] {
		t.Fatalf("modes: %v", inv.enabled)
	}
	want := []migDevice{{"MIG-111", "GPU-bbb", "3g.93gb"}, {"MIG-222", "GPU-bbb", "1g.23gb+me"}}
	if len(inv.devices) != 2 || inv.devices[0] != want[0] || inv.devices[1] != want[1] {
		t.Fatalf("devices: %v", inv.devices)
	}
	if got := migResource("1g.23gb+me"); got != "nvidia.com/mig-1g.23gb.me" {
		t.Fatalf("resource %q", got)
	}
}

func TestMigPluginAllocatesParentAndUUIDs(t *testing.T) {
	inv := &inventoryCache{inv: inventory{enabled: map[string]bool{"GPU-bbb": true},
		devices: []migDevice{{"MIG-1", "GPU-bbb", "1g.23gb"}, {"MIG-2", "GPU-bbb", "1g.23gb"}, {"MIG-3", "GPU-bbb", "3g.93gb"}}}}
	p := migPlugin("1g.23gb", nil, inv)
	if p.resource != "nvidia.com/mig-1g.23gb" || len(p.devices()) != 2 {
		t.Fatalf("%s %v", p.resource, p.devices())
	}
	cr, err := p.allocate([]string{"MIG-1", "MIG-2"})
	if err != nil {
		t.Fatal(err)
	}
	if cr.Envs["NVIDIA_VISIBLE_DEVICES"] != deviceListDir || len(cr.CdiDevices) != 1 || cr.CdiDevices[0].Name != "nvidia.com/gpu=GPU-bbb" {
		t.Fatalf("%v %v", cr.Envs, cr.CdiDevices)
	}
	// A file per MIG device: kubelet adds up mounts of several resources.
	if len(cr.Mounts) != 2 || cr.Mounts[0].ContainerPath != "/var/run/nvidia-container-devices/MIG-1" ||
		cr.Mounts[1].ContainerPath != "/var/run/nvidia-container-devices/MIG-2" ||
		cr.Mounts[0].HostPath != "/dev/null" || !cr.Mounts[0].ReadOnly {
		t.Fatalf("mounts %v", cr.Mounts)
	}
	if _, err := p.allocate([]string{"MIG-3"}); err == nil {
		t.Fatal("a 3g.93gb from the 1g.23gb plugin")
	}
	g := gpuPlugin([]gpu{{"GPU-aaa", "/dev/null"}, {"GPU-bbb", "/dev/null"}}, inv)
	if d := g.devices(); len(d) != 1 || d[0].ID != "GPU-aaa" {
		t.Fatalf("whole GPUs: %v", d)
	}
	if _, err := g.allocate([]string{"GPU-bbb"}); err == nil {
		t.Fatal("a GPU with MIG on allocated whole")
	}
	if _, err := g.allocate([]string{"GPU-aaa"}); err != nil {
		t.Fatal(err)
	}
}

func TestProfilePluginsStartAndStop(t *testing.T) {
	started := make(chan string, 10)
	var mu sync.Mutex
	stopped := map[string]bool{}
	pp := &profilePlugins{start: func(profile string, stop <-chan struct{}) {
		started <- profile
		<-stop
		mu.Lock()
		stopped[profile] = true
		mu.Unlock()
	}}
	pp.sync([]string{"1g.23gb", "3g.93gb"})
	got := []string{<-started, <-started}
	sort.Strings(got)
	if got[0] != "1g.23gb" || got[1] != "3g.93gb" {
		t.Fatalf("started %v", got)
	}
	pp.sync([]string{"1g.23gb", "3g.93gb"}) // running already
	pp.sync([]string{"3g.93gb"})            // returns once 1g.23gb stopped
	mu.Lock()
	if !stopped["1g.23gb"] || stopped["3g.93gb"] || len(started) != 0 || len(pp.running) != 1 {
		t.Fatalf("stopped %v, started again %d, running %v", stopped, len(started), pp.running)
	}
	mu.Unlock()
	pp.sync([]string{"1g.23gb"}) // back: a new plugin
	if p := <-started; p != "1g.23gb" || !stopped["3g.93gb"] {
		t.Fatalf("started %s, stopped %v", p, stopped)
	}
	pp.sync(nil)
}
