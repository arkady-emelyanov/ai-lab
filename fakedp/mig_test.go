package main

import "testing"

func TestParseInventory(t *testing.T) {
	modes := []byte("GPU-aaa, Disabled\nGPU-bbb, Enabled\n")
	list := []byte(`GPU 0: NVIDIA GB200 (UUID: GPU-aaa)
GPU 1: NVIDIA GB200 (UUID: GPU-bbb)
  MIG 3g.93gb     Device  0: (UUID: MIG-111)
  MIG 1g.23gb+me  Device  1: (UUID: MIG-222)
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
	if cr.Envs["NVIDIA_VISIBLE_DEVICES"] != "MIG-1,MIG-2" || len(cr.CdiDevices) != 1 || cr.CdiDevices[0].Name != "nvidia.com/gpu=GPU-bbb" {
		t.Fatalf("%v %v", cr.Envs, cr.CdiDevices)
	}
	if _, err := p.allocate([]string{"MIG-3"}); err == nil {
		t.Fatal("a 3g.93gb from the 1g.23gb plugin")
	}
	g := gpuPlugin([]gpu{{"GPU-aaa", "/dev/null"}, {"GPU-bbb", "/dev/null"}}, inv)
	if d := g.devices(); len(d) != 1 || d[0].ID != "GPU-aaa" {
		t.Fatalf("whole GPUs: %v", d)
	}
}
