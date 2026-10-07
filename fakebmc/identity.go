package main

import "fmt"

// GPU identities computed exactly like fakegpu/fakegpu.h, so UUIDs, serials
// and PCI addresses seen over Redfish match nvidia-smi on the tray.

var pciBus = [8]uint{0x18, 0x2A, 0x3A, 0x5D, 0x9A, 0xAB, 0xBA, 0xDB}

func gpuUUIDBytes(host string, idx int) [16]byte {
	h := uint64(1469598103934665603) // FNV-1a
	for i := 0; i < len(host); i++ {
		h = (h ^ uint64(host[i])) * 1099511628211
	}
	var out [16]byte
	for i := 0; i < 16; i++ {
		h = (h ^ uint64(idx*16+i)) * 1099511628211
		out[i] = byte(h >> 56)
	}
	return out
}

// gpuUUID returns the UUID without nvidia-smi's "GPU-" prefix.
func gpuUUID(host string, idx int) string {
	u := gpuUUIDBytes(host, idx)
	return fmt.Sprintf("%x-%x-%x-%x-%x", u[0:4], u[4:6], u[6:8], u[8:10], u[10:16])
}

func gpuSerial(host string, idx int) string {
	u := gpuUUIDBytes(host, idx)
	return fmt.Sprintf("16523240%02d%02d%02d", u[0]%100, u[1]%100, u[2]%100)
}

func gpuPCIBusID(idx int) string {
	return fmt.Sprintf("%08X:%02X:00.0", 0, pciBus[idx])
}

// traySerial is the tray's (or switch tray's) serial number, stable per name.
func traySerial(host string) string {
	u := gpuUUIDBytes(host, 255)
	return fmt.Sprintf("1821220%02d%02d%02d", u[3]%100, u[4]%100, u[5]%100)
}
