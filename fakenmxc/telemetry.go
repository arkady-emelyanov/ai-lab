package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
)

// Fabric telemetry in Prometheus text format (the role NVIDIA's NMX-T plays
// next to NMX-C): partitions, per-GPU NVLink state and traffic, per-switch
// port state. GPU traffic and utilisation come from each tray's occupancy
// file (the fake GPU stack's shared state), mounted read-only on this host.

// gpuState mirrors struct fg_gpu_state in fakegpu/occupancy.h.
type gpuState struct {
	BusyNs, SampleNs, SampleBusy uint64
	Util, TempMc                 uint32
	TempNs, NVLinkTx, NVLinkRx   uint64
}

const maxTrayGPUs = 8 // FG_MAX_GPUS

func readTelemetry(path string) ([]gpuState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	states := make([]gpuState, maxTrayGPUs)
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, states); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return states, nil
}

type metricWriter struct{ b strings.Builder }

func (m *metricWriter) header(name, typ, help string) {
	fmt.Fprintf(&m.b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, typ)
}

func (m *metricWriter) sample(name string, labels map[string]string, v float64) {
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf("%s=%q", k, labels[k])
	}
	fmt.Fprintf(&m.b, "%s{%s} %g\n", name, strings.Join(parts, ","), v)
}

func (c *controller) serveMetrics(w http.ResponseWriter, _ *http.Request) {
	c.mu.Lock()
	defer c.mu.Unlock()
	var m metricWriter
	per := c.cfg.NVLinks / c.cfg.Switches

	m.header("nvlink_domain_info", "gauge", "NVLink domain of this switch tray.")
	m.sample("nvlink_domain_info", map[string]string{"domain": c.cfg.DomainName, "cluster_uuid": c.cfg.DomainUUID}, 1)

	m.header("nvlink_partition_gpus", "gauge", "GPUs in each NVLink partition.")
	for _, id := range c.sortedIDs() {
		p := c.Partitions[id]
		m.sample("nvlink_partition_gpus", map[string]string{"partition_id": fmt.Sprint(id), "name": p.Name,
			"default": fmt.Sprint(id == defaultPartition)}, float64(len(p.GPUs)))
	}

	telemetry := make([][]gpuState, len(c.cfg.Trays))
	for t, tray := range c.cfg.Trays {
		if tray.TelemetryPath != "" {
			telemetry[t], _ = readTelemetry(tray.TelemetryPath) // tray down or not yet used: no traffic series
		}
	}

	gpuMetrics := []struct{ name, typ, help string }{
		{"nvlink_gpu_partition_id", "gauge", "NVLink partition of the GPU (0 = none)."},
		{"nvlink_gpu_clique_id", "gauge", "NVLink clique the GPU reports (NVML fabric info)."},
		{"nvlink_gpu_links", "gauge", "NVLinks of the GPU."},
		{"nvlink_gpu_active_links", "gauge", "NVLinks of the GPU that are up."},
		{"nvlink_gpu_healthy", "gauge", "1 when all NVLinks of the GPU are up."},
		{"nvlink_gpu_tx_bytes_total", "counter", "Bytes the GPU sent over NVLink."},
		{"nvlink_gpu_rx_bytes_total", "counter", "Bytes the GPU received over NVLink."},
		{"nvlink_gpu_busy_seconds_total", "counter", "Time the GPU spent executing work."},
	}
	type gpuRow struct {
		labels map[string]string
		values []float64
		hasTel bool
	}
	var rows []gpuRow
	switchUp := make([]int, c.cfg.Switches)
	switchTx := make([]float64, c.cfg.Switches)
	switchRx := make([]float64, c.cfg.Switches)
	type portRow struct {
		labels map[string]string
		up     float64
	}
	var ports []portRow

	for gi, g := range c.gpus {
		info := c.gpuInfo(g)
		down := c.disabledLinks(g)
		labels := map[string]string{"host": info.Hostname, "gpu": fmt.Sprint(g.idx), "uuid": g.uuid,
			"slot": fmt.Sprint(info.Location.SlotId)}
		row := gpuRow{labels: labels, values: []float64{float64(info.PartitionId), float64(info.CliqueId),
			float64(c.cfg.NVLinks), float64(info.ActiveNvlinks), b2f(int(info.ActiveNvlinks) == c.cfg.NVLinks), 0, 0, 0}}
		var tel *gpuState
		if ts := telemetry[g.tray]; ts != nil && g.idx < len(ts) {
			tel = &ts[g.idx]
			row.values[5], row.values[6] = float64(tel.NVLinkTx), float64(tel.NVLinkRx)
			row.values[7] = float64(tel.BusyNs) / 1e9
			row.hasTel = true
		}
		rows = append(rows, row)

		// Ports: GPU link l -> switch l%S, port g*(L/S) + l/S. Traffic is
		// attributed to the GPU's active links evenly.
		activeOn := make([]int, c.cfg.Switches)
		for l := 0; l < c.cfg.NVLinks; l++ {
			sw, port := l%c.cfg.Switches, gi*per+l/c.cfg.Switches
			up := !down[l]
			if up {
				switchUp[sw]++
				activeOn[sw]++
			}
			ports = append(ports, portRow{map[string]string{"switch": fmt.Sprintf("NVSwitch_%d", sw),
				"port": fmt.Sprint(port), "host": info.Hostname, "gpu": fmt.Sprint(g.idx), "link": fmt.Sprint(l)}, b2f(up)})
		}
		if tel != nil && info.ActiveNvlinks > 0 {
			for sw := range activeOn {
				share := float64(activeOn[sw]) / float64(info.ActiveNvlinks)
				switchTx[sw] += float64(tel.NVLinkTx) * share
				switchRx[sw] += float64(tel.NVLinkRx) * share
			}
		}
	}

	for i, gm := range gpuMetrics {
		m.header(gm.name, gm.typ, gm.help)
		for _, r := range rows {
			if i >= 5 && !r.hasTel {
				continue
			}
			m.sample(gm.name, r.labels, r.values[i])
		}
	}

	m.header("nvswitch_port_up", "gauge", "1 when the switch port and its GPU link are up; labels give the cabling.")
	for _, p := range ports {
		m.sample("nvswitch_port_up", p.labels, p.up)
	}
	m.header("nvswitch_ports", "gauge", "Ports of the switch.")
	m.header("nvswitch_ports_up", "gauge", "Cabled ports of the switch that are up.")
	m.header("nvswitch_tx_bytes_total", "counter", "Bytes from GPUs carried by the switch (GPU traffic split over its active links).")
	m.header("nvswitch_rx_bytes_total", "counter", "Bytes to GPUs carried by the switch.")
	for sw := 0; sw < c.cfg.Switches; sw++ {
		l := map[string]string{"switch": fmt.Sprintf("NVSwitch_%d", sw), "host": c.cfg.SwitchHost}
		m.sample("nvswitch_ports", l, float64(c.cfg.SwitchPorts))
		m.sample("nvswitch_ports_up", l, float64(switchUp[sw]))
		m.sample("nvswitch_tx_bytes_total", l, switchTx[sw])
		m.sample("nvswitch_rx_bytes_total", l, switchRx[sw])
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(m.b.String()))
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
