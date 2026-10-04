// fakenmxc is a lab NVLink partition controller modelled on NVIDIA's NMX-C:
// it owns the NVLink domain's partitions and publishes each GPU's partition
// (clique) to its tray, where the fake NVML reports it (fabric.cliqueId in
// nvidia-smi, nvmlDeviceGetGpuFabricInfo). gRPC with server reflection, so
// grpcurl can explore it. It also serves fabric telemetry for Prometheus
// (NMX-T's role): partitions, NVLink state and traffic, switch ports.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"net"
	"net/http"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	pb "fakenmxc/gen/nmxlabv1"
)

const version = "1.0.0"

type Config struct {
	Listen        string `json:"listen"`
	MetricsListen string `json:"metrics_listen"`
	DomainName    string `json:"domain_name"`
	DomainUUID    string `json:"domain_uuid"`
	DefaultClique uint32 `json:"default_clique"`
	GPUCount      int    `json:"gpu_count"`
	NVLinks       int    `json:"nvlinks"`
	Switches      int    `json:"switches"`
	SwitchPorts   int    `json:"switch_ports"`
	SwitchHost    string `json:"switch_host"`
	StateDir      string `json:"state_dir"`
	Trays         []Tray `json:"trays"`
}

// Tray is a GPU compute tray of the domain: its hostname, slot in the
// chassis and the sideband directory shared with it.
type Tray struct {
	Name          string `json:"name"`
	Slot          uint32 `json:"slot"`
	SidebandDir   string `json:"sideband_dir"`
	TelemetryPath string `json:"telemetry_path"` // the tray's GPU occupancy file (read-only)
}

func main() {
	path := flag.String("config", "/etc/fakenmxc/config.json", "configuration file")
	flag.Parse()
	data, err := os.ReadFile(*path)
	if err != nil {
		log.Fatal(err)
	}
	cfg := &Config{Listen: ":9370", MetricsListen: ":9372", DefaultClique: 1, GPUCount: 4, NVLinks: 18, Switches: 2, SwitchPorts: 72,
		StateDir: "/var/lib/fakenmxc"}
	if err := json.Unmarshal(data, cfg); err != nil {
		log.Fatalf("%s: %v", *path, err)
	}
	if len(cfg.Trays) == 0 {
		log.Fatalf("%s: no trays configured", *path)
	}

	ctl, err := newController(cfg)
	if err != nil {
		log.Fatal(err)
	}
	if err := ctl.publish(); err != nil { // trays see the current partitions
		log.Fatal(err)
	}

	lis, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		log.Fatal(err)
	}
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("GET /metrics", ctl.serveMetrics)
		log.Printf("fabric telemetry on http://%s/metrics", cfg.MetricsListen)
		log.Fatal(http.ListenAndServe(cfg.MetricsListen, mux))
	}()

	srv := grpc.NewServer()
	pb.RegisterNMXControllerServer(srv, ctl)
	reflection.Register(srv)
	log.Printf("fakenmxc %s: NVLink domain %s (%d GPUs) on %s", version, cfg.DomainName, len(ctl.gpus), cfg.Listen)
	log.Fatal(srv.Serve(lis))
}
