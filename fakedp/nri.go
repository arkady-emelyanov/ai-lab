package main

// The part of NVIDIA's GPU stack that the GPU Operator runs as an NRI plugin
// (the container toolkit's, with the GPU Operator's cdi.nriPluginEnabled):
// a pod asks for CDI devices for one of its containers with the annotation
//
//	nvidia.cdi.k8s.io/container.<container>: <kind>=<device>[,...]
//
// and the plugin adds them to the container as it is created. As with the
// toolkit, only management devices (management.nvidia.com/gpu=all: every
// GPU and the driver, for daemons such as NVIDIA's GPU fractioning) are
// honoured, and only in the namespaces allowed to use them; anything else
// is left out, so the container starts without it. Workloads get their GPUs
// from the device plugin, or as CDI devices added by another NRI plugin
// (NVIDIA's fractiond, for a fraction of a GPU).

import (
	"context"
	"log"
	"strings"
	"time"

	"github.com/containerd/nri/pkg/api"
	"github.com/containerd/nri/pkg/stub"
)

const (
	cdiAnnotationPrefix = "nvidia.cdi.k8s.io/container."
	managementKind      = "management.nvidia.com/gpu"
)

type nriPlugin struct {
	namespaces map[string]bool
}

func (p *nriPlugin) CreateContainer(_ context.Context, pod *api.PodSandbox, ctr *api.Container) (*api.ContainerAdjustment, []*api.ContainerUpdate, error) {
	value, ok := pod.Annotations[cdiAnnotationPrefix+ctr.Name]
	if !ok {
		return nil, nil, nil
	}
	adj := &api.ContainerAdjustment{}
	for _, name := range strings.Split(value, ",") {
		name = strings.TrimSpace(name)
		switch {
		case name == "":
		case !strings.HasPrefix(name, managementKind+"="):
			log.Printf("%s/%s container %s: CDI device %q not allowed by annotation", pod.Namespace, pod.Name, ctr.Name, name)
		case !p.namespaces[pod.Namespace]:
			log.Printf("%s/%s container %s: management device %q not allowed in namespace %s",
				pod.Namespace, pod.Name, ctr.Name, name, pod.Namespace)
		default:
			adj.AddCDIDevice(&api.CDIDevice{Name: name})
			log.Printf("%s/%s container %s: added CDI device %s", pod.Namespace, pod.Name, ctr.Name, name)
		}
	}
	return adj, nil, nil
}

// runNRI serves NRI requests from the container runtime, connecting again
// whenever the connection ends (the runtime restarted, or had NRI off).
func runNRI(namespaces []string) {
	p := &nriPlugin{namespaces: map[string]bool{}}
	for _, ns := range namespaces {
		if ns = strings.TrimSpace(ns); ns != "" {
			p.namespaces[ns] = true
		}
	}
	for {
		s, err := stub.New(p, stub.WithPluginName("fakegpu-cdi"), stub.WithPluginIdx("10"))
		if err == nil {
			log.Printf("NRI plugin connecting (management devices for namespaces %s)", strings.Join(namespaces, ","))
			err = s.Run(context.Background())
		}
		log.Printf("NRI plugin: %v; connecting again", err)
		time.Sleep(5 * time.Second)
	}
}
