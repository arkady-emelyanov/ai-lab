[← README](../README.md) · **ComputeDomains (IMEX)**

# ComputeDomains (IMEX)

## Overview

On GB200, GPUs on different trays share memory over NVLink (multi-node NVLink, MNNVL): NCCL, PyTorch's symmetric memory and MNNVL all-reduces export GPU memory as a *fabric handle* on one tray and import it on another. The driver allows that only to a process with an *IMEX channel*, and only between nodes whose IMEX daemons are connected. On Kubernetes, NVIDIA's DRA driver for GPUs provides both through a **ComputeDomain**: for the pods of a workload, it runs an IMEX daemon on each node they land on and gives each pod an IMEX channel.

In k3s mode the lab runs NVIDIA's DRA driver unchanged (`nvidia_dra: true`, the default; 0.5.1, `nvidia_dra_version`; chart `oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu`, image `registry.k8s.io/dra-driver-nvidia/dra-driver-nvidia-gpu`) with its ComputeDomains only. GPUs stay with the lab's device plugin, NVIDIA's documented setup next to a device plugin, so Kueue, Run:ai and the examples keep requesting `nvidia.com/gpu`.

| Component | Where | Role |
|---|---|---|
| ComputeDomain controller | namespace `nvidia-dra-driver-gpu`, on `sched-control` | for each `ComputeDomain`: the IMEX daemons (a DaemonSet on the nodes its pods use), the channel `ResourceClaimTemplate`, the domain's status |
| ComputeDomain kubelet plugin | DaemonSet on the trays | publishes each tray's IMEX channel and daemon slot (`ResourceSlice`s, driver `compute-domain.nvidia.com`); prepares claims: the channel device node for a pod, the daemon's configuration |
| IMEX daemon (`compute-domain-daemon` running `nvidia-imex`) | a pod per tray of a running ComputeDomain | the domain's IMEX daemons, connected to each other |
| Driver root `/opt/fakegpu/driver` (`fakegpu_driver_root`) | trays | the fake driver laid out as NVIDIA's driver is on a host, for the DRA driver |
| IMEX device nodes | trays | `/dev/nvidia-caps-imex-channels/channel0` and `/dev/nvidia-caps/nvidia-cap1` (IMEX management), created by Incus |
| Kueue mapping | Kueue's configuration, ClusterQueue `gpu` | a channel claim counts as `nvidia.com/imex-channel`, one per tray |

The controller and each tray's plugin use about 10 MiB each.

## How it works

### The driver on the fake GPUs

The DRA driver expects a host's NVIDIA driver at its driver root. The lab gives it one of its own, so that a real driver on the lab's host does not show through: `fakegpu_driver_root` on each tray (roles/fakegpu) has

- the fake `libnvidia-ml` and `libcuda` under the driver version's names (`lib*.so.<fakegpu_driver_version>`, with `.so.1` links): the DRA driver loads NVML from there, and its CDI specifications name the libraries by version;
- `nvidia-smi`, a small native one for `--version` (the DRA driver checks a driver root with it, in its distroless image);
- `nvidia-imex` and `nvidia-imex-ctl` (below);
- `proc/devices` and `proc/driver/nvidia/capabilities/fabric-imex-mgmt`, in place of the host's: the device numbers of the IMEX device nodes (`fakegpu_caps_major`, `fakegpu_imex_channels_major`) and the minor of the IMEX management capability (`altProcDevices` in the chart);
- the tray's GPU state (`/var/lib/fakegpu`, bind-mounted), which the fake NVML there reads for each GPU's NVLink clique (`FAKEGPU_*` variables in the plugin's container).

The plugin finds each GPU's fabric state through NVML (`nvmlDeviceGetGpuFabricInfo`): the lab's NVLink domain and clique (`7f3c2a10-….1`), as GPU Feature Discovery labels the trays. A domain's IMEX daemons form per clique.

### IMEX daemons

`nvidia-imex` and `nvidia-imex-ctl` are the fake driver's (`fakegpu/imex.c`, static, so they run in NVIDIA's image). The compute-domain daemon writes the IMEX configuration and the node list (one DNS name per slot of the domain, resolved through `/etc/hosts`, which it maintains, `127.0.0.2` for an empty slot) and runs `nvidia-imex` with them, signalling it when the list changes; its readiness probe runs `nvidia-imex-ctl -q`. The fake daemon listens on the IMEX ports (`SERVER_PORT` 50000, command port 50005), connects to every peer in the list every 2 s, and answers `READY` once it serves; `nvidia-imex-ctl -N` shows the domain's nodes and their connections. It does not move memory: the fake driver shares fabric memory through files on `/shared` ([Emulated GPUs](fake-gpu.md#how-it-works)).

### IMEX channels

As with the real driver, a process creates or imports fabric memory only with an IMEX channel: the fake `libcuda` returns `CUDA_ERROR_NOT_PERMITTED` from `cuMemCreate` with `CU_MEM_HANDLE_TYPE_FABRIC` and from importing a fabric handle unless the process can open one of `/dev/nvidia-caps-imex-channels/channel<n>`. A pod of a ComputeDomain gets `channel0` from its claim (CDI); other pods have none. Slurm jobs run on the tray itself, which has `channel0`, as a GB200 node with IMEX set up for its jobs has. The existing partition rule still applies: fabric memory reaches only GPUs of the NVLink partition it was created in.

### Scheduling

A pod asks for a channel with a resource claim from the ComputeDomain's `ResourceClaimTemplate`. Kueue's DRA support counts it: the DeviceClass `compute-domain-default-channel.nvidia.com` maps to the resource `nvidia.com/imex-channel` (Kueue's configuration), and the ClusterQueue `gpu` has one per tray. GPUs come from the device plugin as before (`nvidia.com/gpu`), and Kueue's topology-aware scheduling keeps the pods in one NVLink domain.

## Usage

From the login node as a directory user:

```
cd examples/kubernetes
./submit --wait imex-share.yaml        # prints the run's name, writes <name>.out
kubectl get computedomains
kubectl delete computedomain <name>    # the ComputeDomain stays after the JobSet
```

`imex-share.yaml` creates a ComputeDomain of 2 nodes and a JobSet of 2 pods with 4 GPUs each (a tray each) and a channel claim; `../imex_share.py` exports 2 MiB of GPU memory as a fabric handle on one tray and imports and reads it on the other:

```
rank 0 on imex-share-…-rank-0-0: IMEX channels ['channel0']
rank 0 on imex-share-…-rank-0-0: exported 2 MiB as a fabric handle
rank 1 on imex-share-…-rank-0-1: imported the fabric handle, read the pattern
```

While a ComputeDomain's pods run: `kubectl get computedomain <name> -o yaml` shows the domain `Ready` with its nodes and clique; as cluster admin, `bin/kubectl -n nvidia-dra-driver-gpu exec <computedomain-daemon pod> -- nvidia-imex-ctl -c /imexd/imexd.cfg -N` shows the IMEX domain:

```
Node #0   - 10.42.3.97      - READY       - compute-domain-daemon-597f894f47-0000
Node #1   - 10.42.2.161     - READY       - compute-domain-daemon-597f894f47-0001
Domain State: UP
```

For your own workloads: add a ComputeDomain with `channel.resourceClaimTemplate.name`, and in each pod a `resourceClaims` entry for that template and `resources.claims` in the container that needs the channel ([NVIDIA's ComputeDomain documentation](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu/tree/main/site/content/docs)).

## Verification

`make test` in k3s mode runs `imex-share.yaml` as `joe` (rank 1 reads the pattern rank 0 wrote on the other tray) and checks that a GPU pod without a channel gets `CUDA_ERROR_NOT_PERMITTED` for fabric memory ([Testing](testing.md)). `make test-fakegpu` checks the channel rule in the fake driver.

## Limitations

- GPUs through DRA (`resources.gpus.enabled`, the DRA driver's own GPU allocation) are off: on the fake NVML the plugin's GPU discovery fails (MIG queries the fake NVML answers as supported), and the lab's device plugin also gives pods the fake driver's configuration and state, which the DRA driver's CDI specifications would not. So there is still no GPU reset between pods ([Kubernetes](kubernetes.md#limitations)).
- One channel per tray (`channel0`, the default `Single` allocation mode): `allocationMode: All` (all 2048 channels) is not available, and one ComputeDomain's pods per tray at a time get a channel.
- The fake IMEX daemon only shows the domain's state: memory moves through `/shared`, whether or not the daemons are connected.
- Untested with Run:ai: KAI and pods with resource claims in project namespaces.

## References

- [NVIDIA DRA driver for GPUs](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu), [IMEX guide](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/), [Kubernetes DRA](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/), [Kueue: DRA](https://kueue.sigs.k8s.io/docs/concepts/dynamic_resource_allocation/)
- Sources: `fakegpu/imex.c`, `fakegpu/smi_version.c`, `fakegpu/vmm.c` (channel rule), `roles/fakegpu` (driver root), `roles/k3s/templates/addons.yaml.j2` (chart, Kueue mapping), `examples/kubernetes/imex-share.yaml`, `examples/imex_share.py`
