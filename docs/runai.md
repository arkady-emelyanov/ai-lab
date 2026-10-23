[← README](../README.md) · **NVIDIA Run:ai**

# NVIDIA Run:ai

## Overview

With `runai: true` (and `scheduler: k3s`) in `local.yml`, the lab runs [NVIDIA Run:ai](https://github.com/dsx-ai-factory/run-ai), the open-source AI workload and GPU orchestration platform, on its Kubernetes cluster: the control plane (web UI, REST API) and the cluster services (KAI Scheduler, resource management, workload controllers), connected to each other as one Run:ai cluster named `nvl8`. Projects, departments, quotas, fair share, preemption and workloads (training, workspaces, inference) work as on a GPU cluster, on the lab's emulated GB200 GPUs.

Run:ai is installed from its open-source release (3.0.20, `runai_version`): the Helm charts from NGC (`helm.ngc.nvidia.com/nvidia/runai-oss`) and the images from `nvcr.io/nvidia/runai-oss`, both public. Its prerequisites are installed with it.

| Component | Where | Role |
|---|---|---|
| Control plane (chart `control-plane`) | namespace `runai-backend`, on `sched-control` | web UI, REST API, Keycloak (users, SSO), PostgreSQL, NATS, Thanos (metrics) |
| Cluster services (chart `runai-cluster`) | namespace `runai`, on `sched-control` | Run:ai operator, agent and cluster sync (to the control plane), KAI Scheduler (`runai-scheduler`), KAI resource management (projects, node pools, queues), workload controllers, Run:ai's Prometheus |
| HAProxy ingress (a k3s add-on, [Kubernetes](kubernetes.md)) | namespace `haproxy-controller`, `sched-control` ports 80 and 443 | serves the UI and API at `https://runai.<controller address>.nip.io`, and workloads' URLs at its subdomains |
| kube-prometheus-stack | namespace `monitoring` | the Prometheus operator (Run:ai runs its own Prometheus through it), kube-state-metrics, kubelet metrics |
| DCGM exporter | namespace `gpu-operator`, DaemonSet on the trays | NVIDIA's GPU metrics exporter, unchanged, on the fake NVML; Run:ai reads GPU utilisation and memory from it |
| NVIDIA GPU fractioning (0.1.7) | namespace `runai`: operator on `sched-control`, `fractiond` and `mpsd` on the trays | GPU fractions: gives a container part of a GPU and its memory limit, which the fake driver enforces |
| `ClusterPolicy` `cluster-policy` | cluster-wide | the GPU Operator's resource, describing the lab's GPU stack to KAI and GPU fractioning (nothing reconciles it) |
| `runai-api` | `/usr/local/sbin` on `sched-control` | the REST API as the administrator (`runai-api GET /v1/k8s/clusters`) |

Run:ai's system services run on the controller, which is labelled `node-role.kubernetes.io/runai-system`; its workloads run on the trays. With Run:ai, the controller gets 6 CPUs and 16 GiB (`instance_limits_runai`; `make provision` applies them to a running lab). The running control plane and cluster services use about 5 GiB.

## How it works

### Installation

The `runai` role (`roles/runai`, run by `make configure` after the k3s cluster configuration) follows Run:ai's manual Helm installation:

1. A certificate authority of the lab's own (`.secrets/runai/ca.crt`) signs a certificate for the domain and its subdomains. Run:ai's services trust it (`global.customCA`).
2. The prerequisites: kube-prometheus-stack and the DCGM exporter. Like the lab's other add-ons, they are k3s `HelmChart` manifests (`lab-runai-*.yaml` in k3s' manifests directory). HAProxy, Run:ai's recommended ingress controller, is already there: every k3s lab has it.
3. The control plane, with the administrator `admin@ai-lab.local` (`runai_admin_email`) and the password in `.secrets/runai-admin.pass`.
4. The cluster is registered with the control plane through the REST API, and the installer's client secret read from it, as Run:ai's own development setup does (instead of the onboarding wizard in the UI).
5. The cluster services, connecting to the control plane with that secret. The role waits until the control plane reports the cluster `Connected`.

Run:ai's charts request resources for production system nodes (about 15 CPUs in all, and 13 GiB for the control plane); the lab sets small CPU requests for every service and small memory requests for the control plane's, keeping the limits.

### GPUs

Run:ai expects NVIDIA's GPU Operator. What it reads from it, the lab provides on the emulated GPUs:

- GPUs as `nvidia.com/gpu`, allocated by GPU UUID: the lab's device plugin (`fakedp`, [Kubernetes](kubernetes.md#gpus-in-pods)).
- Node labels (`nvidia.com/gpu.product`, `.count`, `.memory`, `.clique`): NVIDIA's GPU Feature Discovery. Run:ai's default node pool detects the NVLink domain from `nvidia.com/gpu.clique`.
- GPU metrics: NVIDIA's DCGM exporter (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE` and more), in namespace `gpu-operator` with the names and labels the GPU Operator gives it. It mounts the tray's fake driver as the GPU Operator's runtime class would mount the real one. Each GPU's metrics carry the pod using it (kubelet's pod resources API), so Run:ai shows GPU utilisation and memory per workload.

### GPU fractions

A workload can ask for part of a GPU (`gpuPortionRequest: 0.5`, or an amount of GPU memory). Run:ai 3.0 allows only NVIDIA's enforced fractions (KAI's `NvFractions` mode, not configurable): KAI places the fraction on a GPU, and NVIDIA's GPU fractioning gives the container that GPU and holds it to its share of the memory. It runs unchanged on the trays, on what the lab provides for NVIDIA's driver, container toolkit and GPU Operator:

- The driver enforces a fraction's memory limit: the fake CUDA driver reads `NVIDIA_GPU_MEMORY_LIMIT` (MiB, which GPU fractioning gives the container) and holds each process to it on each GPU. An allocation beyond it fails with out of memory, and `cuMemGetInfo` reports the limit as the GPU's total ([Emulated GPUs](fake-gpu.md#gpu-fractions)). The driver reports version 615.20.01 (CUDA 13.4) with Run:ai: GPU fractioning needs r615.
- The MPS control daemon its `mpsd` supervises: the fake `nvidia-cuda-mps-control` (control socket and commands, no MPS servers: the fake driver runs every process's work itself).
- NRI in the trays' containerd, and the GPU Operator's NRI plugin (`fakedp -nri`): its daemons get every GPU and the driver through the annotation `nvidia.cdi.k8s.io/container.<name>: management.nvidia.com/gpu=all`, allowed in namespace `runai` only. `fractiond` gives a fractional container its GPU as the CDI device `k8s.device-plugin.nvidia.com/gpu=<UUID>`. The trays have CDI specifications under both names.
- The node label `nvidia.com/gpu.deploy.client=true`, with which the GPU Operator lets GPU client daemons run on a node.
- A `ClusterPolicy` with the NRI plugin on (`cdi.nriPluginEnabled`), the device plugin and container toolkit versions GPU fractioning requires, and a ready status.

GPU fractioning 0.1.7 replaces the 0.1.6 Run:ai bundles (`gpu_fractioning_version`, the images only): 0.1.6 reaches GPUs only through NVIDIA's container runtime (`nvidia` runtime class), 0.1.7 also through the GPU Operator's NRI plugin. The `ClusterPolicy` says `cdi.enabled: false` because of a disagreement between KAI 0.18.3 and GPU fractioning 0.1.7: with CDI enabled, KAI writes a fractional container's GPU as a CDI name and `fractiond` makes it one again (`unresolvable CDI devices k8s.device-plugin.nvidia.com/gpu=k8s.device-plugin.nvidia.com/gpu=…`); without it, KAI writes the GPU's UUID. GPU fractioning reads only `nriPluginEnabled`.

Two half-GPU workloads then share one GPU: each sees `NVIDIA_GPU_MEMORY_LIMIT` of half the GPU's memory (94735 MiB), allocates up to it, and gets out of memory beyond it; `nvidia-smi` on the tray shows both processes on the GPU.

### Names and certificates

The UI and API are at `https://runai.<controller address>.nip.io` (`runai_domain: auto`). [nip.io](https://nip.io) is a public DNS service that resolves a name to the address it contains, so the host's browser and the lab's pods reach the controller by name without a hosts entry. Workloads that expose a URL get a subdomain of it (host-based routing). Set `runai_domain` to a name of your own if the lab has no internet access (the host and the lab must then resolve it and its subdomains to the controller).

## Usage

### Enabling Run:ai

In `local.yml`:

```yaml
scheduler: k3s
runai: true
```

On a running k3s lab: `make provision` (the controller's new limits), then `make configure` (about 15 minutes the first time, mostly pulling images). On a new lab, `make up` does both. The last task prints the address.

### Signing in

Open `https://runai.10.107.111.10.nip.io` (your bridge's prefix may differ) and sign in as `admin@ai-lab.local` with the password in `.secrets/runai-admin.pass`. Browsers warn about the certificate until `.secrets/runai/ca.crt` is imported as a trusted certificate authority, for example in Chrome: Settings → Privacy and security → Security → Manage certificates → Authorities → Import.

In the UI, create a project with a GPU quota on the `default` node pool (8 GPUs), then submit workloads to it: Workload manager → Workloads → New workload. A project's workloads run in namespace `runai-<project>`. An image with the tools the examples use is `docker.io/library/buildpack-deps:noble`; the GPUs bring the fake driver and `nvidia-smi` with them.

### REST API

`runai-api` on the controller signs in as the administrator and makes one call ([API reference](https://docs.nvidia.com/run-ai/self-hosted/reference/home)):

```
bin/ssh sched-control runai-api GET /v1/k8s/clusters                    # the cluster and its state
bin/ssh sched-control runai-api GET /api/v1/org-unit/projects
bin/ssh sched-control "runai-api POST /api/v1/workloads/trainings '{
  \"name\": \"gpu-check\", \"projectId\": \"<project id>\", \"clusterId\": \"<cluster uuid>\",
  \"spec\": {\"image\": \"docker.io/library/buildpack-deps:noble\", \"command\": \"nvidia-smi\", \"args\": \"-L\",
           \"compute\": {\"gpuDevicesRequest\": 2}}}'"
bin/ssh sched-control runai-api GET '/api/v1/workloads?filterBy=name==gpu-check'
```

## Verification

```
bin/ssh sched-control runai-api GET /v1/k8s/clusters | jq '.[] | {name, state: .status.state}'   # Connected
bin/kubectl get pods -n runai-backend; bin/kubectl get pods -n runai                               # all Running
bin/kubectl get --raw '/api/v1/namespaces/runai/services/prometheus-operated:9090/proxy/api/v1/query?query=count(DCGM_FI_DEV_GPU_UTIL)'
```

`make test` with Run:ai checks that the cluster is connected with every required dependency, that Run:ai's Prometheus has DCGM metrics for every GPU, that a 2-GPU training workload submitted through the API is scheduled by Run:ai onto a tray and sees its 2 GPUs, and that two half-GPU workloads share one GPU, each held to half of its memory.

## Limitations

- **GPU fractions.** The memory limit is per process, as MPS enforces it (two processes of one fraction can each use the whole limit). Compute is not divided: a fraction's work runs at the whole GPU's modelled speed, and per-process utilisation is the GPU's split evenly between its processes. `sm-sharing` compute mode and MIG are not modelled.
- **GPU Operator.** There is no GPU Operator: its `ClusterPolicy` is written once by the lab, and nothing acts on changes to it. KAI reads it when its operator starts: adding it to a running Run:ai (an existing lab gaining fractions) needs `bin/kubectl -n runai rollout restart deploy/kai-operator`.
- **Optional frameworks.** Run:ai reports Knative (inference workloads), the Kubeflow training operator (PyTorch, TensorFlow, XGBoost, JAX distributed training), the MPI operator and LeaderWorkerSet (distributed inference) as not installed. Training and workspace workloads, and plain Kubernetes workloads in a project's namespace, need none of them.
- **Users.** The lab's LDAP users are not Run:ai users; the administrator creates users (or connects an identity provider) in the UI.
- **One system node.** The control plane and cluster services have a single replica each, on the controller; Run:ai recommends three system nodes.
- The DCGM exporter has no profiling metrics (`DCGM_FI_PROF_*`): DCGM's profiling module needs the real driver.
- `runai_domain: auto` needs nip.io, a public DNS service.

## References

- [NVIDIA Run:ai (source)](https://github.com/dsx-ai-factory/run-ai), [documentation](https://docs.nvidia.com/run-ai/), [installation with Helm](https://docs.nvidia.com/run-ai/self-hosted/documentation/installation-and-configuration/overview)
- [KAI Scheduler](https://github.com/kai-scheduler/KAI-scheduler), [KAI resource management](https://github.com/kai-scheduler/kai-resource-management), [GPU fractioning](https://github.com/kai-scheduler/kai-gpu-fractioning), [GPU fractions in KAI](https://github.com/kai-scheduler/KAI-scheduler/issues/1544)
- [DCGM exporter](https://github.com/NVIDIA/dcgm-exporter), [HAProxy Kubernetes ingress](https://github.com/haproxytech/kubernetes-ingress), [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
