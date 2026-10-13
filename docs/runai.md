[← README](../README.md) · **NVIDIA Run:ai**

# NVIDIA Run:ai

## Overview

With `runai: true` (and `scheduler: k3s`) in `local.yml`, the lab runs [NVIDIA Run:ai](https://github.com/dsx-ai-factory/run-ai), the open-source AI workload and GPU orchestration platform, on its Kubernetes cluster: the control plane (web UI, REST API) and the cluster services (KAI Scheduler, resource management, workload controllers), connected to each other as one Run:ai cluster named `nvl8`. Projects, departments, quotas, fair share, preemption and workloads (training, workspaces, inference) work as on a GPU cluster, on the lab's emulated GB200 GPUs.

Run:ai is installed from its open-source release (3.0.20, `runai_version`): the Helm charts from NGC (`helm.ngc.nvidia.com/nvidia/runai-oss`) and the images from `nvcr.io/nvidia/runai-oss`, both public. Its prerequisites are installed with it.

| Component | Where | Role |
|---|---|---|
| Control plane (chart `control-plane`) | namespace `runai-backend`, on `sched-control` | web UI, REST API, Keycloak (users, SSO), PostgreSQL, NATS, Thanos (metrics) |
| Cluster services (chart `runai-cluster`) | namespace `runai`, on `sched-control` | Run:ai operator, agent and cluster sync (to the control plane), KAI Scheduler (`runai-scheduler`), KAI resource management (projects, node pools, queues), workload controllers, Run:ai's Prometheus |
| HAProxy ingress | namespace `haproxy-controller`, `sched-control` ports 80 and 443 | serves the UI and API at `https://runai.<controller address>.nip.io`, and workloads' URLs at its subdomains |
| kube-prometheus-stack | namespace `monitoring` | the Prometheus operator (Run:ai runs its own Prometheus through it), kube-state-metrics, kubelet metrics |
| DCGM exporter | namespace `gpu-operator`, DaemonSet on the trays | NVIDIA's GPU metrics exporter, unchanged, on the fake NVML; Run:ai reads GPU utilisation and memory from it |
| `runai-api` | `/usr/local/sbin` on `sched-control` | the REST API as the administrator (`runai-api GET /v1/k8s/clusters`) |

Run:ai's system services run on the controller, which is labelled `node-role.kubernetes.io/runai-system`; its workloads run on the trays. With Run:ai, the controller gets 6 CPUs and 16 GiB (`instance_limits_runai`; `make provision` applies them to a running lab). The running control plane and cluster services use about 5 GiB.

## How it works

### Installation

The `runai` role (`roles/runai`, run by `make configure` after the k3s cluster configuration) follows Run:ai's manual Helm installation:

1. A certificate authority of the lab's own (`.secrets/runai/ca.crt`) signs a certificate for the domain and its subdomains. Run:ai's services trust it (`global.customCA`).
2. The prerequisites: HAProxy (Run:ai's recommended ingress controller), kube-prometheus-stack and the DCGM exporter. Like the lab's other add-ons, they are k3s `HelmChart` manifests (`lab-runai-*.yaml` in k3s' manifests directory).
3. The control plane, with the administrator `admin@ai-lab.local` (`runai_admin_email`) and the password in `.secrets/runai-admin.pass`.
4. The cluster is registered with the control plane through the REST API, and the installer's client secret read from it, as Run:ai's own development setup does (instead of the onboarding wizard in the UI).
5. The cluster services, connecting to the control plane with that secret. The role waits until the control plane reports the cluster `Connected`.

Run:ai's charts request resources for production system nodes (about 15 CPUs in all, and 13 GiB for the control plane); the lab sets small CPU requests for every service and small memory requests for the control plane's, keeping the limits.

### GPUs

Run:ai expects NVIDIA's GPU Operator. What it reads from it, the lab provides on the emulated GPUs:

- GPUs as `nvidia.com/gpu`, allocated by GPU UUID: the lab's device plugin (`fakedp`, [Kubernetes](kubernetes.md#gpus-in-pods)).
- Node labels (`nvidia.com/gpu.product`, `.count`, `.memory`, `.clique`): NVIDIA's GPU Feature Discovery. Run:ai's default node pool detects the NVLink domain from `nvidia.com/gpu.clique`.
- GPU metrics: NVIDIA's DCGM exporter (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE` and more), in namespace `gpu-operator` with the names and labels the GPU Operator gives it. It mounts the tray's fake driver as the GPU Operator's runtime class would mount the real one. Each GPU's metrics carry the pod using it (kubelet's pod resources API), so Run:ai shows GPU utilisation and memory per workload.

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

`make test` with Run:ai checks that the cluster is connected with every required dependency, that Run:ai's Prometheus has DCGM metrics for every GPU, and that a 2-GPU training workload submitted through the API is scheduled by Run:ai onto a tray and sees its 2 GPUs.

## Limitations

- **GPU fractions.** Sharing a GPU between workloads (Run:ai's GPU fractions, KAI's GPU sharing) is off. Run:ai's GPU fractioning (`kai-gpu-fractioning`) has the driver enforce each container's GPU memory limit, which needs NVIDIA driver r615, its container toolkit hook and MPS. KAI's own GPU sharing gives a fractional pod its GPU through `NVIDIA_VISIBLE_DEVICES`, which NVIDIA's container runtime turns into device access; the lab's GPUs reach pods through CDI devices from the device plugin, which a pod gets only by requesting `nvidia.com/gpu`.
- **Optional frameworks.** Run:ai reports Knative (inference workloads), the Kubeflow training operator (PyTorch, TensorFlow, XGBoost, JAX distributed training), the MPI operator and LeaderWorkerSet (distributed inference) as not installed. Training and workspace workloads, and plain Kubernetes workloads in a project's namespace, need none of them.
- **Users.** The lab's LDAP users are not Run:ai users; the administrator creates users (or connects an identity provider) in the UI.
- **One system node.** The control plane and cluster services have a single replica each, on the controller; Run:ai recommends three system nodes.
- The DCGM exporter has no profiling metrics (`DCGM_FI_PROF_*`): DCGM's profiling module needs the real driver.
- `runai_domain: auto` needs nip.io, a public DNS service.

## References

- [NVIDIA Run:ai (source)](https://github.com/dsx-ai-factory/run-ai), [documentation](https://docs.nvidia.com/run-ai/), [installation with Helm](https://docs.nvidia.com/run-ai/self-hosted/documentation/installation-and-configuration/overview)
- [KAI Scheduler](https://github.com/kai-scheduler/KAI-scheduler), [KAI resource management](https://github.com/kai-scheduler/kai-resource-management), [GPU fractioning](https://github.com/kai-scheduler/kai-gpu-fractioning)
- [DCGM exporter](https://github.com/NVIDIA/dcgm-exporter), [HAProxy Kubernetes ingress](https://github.com/haproxytech/kubernetes-ingress), [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
