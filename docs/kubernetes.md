[← README](../README.md) · **Kubernetes (k3s)**

# Kubernetes (k3s)

## Overview

With `scheduler: k3s` in `local.yml`, the lab runs Kubernetes (k3s 1.36) on the emulated hardware. Everything except the scheduler is shared with [Slurm](slurm.md) mode, and in k3s mode no Slurm component is installed.

To switch a built lab: `make down`, change `scheduler`, `make up`. Volumes, homes and the frameworks venv are kept.

| Component | Where | Role |
|---|---|---|
| k3s server | `sched-control` | control plane (SQLite datastore), tainted `CriticalAddonsOnly` so only add-ons run there |
| k3s agents | `sched-worker1`, `sched-worker2` | kubelet, containerd, flannel; label `nvidia.com/gpu.present=true` |
| `fakedp` (systemd, trays) | `fakedp/` | device plugin: advertises `nvidia.com/gpu` (4 per tray, by GPU UUID) and allocates GPUs as CDI devices; with MIG on, a GPU's MIG devices as `nvidia.com/mig-<profile>` instead ([MIG](#mig)) |
| CDI specification | `/etc/cdi/fakegpu.json` on the trays | one device per GPU (`/dev/nvidia<n>`), plus the fake driver, its configuration, sideband and telemetry for every GPU container |
| Node Feature Discovery + GPU Feature Discovery | `node-feature-discovery`, `gpu-feature-discovery` namespaces | NVIDIA's GFD reads the fake NVML and labels the trays: `nvidia.com/gpu.product`, `.memory`, `.count`, `.family`, `.clique` (`<ClusterUUID>.<CliqueId>`), `.machine` (`tray_product_name`, not the host's board), CUDA versions; with MIG on, the MIG devices (`nvidia.com/mig-<profile>.count`, `.memory`, `.multiprocessors`, ...: the mixed strategy) |
| topograph | systemd on `sched-control` | labels the trays from the live fabric: `accelerator.topograph.run/domain` (NVLink domain and clique), `fabric.topograph.run/tier-0/1` (InfiniBand leaf, spine); every minute |
| Kueue | `kueue-system` | ClusterQueue `gpu` (the trays' CPUs, memory, 8 GPUs; with `nvidia_dra`, `nvidia.com/imex-channel`, one per tray, for ComputeDomains' IMEX channel claims); per user namespace a LocalQueue `gpu` and a `default` one that takes workloads without a queue label, so nothing bypasses the quota; topology-aware scheduling over topograph's labels; admits multi-pod jobs as a gang |
| JobSet | `jobset-system` | multi-pod jobs with stable DNS names (rank 0 address for torchrun, Ray head) |
| kube-state-metrics | `kube-system`, NodePort 30808 | cluster state for Prometheus |
| NVIDIA DRA driver, ComputeDomains (`nvidia_dra: true`, the default) | `nvidia-dra-driver-gpu`; kubelet plugin on the trays | IMEX channels and daemons for GPU memory shared over NVLink between trays, as on GB200; GPUs stay with the device plugin ([ComputeDomains](compute-domains.md)) |
| HAProxy ingress | `haproxy-controller`, `sched-control` ports 80 and 443 | the lab's web UIs at names under `<controller address>.nip.io`: plain HTTP (KubeRay), HTTPS only where a component requires it (Run:ai) |
| KubeRay (`kuberay: true`, the default) | `kuberay` | operator for Ray clusters, jobs and services, its API server, dashboard at `http://kuberay.<controller address>.nip.io` and History Server (finished clusters' logs on RustFS, Ray's dashboard for them at `http://ray-history.<controller address>.nip.io`) ([KubeRay](kuberay.md)) |

Add-ons are installed by k3s' Helm controller from `HelmChart` manifests (`roles/k3s/templates/addons.yaml.j2`); versions are pinned in `all.yml`.

With `kuberay: true` (the default), KubeRay runs Ray clusters and jobs on this cluster, admitted by Kueue, with its web dashboard ([KubeRay](kuberay.md)). With `runai: true`, NVIDIA Run:ai runs on this cluster, with its own scheduler (KAI) for its projects' workloads next to Kueue ([NVIDIA Run:ai](runai.md)).

## How it works

### GPUs in pods

NVIDIA's device plugin and container toolkit need a real driver install, so the lab has its own device plugin (`fakedp`) and CDI specification.

- A pod asking for `nvidia.com/gpu: 2` gets two GPU device files and the lab's driver.
- As with the real driver, the pod sees only the GPUs whose device files it has, with the tray's UUIDs.
- Its GPU activity shows in `nvidia-smi`, the GPU exporter and the fabric telemetry, as on the tray. Processes appear with the tray's PIDs on the tray and with the pod's PIDs inside the pod.

### MIG

A GPU split into MIG devices ([Emulated GPUs](fake-gpu.md#mig)) is offered to pods as NVIDIA's device plugin does with its mixed strategy: `fakedp` stops advertising the GPU as `nvidia.com/gpu` and advertises each of its MIG devices under its profile, `nvidia.com/mig-1g.23gb`, `nvidia.com/mig-3g.93gb`, ... (`+me` as `.me`), the same names GPU Feature Discovery labels the node with (`nvidia.com/mig-1g.23gb.count`, ...). It reads the tray's MIG layout from `nvidia-smi` every 10 seconds, so a change reaches the node within about half a minute; it starts only once it has read it, so a GPU with MIG on is never offered whole (nor allocated whole). A container given MIG devices gets their parent GPU (CDI) and, as NVIDIA's plugin with its `volume-mounts` device list strategy, a file per `MIG-` UUID in `/var/run/nvidia-container-devices` (`NVIDIA_VISIBLE_DEVICES` names that directory), and sees only those: `nvidia-smi -L` lists the GPU and its one slice, CUDA has one device with the slice's memory and SMs. Unlike a list in `NVIDIA_VISIBLE_DEVICES`, of which kubelet keeps one value when several resources set it, the files add up, so one container can have MIG devices of several profiles, and whole GPUs besides.

Only MIG devices whose compute instance fills its GPU instance are offered: a GPU instance split into smaller compute instances (`nvidia-smi -L` lists them as `MIG 1c.3g.93gb`, ...) has no resource, as with NVIDIA's mixed strategy (NVIDIA's plugin refuses to start with one; `fakedp` leaves them out). A profile's resource exists while the tray has devices of it: when the last one goes, `fakedp` stops that profile's plugin, and kubelet removes the resource from the node after its grace period (5 minutes), as when NVIDIA's MIG manager restarts NVIDIA's plugin with a new layout.

Kueue's ClusterQueue `gpu` covers each MIG resource, with a quota of what the trays' GPUs can hold of it (7 `1g.23gb` per GPU, ...); the trays' capacity decides the actual mix. Splitting GPUs is an administrator's job, as with NVIDIA's MIG manager: the MIG commands run as root on a tray (`bin/ssh sched-worker2` logs in as root), `bin/ssh sched-worker2 nvidia-smi -i 3 -mig 1`, then `bin/ssh sched-worker2 nvidia-smi mig -i 3 -cgi 3g.93gb,1g.23gb,1g.23gb -C` (or `mig-parted`); a GPU switches only while no process uses it. A pod then asks for a slice:

```
resources:
  limits: {nvidia.com/mig-1g.23gb: 1}
```

### Topology

- GPU Feature Discovery labels each tray with its NVLink clique.
- topograph labels the trays with their NVLink domain and InfiniBand switches.
- Kueue's `nvl` topology orders them spine → leaf → NVLink domain → node. A job annotated `kueue.x-k8s.io/podset-required-topology: accelerator.topograph.run/domain` runs inside one NVLink domain or waits.
- After a partition change, the labels update within a minute.

topograph runs on the controller, outside the cluster, as in Slurm mode; it is given a service-account token so it can still label the nodes.

### Users

Each directory user has:

- a namespace of the same name, with `edit` rights in it (plus JobSets and Kueue; with `nvidia_dra`, ComputeDomains, and read access to ResourceClaims and ResourceClaimTemplates)
- a client certificate and `~/.kube/config` on `/shared`
- read access to nodes, ClusterQueues and the topology

### Running in containers

k3s runs in the same unprivileged Incus containers as everything else. A few settings make that work: the kernel log is passed in, kubelet runs in user-namespace mode, and the host's kernel keyring quota is raised (`make check` reports it).

### Names and images

- Pods resolve the lab's hostnames (`sched-storage`, ...), so S3, JuiceFS and LDAP work from pods by name.
- Docker Hub limits anonymous pulls, so images come through `mirror.gcr.io` (Google's cache of popular Docker Hub images), with Docker Hub as the fallback (`k3s_registry_mirrors`).

## Usage

From the login node (`bin/ssh login`) as a directory user; your kubeconfig is in place and your namespace is the default:

```
kubectl get nodes -L nvidia.com/gpu.clique,accelerator.topograph.run/domain
kubectl get clusterqueues; kubectl get localqueues
cd examples/kubernetes
./submit --wait nvl8-hello.yaml        # like sbatch --wait: prints the run name, writes <name>.out
kubectl get jobsets,workloads          # your jobs and their Kueue admission
```

| Example | Slurm equivalent | What it does |
|---|---|---|
| `kubernetes/gpu-topology.yaml` | `slurm/gpu-topology.sbatch` | one pod per tray with its 4 GPUs: `nvidia-smi -L`, `topo -m`, fabric state |
| `kubernetes/nvl8-hello.yaml` | `slurm/nvl8-hello.sbatch` | 8 pods × 1 GPU, admitted together into one NVLink domain |
| `kubernetes/ddp-train.yaml` | `slurm/ddp-train.sbatch` | PyTorch DDP, 2 pods × 4 GPUs, torchrun rendezvous on pod 0 |
| `kubernetes/ray-cluster.yaml` | `slurm/ray-cluster.sbatch` | Ray head and worker pods, driver on the head |
| `kubernetes/ray-job.yaml` | `slurm/ray-cluster.sbatch` | the same as a KubeRay `RayJob` (`kuberay: true`): KubeRay starts the Ray cluster, submits the driver and deletes the cluster when it ends; shows in the KubeRay dashboard ([KubeRay](kuberay.md)) |
| `kubernetes/vllm-serve.yaml` | – | vLLM's OpenAI-compatible server (official image) as a Deployment and Service on one GPU; answers are empty text of exactly `max_tokens` tokens ([examples](../examples/README.md)) |
| `kubernetes/sglang-serve.yaml` | – | SGLang's OpenAI-compatible server (official image, unmodified; the recommended inference server, see [Inference servers](inference.md)) as a Deployment and Service, on one GPU or with tensor parallelism inside a tray; answers are `"!!!..."` of exactly `max_tokens` tokens ([examples](../examples/README.md)) |
| `kubernetes/imex-share.yaml` | – | a ComputeDomain and 2 pods (a tray each): GPU memory exported as a fabric handle on one tray, imported and read on the other, over the domain's IMEX channels ([ComputeDomains](compute-domains.md)) |
| `kubernetes/sglang-tp8.yaml` | – | SGLang with tensor parallelism across both trays (TP=8, 2 pods × 4 GPUs); its all-reduces cross trays over NVLink, or over InfiniBand when the trays are in different NVLink partitions |

Examples are JobSets (`ray-job.yaml`: a RayJob) queued in Kueue (`kueue.x-k8s.io/queue-name: gpu`); `submit` fills in your uid, gid and home (`@UID@`, `@GID@`, `@HOME@`) and a unique name. Pods use `docker.io/library/buildpack-deps:noble` (Ubuntu 24.04 with Python, so the `/shared` venv runs unchanged) and mount `/shared` and `/scratch` from the tray.

From your machine, as cluster admin: `bin/kubectl ...` (kubeconfig in `.secrets/kubeconfig`).

**Desktop clients ([Freelens](https://github.com/freelensapp/freelens), Lens, k9s, ...)** connect with a kubeconfig file; the API server listens on the controller's bridge address (`https://10.107.111.10:6443`).

- **As cluster admin:** use `.secrets/kubeconfig` directly. In Freelens: *Catalog* → *Clusters* → **+**, then add the file `<repo>/.secrets/kubeconfig` (or copy it into `~/.kube/`, which Freelens reads on its own). The context is called `default`, as k3s names it.
- **As a directory user** (sees their own namespace, nodes and queues): copy their kubeconfig and point it at the bridge address, since `sched-control` does not resolve on your machine:

  ```
  bin/ssh root@login cat /shared/home/joe/.kube/config \
      | sed 's|https://sched-control:|https://10.107.111.10:|' > ~/.kube/ai-lab-joe
  chmod 600 ~/.kube/ai-lab-joe
  ```

  Then add `~/.kube/ai-lab-joe` the same way. Cluster-wide views Freelens shows (all namespaces, events) are partly empty for a user, by design of their RBAC.

`make up` creates a new cluster CA each time the lab is rebuilt from scratch, so re-copy the kubeconfig after `make down && make up`.

## Verification

```
bin/kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,CLIQUE:.metadata.labels.nvidia\.com/gpu\.clique'
bin/ssh sched-control update-topology --dry-run     # trays with their NVLink domain and IB switches
```

`make test` in k3s mode checks that:

- both trays are ready, with 4 GPUs and their clique and topology labels
- the examples run as `joe` and print the expected output (`ray-job.yaml` with KubeRay, admitted by Kueue)
- the 8 single-GPU pods of `nvl8-hello` get 8 different GPUs
- `joe` can read nodes and queues, but not `kube-system`
- a pod can reach S3
- a GPU split into MIG devices is offered as `nvidia.com/mig-<profile>` (and no longer as `nvidia.com/gpu`), and a pod on one `1g.23gb` sees that slice only, with 22528 MiB and 18 SMs
- topograph's labels and the scheduler metrics are in place

`make test-bmc-disruptive` cordons a tray, power-cycles it through its BMC and checks that it comes back ready with its GPUs.

## Limitations

- GPUs are allocated by the lab's own device plugin and CDI specification: NVIDIA's GPU Operator, container toolkit and device plugin need a real driver install. GPU Feature Discovery (NVIDIA), the DCGM exporter (NVIDIA, with Run:ai) and Node Feature Discovery (upstream) are the real ones.
- Users can create pods with `hostPath` volumes in their namespace (the examples need `/shared`), which in a real cluster would be denied by Pod Security; there is no admission policy forcing pods to run as the user's uid.
- NVIDIA's DRA driver runs with ComputeDomains only (IMEX channels); GPUs come from the lab's device plugin, not through DRA ([ComputeDomains](compute-domains.md#limitations)). NVLink placement is through Kueue topology-aware scheduling.
- Kubernetes users authenticate with client certificates, not LDAP.
- Fake NCCL ranks exchange only their NVLink partitions, so a pod failing does not fail its peers' collectives ([Emulated GPUs](fake-gpu.md)).
- No GPU handover between pods. The device-plugin API the lab uses has no hook on the node after a pod ends that could reset its GPUs (DRA drivers have one; the lab's GPUs do not go through DRA). GPUs are reset at tray boot, and their MIG instances go (MIG mode stays, as with the real driver: split them again after a tray's boot). After an NVLink partition change, reset idle GPUs by hand (`bin/ssh sched-worker2 nvidia-smi --gpu-reset`) so the labels pick up the new clique.

## References

- [k3s](https://docs.k3s.io/), [CoreDNS customization](https://docs.k3s.io/advanced#coredns-custom-configuration-imports), [Helm controller](https://docs.k3s.io/helm)
- [Container Device Interface](https://github.com/cncf-tags/container-device-interface), [kubelet device plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [NVIDIA GPU Feature Discovery](https://github.com/NVIDIA/k8s-device-plugin/tree/main/docs/gpu-feature-discovery), [Node Feature Discovery](https://kubernetes-sigs.github.io/node-feature-discovery/)
- [Kueue topology-aware scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/), [JobSet](https://jobset.sigs.k8s.io/)
- [topograph Kubernetes engine](https://github.com/dsx-ai-factory/topograph/blob/main/docs/engines/k8s.md)
- Sources: `roles/k3s/` (server, agents, GPUs and CDI, add-ons, users), `fakedp/`, `examples/kubernetes/`, `roles/topograph/`, `roles/monitoring/templates/sched-rules.yml.j2`
