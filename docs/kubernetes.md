[← README](../README.md) · **Kubernetes (k3s)**

# Kubernetes (k3s)

## Overview

With `scheduler: k3s` in `local.yml`, the lab runs Kubernetes (k3s 1.36) on the emulated hardware. Everything except the scheduler is shared with [Slurm](slurm.md) mode, and in k3s mode no Slurm component is installed.

To switch a built lab: `make down`, change `scheduler`, `make up`. Volumes, homes and the frameworks venv are kept.

| Component | Where | Role |
|---|---|---|
| k3s server | `sched-control` | control plane (SQLite datastore), tainted `CriticalAddonsOnly` so only add-ons run there |
| k3s agents | `sched-worker1`, `sched-worker2` | kubelet, containerd, flannel; label `nvidia.com/gpu.present=true` |
| `fakedp` (systemd, trays) | `fakedp/` | device plugin: advertises `nvidia.com/gpu` (4 per tray, by GPU UUID) and allocates GPUs as CDI devices |
| CDI specification | `/etc/cdi/fakegpu.json` on the trays | one device per GPU (`/dev/nvidia<n>`), plus the fake driver, its configuration, sideband and telemetry for every GPU container |
| Node Feature Discovery + GPU Feature Discovery | `node-feature-discovery`, `gpu-feature-discovery` namespaces | NVIDIA's GFD reads the fake NVML and labels the trays: `nvidia.com/gpu.product`, `.memory`, `.count`, `.family`, `.clique` (`<ClusterUUID>.<CliqueId>`), `.machine` (`tray_product_name`, not the host's board), CUDA versions |
| topograph | systemd on `sched-control` | labels the trays from the live fabric: `accelerator.topograph.run/domain` (NVLink domain and clique), `fabric.topograph.run/tier-0/1` (InfiniBand leaf, spine); every minute |
| Kueue | `kueue-system` | ClusterQueue `gpu` (the trays' CPUs, memory, 8 GPUs); per user namespace a LocalQueue `gpu` and a `default` one that takes workloads without a queue label, so nothing bypasses the quota; topology-aware scheduling over topograph's labels; admits multi-pod jobs as a gang |
| JobSet | `jobset-system` | multi-pod jobs with stable DNS names (rank 0 address for torchrun, Ray head) |
| kube-state-metrics | `kube-system`, NodePort 30808 | cluster state for Prometheus |

Add-ons are installed by k3s' Helm controller from `HelmChart` manifests (`roles/k3s/templates/addons.yaml.j2`); versions are pinned in `all.yml`.

## How it works

### GPUs in pods

NVIDIA's device plugin and container toolkit need a real driver install, so the lab has its own device plugin (`fakedp`) and CDI specification.

- A pod asking for `nvidia.com/gpu: 2` gets two GPU device files and the lab's driver.
- As with the real driver, the pod sees only the GPUs whose device files it has, with the tray's UUIDs.
- Its GPU activity shows in `nvidia-smi`, the GPU exporter and the fabric telemetry, as on the tray. Processes appear with the tray's PIDs on the tray and with the pod's PIDs inside the pod.

### Topology

- GPU Feature Discovery labels each tray with its NVLink clique.
- topograph labels the trays with their NVLink domain and InfiniBand switches.
- Kueue's `nvl` topology orders them spine → leaf → NVLink domain → node. A job annotated `kueue.x-k8s.io/podset-required-topology: accelerator.topograph.run/domain` runs inside one NVLink domain or waits.
- After a partition change, the labels update within a minute.

topograph runs on the controller, outside the cluster, as in Slurm mode; it is given a service-account token so it can still label the nodes.

### Users

Each directory user has:

- a namespace of the same name, with `edit` rights in it (plus JobSets and Kueue)
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

Examples are JobSets queued in Kueue (`kueue.x-k8s.io/queue-name: gpu`); `submit` fills in your uid, gid and home (`@UID@`, `@GID@`, `@HOME@`) and a unique name. Pods use `docker.io/library/buildpack-deps:noble` (Ubuntu 24.04 with Python, so the `/shared` venv runs unchanged) and mount `/shared` and `/scratch` from the tray.

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
- the four examples run as `joe` and print the expected output
- the 8 single-GPU pods of `nvl8-hello` get 8 different GPUs
- `joe` can read nodes and queues, but not `kube-system`
- a pod can reach S3
- topograph's labels and the scheduler metrics are in place

`make test-bmc-disruptive` cordons a tray, power-cycles it through its BMC and checks that it comes back ready with its GPUs.

## Limitations

- GPUs are allocated by the lab's own device plugin and CDI specification: NVIDIA's GPU Operator, container toolkit and device plugin need a real driver install. GPU Feature Discovery (NVIDIA) and Node Feature Discovery (upstream) are the real ones.
- Users can create pods with `hostPath` volumes in their namespace (the examples need `/shared`), which in a real cluster would be denied by Pod Security; there is no admission policy forcing pods to run as the user's uid.
- No NVIDIA DRA driver or ComputeDomains (IMEX); NVLink placement is through Kueue topology-aware scheduling only.
- Kubernetes users authenticate with client certificates, not LDAP.
- Fake NCCL ranks exchange only their NVLink partitions, so a pod failing does not fail its peers' collectives ([Emulated GPUs](fake-gpu.md)).
- No GPU handover between pods. The device-plugin API the lab uses has no hook on the node after a pod ends that could reset its GPUs (DRA drivers have one; the lab does not use DRA). GPUs are reset at tray boot. After an NVLink partition change, reset idle GPUs by hand (`bin/ssh sched-worker2 nvidia-smi --gpu-reset`) so the labels pick up the new clique.

## References

- [k3s](https://docs.k3s.io/), [CoreDNS customization](https://docs.k3s.io/advanced#coredns-custom-configuration-imports), [Helm controller](https://docs.k3s.io/helm)
- [Container Device Interface](https://github.com/cncf-tags/container-device-interface), [kubelet device plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [NVIDIA GPU Feature Discovery](https://github.com/NVIDIA/k8s-device-plugin/tree/main/docs/gpu-feature-discovery), [Node Feature Discovery](https://kubernetes-sigs.github.io/node-feature-discovery/)
- [Kueue topology-aware scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/), [JobSet](https://jobset.sigs.k8s.io/)
- [topograph Kubernetes engine](https://github.com/dsx-ai-factory/topograph/blob/main/docs/engines/k8s.md)
- Sources: `roles/k3s/` (server, agents, GPUs and CDI, add-ons, users), `fakedp/`, `examples/kubernetes/`, `roles/topograph/`, `roles/monitoring/templates/sched-rules.yml.j2`
