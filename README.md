# AI lab: an emulated NVIDIA GB200 NVL8 GPU cluster

![The Lab banner](/docs/assets/social-preview.png)

> **Independent research project, not affiliated with or endorsed by NVIDIA.** It emulates NVIDIA hardware and software interfaces in software, for personal research and education. See [Disclaimer](#disclaimer).

A complete GPU cluster on one Linux machine, for building and testing cluster software, self-service tooling and operations without GPUs. It emulates one NVIDIA GB200-class **NVL8** NVLink domain (two GPU trays with four GPUs each, an NVLink switch tray, BMCs, an InfiniBand fabric) and runs the real software stack around it: Slurm with accounting or Kubernetes (k3s) with Kueue, LDAP identity, shared and object storage, Prometheus and Grafana.

The GPUs are fake but behave like real ones to everything above them: applications launched through Slurm or Kubernetes initialise PyTorch, NCCL or Ray, every call succeeds and takes realistic simulated time, and the GPUs report realistic load, memory, power and temperature. Nothing is computed. Management interfaces (Redfish BMCs, an NVLink partition controller, fabric telemetry) change what the GPUs report and what the scheduler places jobs on.

What you get:

- **Users and jobs:** LDAP users with SSH access and one scheduler of your choice (`scheduler: slurm | k3s`): Slurm with GPU scheduling, accounting and NVLink-aware block topology, or k3s with GPU pods, NVIDIA GPU Feature Discovery labels, Kueue gang and NVLink-topology-aware scheduling and JobSets; PyTorch DDP and Ray examples on 8 GPUs for both.
- **Hardware management:** Redfish BMCs per tray (power, NVLink ports) and for the switch tray (144 ports), an NMX-C-style partition controller, topology discovery with topograph.
- **Platform:** shared filesystem (JuiceFS on RustFS), per-user S3 buckets, node-local scratch, Prometheus with GPU, scheduler and NVLink fabric metrics, Grafana dashboards.
- **Operations:** one command to build, one to test, everything in Ansible.

## Use cases

- **Cluster software and self-service:** develop job portals, scheduler plugins, quota and accounting tools, or user-facing CLIs against a real Slurm or Kubernetes, LDAP and S3 stack with GPU nodes.
- **Monitoring and operations:** build dashboards, alerts and runbooks on realistic GPU, scheduler and NVLink fabric metrics; rehearse node power cycles, NVLink failures and partition changes through Redfish and the partition controller.
- **Hardware management tooling:** test Redfish clients, BMC automation and topology-aware scheduling (topograph with Slurm `topology/block` or Kueue node labels) without a rack.
- **AI pipelines:** run PyTorch DDP and Ray jobs end to end on 8 GPUs with realistic timing and memory accounting, to test orchestration, data movement and failure handling rather than numerics.
- **CI for infrastructure code:** `make up && make test` builds and verifies the whole cluster from scratch on one machine.

## Architecture at a glance

```
                              your machine (Incus host)
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │  bin/ssh slurm ─► sched-login    .11   login node, user shells                  │
 │                   sched-control  .10   slurmctld+slurmdbd or k3s server, LDAP,  │
 │                                        Prometheus, Grafana, topograph           │
 │                   sched-worker1  .21 ┐ GPU trays: slurmd or k3s agent,          │
 │                                      │ 4 × fake GB200 each                      │
 │                   sched-worker2  .22 ┘ one NVL8 NVLink domain                   │
 │                   sched-storage  .12   RustFS (S3), Redis                       │
 │                   sched-nvswitch .34   NVLink partition controller, telemetry   │
 │  bin/redfish ───► sched-worker1-bmc .31, sched-worker2-bmc .32,                 │
 │                   sched-nvswitch-bmc .33   (Redfish BMCs)                       │
 │                                                                                 │
 │  /shared (homes, venv) · /pfs (JuiceFS) · /scratch (per tray) · S3 per user     │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

Addresses are on the Incus bridge (`10.107.111.<n>` here). The scheduler is Slurm or k3s, never both (`scheduler` in `inventory/group_vars/all.yml`); everything else is shared. Details: [Architecture](docs/architecture.md).

## Quickstart

**Requirements:** Linux with [Incus](https://linuxcontainers.org/incus/), `make`, Python 3, Go, `jq`, git; your user in the `incus-admin` group. The running cluster uses about 10 GiB of RAM (16 GiB free recommended; container memory limits add up to 27 GiB) and 4+ CPU cores; plan about 25 GB of disk under `/var/lib/incus`, including the 5.6 GB frameworks venv.

```
make init          # host side: Ansible venv, secrets, Go builds; reports any host fix needed
make up            # create and configure the cluster (~15 minutes)
make frameworks    # PyTorch + Ray on the cluster (optional, several GB)
make test          # end-to-end checks
```

`make init` prints the exact command for anything the host still needs (group membership, an AppArmor rule for Incus DNS, the inotify and kernel keyring limits); see [Platform](docs/platform.md#troubleshooting).

**Kubernetes instead of Slurm:** set `scheduler: k3s` in `inventory/group_vars/all.yml` before `make up` (on a built cluster: `make down`, change it, `make up`; volumes are kept). See [Kubernetes (k3s)](docs/kubernetes.md).

**First steps:**

```
bin/scp examples/* slurm:                       # copy the example jobs to joe's home
bin/ssh slurm                                   # login node as joe (password: joe)
sinfo -N -o "%N %G %T"                          # two trays, gpu:gb200:4 each
srun -N2 --gpus-per-node=4 nvidia-smi -L        # all 8 GPUs
sbatch nvl8-hello.sbatch                        # one task per GPU across the domain
```

With k3s: `bin/ssh slurm`, then `kubectl get nodes -L nvidia.com/gpu.clique`, `cd examples/k8s && ./submit --wait nvl8-hello.yaml` ([Kubernetes](docs/kubernetes.md#usage)).

Then open Grafana at `http://10.107.111.10:3000` (user `admin`, password in `.secrets/grafana.pass`) and watch the **Lab overview** while a job runs, for example `ddp-train.sbatch` from the [examples](docs/frameworks-and-examples.md).

## Lab endpoints

Reachable from the host machine. Addresses are on the Incus bridge (`10.107.111.0/24` here; yours may differ, see `incus network get incusbr0 ipv4.address`).

| Service | Endpoint | Credentials | Docs |
|---|---|---|---|
| Login node (SSH) | `bin/ssh slurm` (10.107.111.11:22) | `joe` / `joe`, or your key | [Identity and access](docs/identity-and-access.md) |
| Any instance as root (SSH) | `bin/ssh root@<instance>` | `.secrets/ssh/id_ed25519` | [Identity and access](docs/identity-and-access.md) |
| Grafana | http://10.107.111.10:3000 | `admin` / `.secrets/grafana.pass` | [Monitoring](docs/monitoring.md) |
| Prometheus | http://10.107.111.10:9090 | none | [Monitoring](docs/monitoring.md) |
| Slurm exporter (Slurm mode) | http://10.107.111.10:9092/metrics | none | [Monitoring](docs/monitoring.md) |
| Kubernetes API (k3s mode) | https://10.107.111.10:6443 (`bin/kubectl`) | admin: `.secrets/kubeconfig`; users: `~/.kube/config` | [Kubernetes](docs/kubernetes.md) |
| kube-state-metrics (k3s mode) | http://10.107.111.10:30808/metrics | none | [Monitoring](docs/monitoring.md) |
| topograph API | http://10.107.111.10:49021 | none | [Topology discovery](docs/topology.md) |
| RustFS S3 API | http://10.107.111.12:9000 | admin: `.secrets/rustfs.access` / `.secrets/rustfs.secret`; users: `<name>` / `.secrets/users/<name>.s3` | [Storage](docs/storage.md) |
| RustFS console | http://10.107.111.12:9001/rustfs/console/ | as S3 API | [Storage](docs/storage.md) |
| GPU tray BMCs (Redfish) | https://10.107.111.31, https://10.107.111.32 (`bin/redfish sched-worker1 …`) | `root` / `0penBmc` | [BMCs](docs/bmc-redfish.md) |
| NVLink switch tray BMC (Redfish) | https://10.107.111.33 (`bin/redfish sched-nvswitch …`) | `root` / `0penBmc` | [BMCs](docs/bmc-redfish.md) |
| NVLink partition controller (gRPC) | 10.107.111.34:9370 (plaintext, reflection) | none | [NVLink partitions](docs/nvlink-partitions.md) |
| Fabric telemetry | http://10.107.111.34:9372/metrics | none | [NVLink partitions](docs/nvlink-partitions.md) |
| GPU exporters | http://10.107.111.21:9835/metrics, http://10.107.111.22:9835/metrics | none | [Monitoring](docs/monitoring.md) |
| node_exporter, JuiceFS metrics | `<cluster node>:9100/metrics`, `<cluster node>:9567/metrics` | none | [Monitoring](docs/monitoring.md) |

The BMCs use self-signed certificates (`curl -k`).

## Layers and components

Each page covers: overview, usage, verification, references (plus configuration and limitations where relevant).

| Layer | Component | Page |
|---|---|---|
| Infrastructure | Incus containers, volumes, Ansible, `make` targets, secrets, host troubleshooting | [Platform](docs/platform.md) |
| | Instances, addresses, wiring between layers | [Architecture](docs/architecture.md) |
| Emulated hardware | Fake NVIDIA stack: CUDA, NVML, cuBLAS, NCCL, cuDNN, `nvidia-smi`; simulated timing and GPU occupancy | [Fake GPUs](docs/fake-gpu.md) |
| | Redfish BMCs for the GPU trays and the NVLink switch tray | [BMCs](docs/bmc-redfish.md) |
| | NVLink partition controller (NMX-C-like, gRPC) and fabric telemetry | [NVLink partitions](docs/nvlink-partitions.md) |
| | Emulated InfiniBand fabric and topograph-generated scheduler topology | [Topology discovery](docs/topology.md) |
| Platform services | OpenLDAP, SSSD, SSH, users, `bin/ssh` | [Identity and access](docs/identity-and-access.md) |
| | `/shared`, `/pfs` (JuiceFS), `/scratch`, RustFS S3 with per-user buckets | [Storage](docs/storage.md) |
| | Prometheus, exporters (node, GPU, Slurm or kube-state-metrics, NVLink, JuiceFS), Grafana | [Monitoring](docs/monitoring.md) |
| Scheduling (one of) | Slurm, accounting, GPU GRES, block topology, scratch | [Slurm](docs/slurm.md) |
| | k3s, GPU device plugin and CDI, GPU Feature Discovery, Kueue, JobSet | [Kubernetes (k3s)](docs/kubernetes.md) |
| Applications | PyTorch and Ray venv, example jobs | [Frameworks and examples](docs/frameworks-and-examples.md) |
| Quality | `make test` coverage | [Testing](docs/testing.md) |

## Everyday commands

| Command | Purpose |
|---|---|
| `make up`, `make configure` | build the cluster; re-apply configuration after changing `inventory/group_vars/all.yml` |
| `make test` | end-to-end checks |
| `make down`, `make purge` | delete instances (keep volumes); delete everything |
| `bin/ssh slurm` (or `login`), `bin/ssh root@<instance>` | SSH as joe / root; `bin/scp`, `bin/ssh-copy-id` likewise |
| `bin/kubectl <args>` | kubectl as cluster admin (k3s mode) |
| `bin/grpcurl <args>` | grpcurl for the NVLink partition controller (built on first use) |
| `make shell`, `make shell-<instance>` | shells through `incus exec` |
| `bin/redfish <tray> <path> [curl args]` | Redfish requests to a BMC |
| `bin/ssh sched-control update-topology` | regenerate the scheduler's topology now (a timer does it every minute) |

## Repository layout

```
Makefile                 entry points (init, up, configure, frameworks, test, down, purge, shell)
bin/                     ssh / scp / ssh-copy-id wrappers, redfish, kubectl and grpcurl helpers
inventory/               instances and groups (hosts.yml), all tunables (group_vars/all.yml)
playbooks/               provision (Incus), site (configuration), frameworks, test, destroy
roles/                   one role per component (see the component pages)
fakegpu/                 fake NVIDIA userspace: C stubs, symbol lists, nvidia-smi
fakebmc/                 Redfish BMC service (Go): GPU tray and switch tray roles
fakenmxc/                NVLink partition controller and fabric telemetry (Go, gRPC)
fakeib/                  ibnetdiscover look-alike for the emulated InfiniBand fabric
fakedp/                  Kubernetes GPU device plugin (Go, CDI) for the k3s scheduler
examples/                example Slurm jobs (topology, scheduling, DDP, Ray); k8s/: the same as JobSets
tests/bmc/               BMC integration tests (pytest)
docs/                    component documentation
.secrets/                generated keys and passwords (git-ignored)
.cache/                  build tools, topograph checkout, k3s binary (git-ignored)
```

## Third-party software

| Software | Version | Licence | Used for |
|---|---|---|---|
| Slurm | 23.11 (Ubuntu) | GPL-2.0 | scheduler, accounting |
| OpenLDAP, SSSD, MariaDB, Redis | Ubuntu 24.04 | various OSS | identity, accounting DB, JuiceFS metadata |
| RustFS, rustfs/cli | 1.0.1, 0.1.36 | Apache-2.0 | S3 object storage, client |
| JuiceFS | 1.4.1 | Apache-2.0 | shared filesystem |
| topograph | pinned commit | Apache-2.0 | topology discovery |
| Prometheus, node_exporter, Grafana | Ubuntu / Grafana APT | Apache-2.0, AGPL-3.0 (Grafana) | monitoring |
| nvidia_gpu_exporter | 1.15.1 | MIT | GPU metrics |
| prometheus-slurm-exporter (rivosinc) | 1.8.0 | Apache-2.0 | Slurm metrics |
| k3s | 1.36 | Apache-2.0 | Kubernetes (k3s mode) |
| Kueue, JobSet, Node Feature Discovery, kube-state-metrics | pinned charts | Apache-2.0 | queueing and topology-aware scheduling, multi-pod jobs, node labels, cluster metrics (k3s mode) |
| NVIDIA GPU Feature Discovery (k8s-device-plugin image) | 0.20.1 | Apache-2.0 | GPU node labels (k3s mode) |
| PyTorch, Ray | latest at install | BSD-3, Apache-2.0 | frameworks |
| ZLUDA (layouts ported into `fakegpu/dark_api.c`) | | Apache-2.0 | `libcudart` export tables |

## Limitations

- The emulation stops at computation: GPU kernels do not run, so results are not numerically meaningful.
- Copies larger than 64 MiB are timed but not performed.
- Fake NCCL ranks do not communicate: a failed link or rank does not fail its peers' collectives.
- Containers share the host kernel, so Slurm jobs are not confined by cgroups.
- In k3s mode users may mount host paths in their pods, and GPUs come from the lab's own device plugin rather than NVIDIA's.
- The partition controller is not wire-compatible with NVIDIA's NMX-C (its `.proto` is proprietary).
- Slurm is 23.11 and LDAP has no TLS.
- The fabric is a single NVLink domain with one InfiniBand leaf.

Details are in each component page and in [Architecture](docs/architecture.md#limitations).

## Disclaimer

This is an independent, personal research and education project. It is not affiliated with, endorsed by, sponsored by or supported by NVIDIA Corporation.

- **Emulation, not NVIDIA software.** The lab imitates the interfaces of NVIDIA hardware and software (CUDA, NVML, NCCL, cuBLAS, cuDNN and `nvidia-smi` behaviour, GB200 BMC Redfish resources, NMX-C partition semantics) so that other software can be exercised without GPUs. It contains no NVIDIA source code, binaries, firmware or proprietary specifications: the stub libraries are written from publicly documented API names, the partition controller uses its own `.proto` rather than NVIDIA's, and the behaviour modelled comes from NVIDIA's public documentation and open-source reference implementations cited in each component page.
- **Not a substitute for real hardware.** Nothing is computed, and timings, telemetry and failure behaviour are approximations. Results obtained on the lab say nothing about the performance or correctness of real NVIDIA systems, and must not be presented as if they did.
- **Intended use.** The project is meant for personal research, learning and experimentation with cluster software. It is not intended, tested or supported for production or commercial use, and comes with no warranty of any kind (see the licence).
- **Trademarks.** NVIDIA, CUDA, NVLink, NVSwitch, Grace, Blackwell, GB200, BlueField, ConnectX, NCCL, NVML and other NVIDIA marks are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Slurm, Kubernetes, k3s and the other names used here are trademarks of their respective owners. They are used only to describe what is being emulated or integrated with.

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party components keep their own licences (see [Third-party software](#third-party-software)).
