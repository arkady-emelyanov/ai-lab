# Slurm lab: an emulated NVIDIA GB200 NVL8 GPU cluster

![The Lab banner](/docs/assets/social-preview.png)

A complete GPU cluster on one Linux machine, for building and testing cluster software, self-service tooling and operations without GPUs. It emulates one NVIDIA GB200-class **NVL8** NVLink domain (two GPU trays with four GPUs each, an NVLink switch tray, BMCs, an InfiniBand fabric) and runs the real software stack around it: Slurm with accounting, LDAP identity, shared and object storage, Prometheus and Grafana.

The GPUs are fake but behave like real ones to everything above them: applications launched through Slurm initialise PyTorch, NCCL or Ray, every call succeeds and takes realistic simulated time, and the GPUs report realistic load, memory, power and temperature. Nothing is computed. Management interfaces (Redfish BMCs, an NVLink partition controller, fabric telemetry) change what the GPUs report and what Slurm schedules on.

What you get:

- **Users and jobs:** LDAP users with SSH access, Slurm with GPU scheduling, accounting and NVLink-aware block topology; PyTorch DDP and Ray examples on 8 GPUs.
- **Hardware management:** Redfish BMCs per tray (power, NVLink ports) and for the switch tray (144 ports), an NMX-C-style partition controller, topology discovery with topograph.
- **Platform:** shared filesystem (JuiceFS on RustFS), per-user S3 buckets, node-local scratch, Prometheus with GPU, Slurm and NVLink fabric metrics, Grafana dashboards.
- **Operations:** one command to build, one to test, everything in Ansible.

## Use cases

- **Cluster software and self-service:** develop job portals, scheduler plugins, quota and accounting tools, or user-facing CLIs against a real Slurm, LDAP and S3 stack with GPU nodes.
- **Monitoring and operations:** build dashboards, alerts and runbooks on realistic GPU, Slurm and NVLink fabric metrics; rehearse node power cycles, NVLink failures and partition changes through Redfish and the partition controller.
- **Hardware management tooling:** test Redfish clients, BMC automation and topology-aware scheduling (topograph, `topology/block`) without a rack.
- **AI pipelines:** run PyTorch DDP and Ray jobs end to end on 8 GPUs with realistic timing and memory accounting, to test orchestration, data movement and failure handling rather than numerics.
- **CI for infrastructure code:** `make up && make test` builds and verifies the whole cluster from scratch on one machine.

## Architecture at a glance

```
                              your machine (Incus host)
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │  bin/ssh slurm ─► slurm-login    .11   login node, user shells                  │
 │                   slurm-control  .10   slurmctld, slurmdbd, LDAP, Prometheus,   │
 │                                        Grafana, topograph                       │
 │                   slurm-worker1  .21 ┐ GPU trays: slurmd, 4 × fake GB200 each   │
 │                   slurm-worker2  .22 ┘ one NVL8 NVLink domain                   │
 │                   slurm-storage  .12   RustFS (S3), Redis                       │
 │                   slurm-nvswitch .34   NVLink partition controller, telemetry   │
 │  bin/redfish ───► slurm-worker1-bmc .31, slurm-worker2-bmc .32,                 │
 │                   slurm-nvswitch-bmc .33   (Redfish BMCs)                       │
 │                                                                                 │
 │  /shared (homes, venv) · /pfs (JuiceFS) · /scratch (per tray) · S3 per user     │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

Addresses are on the Incus bridge (`10.107.111.<n>` here). Details: [Architecture](docs/architecture.md).

## Quickstart

**Requirements:** Linux with [Incus](https://linuxcontainers.org/incus/), `make`, Python 3, Go, `jq`, git; your user in the `incus-admin` group. The running cluster uses about 10 GiB of RAM (16 GiB free recommended; container memory limits add up to 27 GiB) and 4+ CPU cores; plan about 25 GB of disk under `/var/lib/incus`, including the 5.6 GB frameworks venv.

```
make init          # host side: Ansible venv, secrets, Go builds; reports any host fix needed
make up            # create and configure the cluster (~15 minutes)
make frameworks    # PyTorch + Ray on the cluster (optional, several GB)
make test          # end-to-end checks
```

`make init` prints the exact command for anything the host still needs (group membership, an AppArmor rule for Incus DNS, the inotify limit); see [Platform](docs/platform.md#troubleshooting).

**First steps:**

```
bin/scp examples/* slurm:                       # copy the example jobs to joe's home
bin/ssh slurm                                   # login node as joe (password: joe)
sinfo -N -o "%N %G %T"                          # two trays, gpu:gb200:4 each
srun -N2 --gpus-per-node=4 nvidia-smi -L        # all 8 GPUs
sbatch nvl8-hello.sbatch                        # one task per GPU across the domain
```

Then open Grafana at `http://10.107.111.10:3000` (user `admin`, password in `.secrets/grafana.pass`) and watch the **Slurm lab overview** while a job runs, for example `ddp-train.sbatch` from the [examples](docs/frameworks-and-examples.md).

## Lab endpoints

Reachable from the host machine. Addresses are on the Incus bridge (`10.107.111.0/24` here; yours may differ, see `incus network get incusbr0 ipv4.address`).

| Service | Endpoint | Credentials | Docs |
|---|---|---|---|
| Login node (SSH) | `bin/ssh slurm` (10.107.111.11:22) | `joe` / `joe`, or your key | [Identity and access](docs/identity-and-access.md) |
| Any instance as root (SSH) | `bin/ssh root@<instance>` | `.secrets/ssh/id_ed25519` | [Identity and access](docs/identity-and-access.md) |
| Grafana | http://10.107.111.10:3000 | `admin` / `.secrets/grafana.pass` | [Monitoring](docs/monitoring.md) |
| Prometheus | http://10.107.111.10:9090 | none | [Monitoring](docs/monitoring.md) |
| Slurm exporter | http://10.107.111.10:9092/metrics | none | [Monitoring](docs/monitoring.md) |
| topograph API | http://10.107.111.10:49021 | none | [Topology discovery](docs/topology.md) |
| RustFS S3 API | http://10.107.111.12:9000 | admin: `.secrets/rustfs.access` / `.secrets/rustfs.secret`; users: `<name>` / `.secrets/users/<name>.s3` | [Storage](docs/storage.md) |
| RustFS console | http://10.107.111.12:9001/rustfs/console/ | as S3 API | [Storage](docs/storage.md) |
| GPU tray BMCs (Redfish) | https://10.107.111.31, https://10.107.111.32 (`bin/redfish slurm-worker1 …`) | `root` / `0penBmc` | [BMCs](docs/bmc-redfish.md) |
| NVLink switch tray BMC (Redfish) | https://10.107.111.33 (`bin/redfish slurm-nvswitch …`) | `root` / `0penBmc` | [BMCs](docs/bmc-redfish.md) |
| NVLink partition controller (gRPC) | 10.107.111.34:9370 (plaintext, reflection) | none | [NVLink partitions](docs/nvlink-partitions.md) |
| Fabric telemetry | http://10.107.111.34:9372/metrics | none | [NVLink partitions](docs/nvlink-partitions.md) |
| GPU exporters | http://10.107.111.21:9835/metrics, http://10.107.111.22:9835/metrics | none | [Monitoring](docs/monitoring.md) |
| node_exporter, JuiceFS metrics | `<Slurm node>:9100/metrics`, `<Slurm node>:9567/metrics` | none | [Monitoring](docs/monitoring.md) |

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
| | Emulated InfiniBand fabric and topograph-generated Slurm topology | [Topology discovery](docs/topology.md) |
| Platform services | OpenLDAP, SSSD, SSH, users, `bin/ssh` | [Identity and access](docs/identity-and-access.md) |
| | `/shared`, `/pfs` (JuiceFS), `/scratch`, RustFS S3 with per-user buckets | [Storage](docs/storage.md) |
| | Prometheus, exporters (node, GPU, Slurm, NVLink, JuiceFS), Grafana | [Monitoring](docs/monitoring.md) |
| Scheduling | Slurm, accounting, GPU GRES, block topology, scratch | [Slurm](docs/slurm.md) |
| Applications | PyTorch and Ray venv, example jobs | [Frameworks and examples](docs/frameworks-and-examples.md) |
| Quality | `make test` coverage | [Testing](docs/testing.md) |

## Everyday commands

| Command | Purpose |
|---|---|
| `make up`, `make configure` | build the cluster; re-apply configuration after changing `inventory/group_vars/all.yml` |
| `make test` | end-to-end checks |
| `make down`, `make purge` | delete instances (keep volumes); delete everything |
| `bin/ssh slurm`, `bin/ssh root@<instance>` | SSH as joe / root; `bin/scp`, `bin/ssh-copy-id` likewise |
| `make shell`, `make shell-<instance>` | shells through `incus exec` |
| `bin/redfish <tray> <path> [curl args]` | Redfish requests to a BMC |
| `bin/ssh slurm-control update-topology` | regenerate Slurm's topology now (a timer does it every minute) |

## Repository layout

```
Makefile                 entry points (init, up, configure, frameworks, test, down, purge, shell)
bin/                     ssh / scp / ssh-copy-id wrappers, redfish helper
inventory/               instances and groups (hosts.yml), all tunables (group_vars/all.yml)
playbooks/               provision (Incus), site (configuration), frameworks, test, destroy
roles/                   one role per component (see the component pages)
fakegpu/                 fake NVIDIA userspace: C stubs, symbol lists, nvidia-smi
fakebmc/                 Redfish BMC service (Go): GPU tray and switch tray roles
fakenmxc/                NVLink partition controller and fabric telemetry (Go, gRPC)
fakeib/                  ibnetdiscover look-alike for the emulated InfiniBand fabric
examples/                example Slurm jobs (topology, scheduling, DDP, Ray)
docs/                    component documentation
.secrets/                generated keys and passwords (git-ignored)
.cache/                  build tools and the topograph checkout (git-ignored)
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
| PyTorch, Ray | latest at install | BSD-3, Apache-2.0 | frameworks |
| ZLUDA (layouts ported into `fakegpu/dark_api.c`) | | Apache-2.0 | `libcudart` export tables |

## Limitations

The emulation stops at computation: GPU kernels do not run, so results are not numerically meaningful, and copies larger than 64 MiB are timed but not performed. Containers share the host kernel, so Slurm jobs are not confined by cgroups. Slurm is 23.11, LDAP has no TLS, and the fabric is a single NVLink domain with one InfiniBand leaf. Details are in each component page and in [Architecture](docs/architecture.md#limitations).

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party components keep their own licences (see [Third-party software](#third-party-software)).
