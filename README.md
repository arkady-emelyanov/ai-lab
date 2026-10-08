# AI lab: a GPU cluster modelled on GB200 NVL72, scaled down to two compute trays (NVL8)

![AI lab overview: the platform instances (login, control, storage) and the NVL8 NVLink domain: an NVLink switch tray with two NVSwitches, two GPU trays with four GB200 each, a Redfish BMC per tray, an InfiniBand leaf and spine](/docs/assets/overview.png)

> **Independent research project, not affiliated with or endorsed by NVIDIA.** It emulates NVIDIA hardware and software interfaces in software, for personal research and education. See [Disclaimer](#disclaimer).

A complete GPU cluster on a single Linux machine, for building and testing cluster software, self-service tooling and operations without GPUs. It is modelled on NVIDIA GB200 NVL72, scaled down to two compute trays: one NVLink domain the lab calls **NVL8** (its own name, not an NVIDIA product), with two GPU trays of four GPUs each, an NVLink switch tray, BMCs and an InfiniBand fabric. Around it runs the real software stack: Slurm with accounting or Kubernetes (k3s) with Kueue, LDAP identity, shared and object storage, Prometheus and Grafana.

It is software-in-the-loop: real, unmodified software (schedulers, frameworks, tools) runs against a behavioural model of the hardware. PyTorch, NCCL and Ray jobs start, run and report GPU load, memory, power and temperature as on real GPUs, but GPU kernels never actually execute: there is no real GPU math. Details in [What is real and what is modelled](#what-is-real-and-what-is-modelled).

A blog series walks through the lab, starting with [AI lab, part 1: a GB200 NVL72-style cluster, scaled down to your laptop](https://blog.emelianov.cloud/ai-lab/01-intro/).

## Use cases

- **Cluster software and self-service:** develop job portals, scheduler plugins, quota and accounting tools, or user-facing CLIs against a real Slurm or Kubernetes, LDAP and S3 stack with GPU nodes.
- **Monitoring and operations:** build dashboards, alerts and runbooks on live GPU, scheduler and NVLink fabric metrics; rehearse node power cycles, NVLink failures and partition changes through Redfish and the partition controller.
- **Hardware management tooling:** test Redfish clients, BMC automation and topology-aware scheduling (topograph with Slurm `topology/block` or Kueue node labels) without a rack.
- **AI pipelines:** run PyTorch DDP and Ray jobs end to end on 8 GPUs with modelled timing and memory accounting, to test orchestration, data movement and failure handling rather than numerics.
- **CI for infrastructure code:** `make up && make test` builds and verifies the whole cluster from scratch on one machine.

**Not for:**

- benchmarking or capacity planning from the lab's timings
- checking numerical results or model accuracy
- developing or tuning CUDA kernels
- rehearsing hardware failures the lab does not model ([What is real and what is modelled](#what-is-real-and-what-is-modelled))

## What is real and what is modelled

AI lab models what the hardware shows to software (APIs, topology, telemetry, timing, failures), not the silicon. Everything above that boundary is real software, unmodified.

| Component | Real or modelled | Faithful | Not modelled |
|---|---|---|---|
| Slurm, k3s, Kueue, JobSet, topograph, GPU Feature Discovery | real | configuration, scheduling, accounting, topology-aware placement | Slurm jobs are not confined by cgroups (containers share the host kernel); Slurm is 23.11, without `--segment` and `BlockSizes`; k3s gets its GPUs from the lab's own device plugin, not NVIDIA's, and users may mount host paths in their pods |
| PyTorch, Ray, your applications | real, unmodified | initialisation, process groups, control flow, the errors the APIs below return | numerical results |
| CUDA driver and runtime, cuBLAS, cuDNN | modelled ([`fakegpu`](docs/fake-gpu.md)) | device properties, contexts, streams, events, allocations up to the GPU's memory (out-of-memory beyond it), timing | kernels do not run; copies over 64 MiB are timed, not performed |
| NVML, `nvidia-smi` | modelled ([`fakegpu`](docs/fake-gpu.md)) | identity, PCIe, NVLink state, fabric clique, NUMA layout, processes, GPU reset; utilisation, power and temperature from the load | power and thermals follow a model of load, not measured curves |
| NCCL | modelled ([`fakegpu`](docs/fake-gpu.md)) | communicators and splits, collectives timed over NVLink inside a partition and InfiniBand across partitions | data is not exchanged; a failed peer or link does not fail collectives |
| NVLink and NVSwitch | modelled ([partition controller](docs/nvlink-partitions.md), BMCs) | 18 links per GPU to two 72-port switches, partitions and cliques applied at GPU reset, disabled links, fabric telemetry | NMX-C's wire protocol; one NVL8 domain only |
| BMCs | modelled ([Redfish](docs/bmc-redfish.md)) | GB200 resource layout, power actions that stop and start the tray, GPU sensors, firmware inventory | IPMI, BlueField DPUs |
| InfiniBand | modelled ([topology](docs/topology.md)) | `ibnetdiscover` topology for topograph, NIC byte counters, 400 Gb/s in NCCL timing | packets, subnet manager; one leaf switch |
| Grace CPUs | modelled (NUMA layout only) | two Grace NUMA nodes per tray in `nvidia-smi topo -m` and `numactl -H` | the trays run on the host's x86-64 cores, not Grace (aarch64): `uname -m` and `scontrol show node` say `x86_64` |
| Prometheus, Grafana, LDAP, JuiceFS, RustFS (S3) | real | everything | the lab runs LDAP without TLS |

**Timing is a behavioural model, not a prediction.** Each operation's duration comes from its size and NVIDIA's published GB200 figures (compute per precision, memory, NVLink and InfiniBand bandwidth). Jobs take plausible time and put plausible load on the GPUs, but the model is not calibrated against hardware and does not predict real GB200 performance. Details: [Emulated GPUs](docs/fake-gpu.md#how-it-works).

Details are in each component page and in [Architecture](docs/architecture.md#limitations).

## Quickstart

> [!TIP]
> **Using an AI coding agent?** Ask it to set up the lab: [AGENTS.md](AGENTS.md) tells it how (host checks, the `sudo` steps with their reasons, build, hand-over).
>
> To work with the running lab (jobs, GPUs, NVLink partitions, BMCs, metrics), agents use the [ai-lab skill](skills/ai-lab/SKILL.md). Claude Code picks it up in this repository; to have it in other projects too:
>
> ```
> mkdir -p ~/.claude/skills/ai-lab
> curl -fsSL https://raw.githubusercontent.com/arkady-emelyanov/ai-lab/main/skills/ai-lab/SKILL.md -o ~/.claude/skills/ai-lab/SKILL.md
> ```

**Requirements:**

- x86-64 Linux with [Incus](https://linuxcontainers.org/incus/), `make`, Python 3, Go, `jq`, git.
- Your user in the `incus-admin` group.
- RAM: the running cluster uses about 10 GiB; 16 GiB free recommended. Container memory limits add up to 27 GiB (32 GiB with k3s).
- CPU: 4 or more cores.
- Disk: about 25 GB under `/var/lib/incus`, including the 5.6 GB frameworks venv.

```
make init          # host side: Ansible venv, secrets, Go builds; reports any host fix needed
make up            # create and configure the cluster (~15-25 minutes)
make frameworks    # PyTorch + Ray on the cluster (optional, several GB)
make test          # end-to-end checks
```

`make init` prints the exact command for anything the host still needs (group membership, an AppArmor rule for Incus DNS, the inotify and kernel keyring limits); see [Platform](docs/platform.md#troubleshooting).

**Your settings:** `make init` creates `local.yml` (git-ignored). It overrides `inventory/group_vars/all.yml`: put anything you change there.

**Kubernetes instead of Slurm:** set `scheduler: k3s` in `local.yml` before `make up` (on a built cluster: `make down`, change it, `make up`; volumes are kept). See [Kubernetes (k3s)](docs/kubernetes.md).

**First steps:**

```
bin/scp -r examples login:                      # copy the example jobs to joe's home
bin/ssh login                                   # login node as joe (joe's lab key, no password)
```

With Slurm ([Slurm](docs/slurm.md#usage)):

```
cd examples/slurm
sinfo -N -o "%N %G %T"                          # two trays, gpu:gb200:4 each
srun -N2 --gpus-per-node=4 nvidia-smi -L        # all 8 GPUs
sbatch nvl8-hello.sbatch                        # one task per GPU across the domain
```

With Kubernetes ([Kubernetes](docs/kubernetes.md#usage)):

```
cd examples/kubernetes
kubectl get nodes -L nvidia.com/gpu.clique      # two trays, 4 GPUs and an NVLink clique each
./submit --wait nvl8-hello.yaml                 # one pod per GPU across the domain
```

Then open Grafana at `http://10.107.111.10:3000` (user `admin`, password in `.secrets/grafana.pass`) and watch the **Lab overview** while a job runs, for example `ddp-train` from the [examples](examples/README.md).

## Documentation

- **Infrastructure:** [Platform](docs/platform.md) (make targets, configuration, troubleshooting) · [Architecture](docs/architecture.md) · [Lab endpoints](docs/endpoints.md) (addresses and credentials)
- **Emulated hardware:** [Emulated GPUs](docs/fake-gpu.md) · [BMCs](docs/bmc-redfish.md) · [NVLink partitions](docs/nvlink-partitions.md) · [Topology discovery](docs/topology.md)
- **Platform services:** [Identity and access](docs/identity-and-access.md) · [Storage](docs/storage.md) · [Monitoring](docs/monitoring.md)
- **Scheduling:** [Slurm](docs/slurm.md) · [Kubernetes (k3s)](docs/kubernetes.md)
- **Applications and tests:** [Frameworks and examples](docs/frameworks-and-examples.md) · [Testing](docs/testing.md)

## Everyday commands

| Command | Purpose |
|---|---|
| `make up`, `make configure` | build the cluster; re-apply configuration after changing `local.yml` |
| `make test` | end-to-end checks |
| `make test-fakegpu` | fake GPU library tests on this machine, no lab needed |
| `make test-bmc` | BMC integration tests (`-disruptive`, `-conformance` tiers) |
| `make down`, `make purge` | delete instances (keep volumes); delete everything |
| `bin/ssh login`, `bin/ssh root@<instance>` | SSH as joe / root; `bin/scp`, `bin/ssh-copy-id` likewise |
| `bin/kubectl <args>` | kubectl as cluster admin (k3s mode) |
| `bin/nvlink <command>` | NVLink domain and partitions: `domain`, `gpus`, `topology`, `partitions`, `create`, `delete`, `add`, `remove` |
| `bin/grpcurl <args>` | grpcurl for the NVLink partition controller's raw API (built on first use) |
| `make shell`, `make shell-<instance>` | shells through `incus exec` |
| `bin/redfish <tray> <path> [curl args]` | Redfish requests to a BMC |
| `bin/ssh sched-control update-topology` | regenerate the scheduler's topology now (a timer does it every minute) |

## Repository layout

```
AGENTS.md                instructions for AI coding agents (deploying the lab for a user; CLAUDE.md points to it)
skills/ai-lab/           installable agent skill for operating the lab (jobs, GPUs, partitions, BMCs, metrics)
local.yml                your settings, overriding inventory/group_vars/all.yml (created by make init, not in git)
Makefile                 entry points (init, up, configure, frameworks, test, down, purge, shell)
bin/                     ssh / scp / ssh-copy-id wrappers, redfish, nvlink, kubectl and grpcurl helpers
inventory/               instances and groups (hosts.yml), all tunables and their defaults (group_vars/all.yml)
playbooks/               provision (Incus), site (configuration), frameworks, test, destroy
roles/                   one role per component (see the component pages)
fakegpu/                 fake NVIDIA userspace: C stubs, symbol lists, nvidia-smi
fakebmc/                 Redfish BMC service (Go): GPU tray and switch tray roles
fakenmxc/                NVLink partition controller and fabric telemetry (Go, gRPC)
fakeib/                  ibnetdiscover look-alike for the emulated InfiniBand fabric
fakedp/                  Kubernetes GPU device plugin (Go, CDI) for the k3s scheduler
examples/                the same four jobs (topology, scheduling, DDP, Ray) for slurm/ and kubernetes/
tests/bmc/               BMC integration tests (pytest)
tests/fakegpu/           fake GPU library tests on the host, no lab needed (pytest)
docs/                    component documentation
.secrets/                generated keys and passwords (git-ignored)
.cache/                  build tools, topograph checkout, k3s binary (git-ignored)
```

## Disclaimer

This is an independent, personal research and education project. It is not affiliated with, endorsed by, sponsored by or supported by NVIDIA Corporation.

- **Emulation, not NVIDIA software.** The lab imitates the interfaces of NVIDIA hardware and software (CUDA, NVML, NCCL, cuBLAS, cuDNN and `nvidia-smi` behaviour, GB200 BMC Redfish resources, NMX-C partition semantics) so that other software can be exercised without GPUs. It contains no NVIDIA source code, binaries, firmware or proprietary specifications: the stub libraries are the lab's own code exporting the same function names as NVIDIA's libraries (so programs link and load; `fakegpu/symbols/*.syms` are those names, read from the libraries' public dynamic symbol tables, and the CUDA runtime's undocumented export-table layouts come from the open-source [ZLUDA](https://github.com/vosen/ZLUDA) project), the partition controller uses its own `.proto` rather than NVIDIA's, and the behaviour modelled comes from NVIDIA's public documentation and open-source reference implementations cited in each component page.
- **Not a substitute for real hardware.** GPU kernels never actually execute (no real GPU math), and timings, telemetry and failure behaviour are approximations. Results obtained on the lab say nothing about the performance or correctness of real NVIDIA systems, and must not be presented as if they did.
- **Intended use.** The project is meant for personal research, learning and experimentation with cluster software. It is not intended, tested or supported for production or commercial use, and comes with no warranty of any kind (see the licence).
- **Trademarks.** NVIDIA, CUDA, NVLink, NVSwitch, Grace, Blackwell, GB200, BlueField, ConnectX, NCCL, NVML and other NVIDIA marks are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Slurm, Kubernetes, k3s and the other names used here are trademarks of their respective owners. They are used only to describe what is being emulated or integrated with.

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party software the lab installs keeps its own licence; the code ported from ZLUDA (Apache-2.0) is credited in `fakegpu/dark_api.c`.
