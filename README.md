# AI lab: a GPU cluster modelled on GB200 NVL72, scaled down to two compute trays (NVL8)

![AI lab overview: the platform instances (login, control, storage) and the NVL8 NVLink domain: an NVLink switch tray with two NVSwitches, two GPU trays with four GB200 each, a Redfish BMC per tray, an InfiniBand leaf and spine](/docs/assets/overview.png)

> **Independent research project, not affiliated with or endorsed by NVIDIA.** It emulates NVIDIA hardware and software interfaces in software, for personal research and education. See [Disclaimer](#disclaimer).

It is software-in-the-loop: real, unmodified software (schedulers, frameworks, tools) runs against a behavioural model of the hardware. PyTorch, NCCL and Ray jobs start, run and report GPU load, memory, power and temperature as on real GPUs, but GPU kernels never actually execute: there is no real GPU math. Details in [What is real and what is modelled](docs/real-and-modelled.md).

A blog series walks through the lab, starting with [AI lab, part 1: a GB200 NVL72-style cluster, scaled down to your laptop](https://blog.emelianov.cloud/ai-lab/01-intro/).

## Use cases

- **Cluster software and self-service:** develop job portals, scheduler plugins, quota and accounting tools, or user-facing CLIs against a real Slurm or Kubernetes, LDAP and S3 stack with GPU nodes.
- **Monitoring and operations:** build dashboards, alerts and runbooks on live GPU, scheduler and NVLink fabric metrics; rehearse node power cycles, NVLink failures and partition changes through Redfish and the partition controller.
- **Hardware management tooling:** test Redfish clients, BMC automation and topology-aware scheduling (topograph with Slurm `topology/block` or Kueue node labels) without a rack.
- **AI pipelines:** run PyTorch DDP and Ray jobs end to end on 8 GPUs with modelled timing and memory accounting, to test orchestration, data movement and failure handling rather than numerics.
- **CI for infrastructure code:** `make up && make test` builds and verifies the whole cluster from scratch on one machine.

**Not for:**

- benchmarking or capacity planning
- checking numerical results or model accuracy
- developing or tuning CUDA kernels
- rehearsing hardware failures the lab does not model

## Quickstart

> [!TIP]
> **Using an AI coding agent?** Ask it to set up the lab: [AGENTS.md](AGENTS.md) tells it how (host checks, the `sudo` steps with their reasons, build, hand-over).
>
> To work with the running lab (jobs, GPUs, NVLink partitions, BMCs, metrics), agents use the [ai-lab skill](skills/ai-lab/SKILL.md). Claude Code picks it up in this repository; to have it in other projects too:
>

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

`make init` prints the exact fix for anything the host still needs ([Platform](docs/platform.md#troubleshooting)) and creates `local.yml`, your git-ignored settings, which override `inventory/group_vars/all.yml`. The lab runs Slurm by default, for Kubernetes, set `scheduler: k3s` before `make up` ([Kubernetes](docs/kubernetes.md)).

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

- **What the lab models:** [What is real and what is modelled](docs/real-and-modelled.md)
- **Infrastructure:** [Platform](docs/platform.md) (make targets, configuration, troubleshooting) · [Architecture](docs/architecture.md) · [Lab endpoints](docs/endpoints.md) (addresses and credentials)
- **Emulated hardware:** [Emulated GPUs](docs/fake-gpu.md) · [BMCs](docs/bmc-redfish.md) · [NVLink partitions](docs/nvlink-partitions.md) · [Topology discovery](docs/topology.md)
- **Platform services:** [Identity and access](docs/identity-and-access.md) · [Storage](docs/storage.md) · [Monitoring](docs/monitoring.md)
- **Scheduling:** [Slurm](docs/slurm.md) · [Kubernetes (k3s)](docs/kubernetes.md)
- **Applications and tests:** [Frameworks and examples](docs/frameworks-and-examples.md) · [Testing](docs/testing.md)

## Everyday commands

| Command | Purpose |
|---|---|
| `make up` | build the lab |
| `make configure` | apply changed settings (`local.yml`) to the running lab |
| `make test` | end-to-end checks ([Testing](docs/testing.md) has the rest) |
| `make down`, `make purge` | delete the instances (volumes kept); delete everything |
| `bin/ssh login`, `bin/ssh root@<instance>` | log in as joe, or as root anywhere |
| `bin/kubectl` | kubectl as cluster admin (k3s mode) |
| `bin/nvlink` | NVLink domain and partitions: list, create, delete, add or remove GPUs |
| `bin/redfish <tray> <path>` | Redfish request to a tray's BMC |

## Disclaimer

This is an independent, personal research and education project. It is not affiliated with, endorsed by, sponsored by or supported by NVIDIA Corporation.

- **Emulation, not NVIDIA software.** The lab imitates the interfaces of NVIDIA hardware and software (CUDA, NVML, NCCL, cuBLAS, cuDNN, `nvidia-smi`, GB200 BMC Redfish resources, NMX-C partitions) so that other software can be exercised without GPUs. It contains no NVIDIA source code, binaries, firmware or proprietary specifications:
  - The stub libraries are the lab's own code. They export the same function names as NVIDIA's libraries, so programs link and load; `fakegpu/symbols/*.syms` lists those names, read from the libraries' public symbol tables.
  - The CUDA runtime's undocumented export-table layouts come from the open-source [ZLUDA](https://github.com/vosen/ZLUDA) project.
  - The partition controller uses its own `.proto`, not NVIDIA's.
  - The behaviour modelled comes from NVIDIA's public documentation and open-source reference implementations, cited in each component page.
- **Not a substitute for real hardware.** GPU kernels never actually execute (no real GPU math), and timings, telemetry and failure behaviour are approximations. Results obtained on the lab say nothing about the performance or correctness of real NVIDIA systems, and must not be presented as if they did.
- **Intended use.** The project is meant for personal research, learning and experimentation with cluster software. It is not intended, tested or supported for production or commercial use, and comes with no warranty of any kind (see the licence).
- **Trademarks.** NVIDIA, CUDA, NVLink, NVSwitch, Grace, Blackwell, GB200, BlueField, ConnectX, NCCL, NVML and other NVIDIA marks are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Slurm, Kubernetes, k3s and the other names used here are trademarks of their respective owners. They are used only to describe what is being emulated or integrated with.

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party software the lab installs keeps its own licence; the code ported from ZLUDA (Apache-2.0) is credited in `fakegpu/dark_api.c`.
