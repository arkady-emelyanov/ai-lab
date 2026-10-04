[← README](../README.md) · **Architecture**

# Architecture

## Overview

The lab is a set of Incus system containers on one Linux host, configured by Ansible into a GPU cluster that behaves like one GB200-class **NVL8** NVLink domain: two GPU compute trays with four GPUs each, an NVLink switch tray, management controllers (BMCs) and the services around a production Slurm cluster. The GPUs, the NVLink fabric and the InfiniBand network are emulated; everything else (Slurm, LDAP, storage, monitoring) is real software.

Device nodes, NVML and CUDA behaviour, Redfish resources, partition semantics and metrics are modelled on NVIDIA hardware and its management stack; emulation shortcuts are listed under [Limitations](#limitations).

## Layers

```
 ┌─ Users & applications ────────────────────────────────────────────────────┐
 │  joe (LDAP) · bin/ssh · sbatch/srun · PyTorch DDP · Ray · S3 clients       │
 ├─ Scheduling & accounting ─────────────────────────────────────────────────┤
 │  Slurm 23.11 (slurmctld, slurmd, slurmdbd+MariaDB) · topology/block        │
 ├─ Platform services ───────────────────────────────────────────────────────┤
 │  OpenLDAP+SSSD · JuiceFS /pfs · RustFS S3 · Prometheus+Grafana · topograph │
 ├─ Emulated hardware ───────────────────────────────────────────────────────┤
 │  fakegpu (CUDA/NVML/libs) · NVSwitch tray (fakenmxc) · BMCs (fakebmc)      │
 │  InfiniBand fabric (fake ibnetdiscover)                                    │
 ├─ Infrastructure ──────────────────────────────────────────────────────────┤
 │  Incus containers, volumes, bridge · Ansible playbooks · make targets      │
 └───────────────────────────────────────────────────────────────────────────┘
```

## Instances

| Instance | Address | Role | Incus project | Docs |
|---|---|---|---|---|
| `slurm-control` | 10.107.111.10 | slurmctld, slurmdbd + MariaDB, OpenLDAP, Prometheus, Grafana, Slurm exporter, topograph | default | [Slurm](slurm.md), [Identity](identity-and-access.md), [Monitoring](monitoring.md), [Topology](topology.md) |
| `slurm-login` | 10.107.111.11 | login node: Slurm client, tools, user shells | default | [Identity & access](identity-and-access.md) |
| `slurm-storage` | 10.107.111.12 | RustFS (S3), Redis (JuiceFS metadata) | default | [Storage](storage.md) |
| `slurm-worker1`, `slurm-worker2` | .21, .22 | GPU trays: slurmd, 4 fake GB200 each, GPU exporter | `trays` | [Fake GPUs](fake-gpu.md), [Slurm](slurm.md) |
| `slurm-worker1-bmc`, `slurm-worker2-bmc` | .31, .32 | Redfish BMC per GPU tray | default | [BMCs](bmc-redfish.md) |
| `slurm-nvswitch-bmc` | .33 | Redfish BMC of the NVLink switch tray | default | [BMCs](bmc-redfish.md) |
| `slurm-nvswitch` | .34 | switch tray host: NVLink partition controller and fabric telemetry | default | [NVLink partitions](nvlink-partitions.md) |

Addresses are pinned on the Incus bridge (`incusbr0`, `10.107.111.0/24` here; the last octet comes from `ip_host` in `inventory/hosts.yml`). Every instance has the others in `/etc/hosts`. The instance name is also the hostname, the Slurm node name and the `bin/ssh` alias.

## How the pieces connect

**Sideband and telemetry volumes.** Real BMCs and switch trays talk to GPUs over out-of-band links (I2C/MCTP, NVLink management). The lab stands these in with small Incus volumes:

| Volume (per tray) | Writers | Readers | Content |
|---|---|---|---|
| `sideband-<tray>` | tray BMC (`nvlink-disabled`), switch BMC (`nvlink-disabled-switch`), partition controller (`fabric-clique`) | the tray's fake NVML (read-only mount) | NVLink port state, partition (clique) per GPU |
| `telemetry-<tray>` | every CUDA process on the tray (`occupancy`) | partition controller on `slurm-nvswitch` (read-only) | GPU processes, memory, busy time, NVLink traffic |

**Power control.** GPU trays live in the Incus project `trays`. Each tray BMC holds an Incus client certificate restricted to that project and drives the tray's power through the Incus API on the bridge address (`:8443`); it cannot reach any other instance.

**Data paths.**

```
 job (srun) ─► CUDA/NCCL calls ─► fakegpu: simulated time + occupancy ─► NVML ─► nvidia-smi, GPU exporter ─► Prometheus
                                                     │
                                                     └─ telemetry volume ─► fakenmxc /metrics ─► Prometheus
 Redfish PATCH (BMC) / gRPC (fakenmxc) ─► sideband volume ─► NVML link state, clique ─► topograph ─► Slurm topology.conf
```

## Storage volumes

| Volume / pool | Mounted at | Purpose |
|---|---|---|
| `cluster-shared` (pool `default`) | `/shared` on Slurm nodes | homes (`/shared/home/<user>`), frameworks venv |
| `scratch-<tray>` (pool `local-nvme`, btrfs, 50 GiB quota) | `/scratch` on each tray | node-local scratch, JuiceFS cache |
| `rustfs-data` (pool `default`) | `/var/lib/rustfs` on `slurm-storage` | object data |
| `sideband-<tray>`, `telemetry-<tray>` | see above | emulated out-of-band links |

## Limitations

- GPU kernels do not compute; tensors computed on the GPU hold zeros or garbage. Copies above `fakegpu_copy_max_mb` (64 MiB) are timed but not performed, so large GPU buffers cost no host RAM. See [Fake GPUs](fake-gpu.md#limitations).
- Containers share the host kernel: Slurm tracks jobs by process (`proctrack/linuxproc`) without cgroup confinement or CPU binding.
- Slurm is the Ubuntu 24.04 package (23.11): no `--segment`, no `BlockSizes`.
- LDAP runs without TLS inside the lab network.
- One NVLink domain, one switch tray, one InfiniBand leaf: enough to exercise every interface, not a scale model.

## References

- [Incus documentation](https://linuxcontainers.org/incus/docs/main/): instances, projects, storage volumes, restricted certificates
- [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/): the system family the lab models
- Component pages linked from the [README](../README.md#layers-and-components)
