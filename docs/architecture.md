[← README](../README.md) · **Architecture**

# Architecture

## Overview

The lab is a set of Incus system containers on one Linux host, configured by Ansible into a GPU cluster. It is modelled on NVIDIA GB200 NVL72, scaled down to two compute trays: one NVLink domain the lab calls **NVL8** (its own name, not an NVIDIA product). It has:

- two GPU trays with four GPUs each, an NVLink switch tray and their BMCs
- the services around a production GPU cluster, scheduled by Slurm or by Kubernetes (k3s), one or the other

The GPUs, the NVLink fabric and the InfiniBand network are emulated; everything else (Slurm or k3s, LDAP, storage, monitoring) is real software.

Device nodes, NVML and CUDA behaviour, Redfish resources, partition semantics and metrics are modelled on NVIDIA hardware and its management stack; emulation shortcuts are listed under [Limitations](#limitations).

## Topology

### Hosts and the management network

Every instance sits on the Incus bridge; your machine reaches them through the `bin/` wrappers and a browser. The scheduler daemons are Slurm's or k3s', never both.

```
 your machine (Incus host): bin/ssh · bin/kubectl · bin/redfish · bin/nvlink · browser (Grafana, Prometheus)
        │
 ═══════╧════════════ management Ethernet: Incus bridge incusbr0, 10.107.111.0/24 ═══════════════════════
        │                   │                   │                   │                      │
 ┌──────┴───────┐ ┌─────────┴────────┐ ┌────────┴───────┐ ┌─────────┴──────────┐ ┌─────────┴──────────┐
 │ sched-login  │ │ sched-control    │ │ sched-storage  │ │ sched-worker1  .21 │ │ sched-nvswitch .34 │
 │ .11          │ │ .10              │ │ .12            │ │ sched-worker2  .22 │ │ switch tray host:  │
 │ user shells: │ │ slurmctld+dbd or │ │ RustFS (S3)    │ │ GPU trays, project │ │ partition control- │
 │ sbatch/srun  │ │ k3s server, LDAP,│ │ Redis (JuiceFS │ │ `trays`: 4 × GB200,│ │ ler (NMX-C-style), │
 │ or kubectl   │ │ Prometheus, Gra- │ │ metadata)      │ │ slurmd or k3s      │ │ fabric telemetry   │
 │              │ │ fana, topograph  │ │                │ │ agent, GPU exporter│ │                    │
 └──────────────┘ └──────────────────┘ └────────────────┘ └────────────────────┘ └────────────────────┘
```

The BMCs share the bridge too (in a real rack they have a separate out-of-band network): `sched-worker1-bmc` .31, `sched-worker2-bmc` .32 and `sched-nvswitch-bmc` .33. They power their trays and switch NVLink ports on and off; `bin/redfish` and the Redfish exporter on `sched-control` talk to them.

### Compute fabrics

Two emulated fabrics connect the GPUs: InfiniBand between the trays (what `ibnetdiscover` reports) and NVLink through the switch tray (what NVML, the switch BMC and the partition controller report). topograph reads both.

```
                 ┌──────────────────────────┐
                 │ spine  LAB-IBSPINE-01    │      InfiniBand: between the trays
                 └────────────┬─────────────┘
                              │ 4 uplinks
                 ┌────────────┴─────────────┐
                 │ leaf   LAB-IBLEAF-01     │
                 └──────┬────────────┬──────┘
               4 × NDR  │            │  4 × NDR
      ┌─────────────────┴──┐      ┌──┴─────────────────┐
      │ sched-worker1      │      │ sched-worker2      │
      │ HCA mlx5_0 … _3    │      │ HCA mlx5_0 … _3    │
      │ GPU0 GPU1 GPU2 GPU3│      │ GPU0 GPU1 GPU2 GPU3│
      └─────────┬──────────┘      └──────────┬─────────┘
                │   each GPU: 18 NVLink5 links, 9 to each switch
                └──────────────┬─────────────┘
                               │ 8 GPUs × 18 = 144 links
              ┌────────────────┴────────────────────────┐
              │ NVLink switch tray                      │      NVLink: inside the domain
              │ NVSwitch_0 (72 ports)  NVSwitch_1 (72)  │
              │ partitions: sched-nvswitch (NMX-C-like) │
              └─────────────────────────────────────────┘
```

A real GB200 NVL72 rack spreads each GPU's 18 links over 9 switch trays (one link per NVSwitch chip); the lab models one switch tray, so each GPU uses 9 ports on each of its two chips.

## Instances

| Instance | Address | Role | Incus project | Host cores | Docs |
|---|---|---|---|---|---|
| `sched-control` | 10.107.111.10 | slurmctld, slurmdbd + MariaDB, Slurm exporter (or k3s server and add-ons, kube-state-metrics), OpenLDAP, Prometheus, Redfish exporter, Grafana, topograph | default | shared | [Slurm](slurm.md), [Kubernetes](kubernetes.md), [Identity](identity-and-access.md), [Monitoring](monitoring.md), [Topology](topology.md) |
| `sched-login` | 10.107.111.11 | login node: Slurm client or kubectl, tools, user shells | default | shared | [Identity & access](identity-and-access.md) |
| `sched-storage` | 10.107.111.12 | RustFS (S3), Redis (JuiceFS metadata) | default | shared | [Storage](storage.md) |
| `sched-worker1`, `sched-worker2` | .21, .22 | GPU trays: slurmd (or k3s agent and device plugin), 4 emulated GB200 each, GPU exporter | `trays` | own, per tray | [Emulated GPUs](fake-gpu.md), [Slurm](slurm.md), [Kubernetes](kubernetes.md) |
| `sched-worker1-bmc`, `sched-worker2-bmc` | .31, .32 | Redfish BMC per GPU tray | default | shared | [BMCs](bmc-redfish.md) |
| `sched-nvswitch-bmc` | .33 | Redfish BMC of the NVLink switch tray | default | shared | [BMCs](bmc-redfish.md) |
| `sched-nvswitch` | .34 | switch tray host: NVLink partition controller and fabric telemetry | default | shared | [NVLink partitions](nvlink-partitions.md) |

Addresses are pinned on the Incus bridge (`incusbr0`, `10.107.111.0/24` here; the last octet comes from `ip_host` in `inventory/hosts.yml`). Every instance has the others in `/etc/hosts`. The instance name is also the hostname, the Slurm or Kubernetes node name and the `bin/ssh` alias.

**Host cores.** Each GPU tray gets whole physical cores of its own, so the trays never compete with the rest of the lab for a core ([Platform: CPU placement](platform.md#configuration)).

## How the pieces connect

**Power control.** GPU trays live in the Incus project `trays`. Each tray BMC holds an Incus client certificate restricted to that project and drives the tray's power through the Incus API on the bridge address (`:8443`); it cannot reach any other instance.

**Data paths.**

```
 job (srun or pod) ─► CUDA/NCCL calls ─► fakegpu: simulated time + occupancy ─► NVML ─► nvidia-smi, GPU exporter ─► Prometheus
                                                     │
                                                     └─ NVLink traffic ─► fakenmxc /metrics ─► Prometheus
 Redfish PATCH (BMC) / gRPC (fakenmxc) ─► NVML link state, clique ─► topograph ─► Slurm topology.conf or Kubernetes node labels (Kueue)
```

## Storage volumes

| Volume / pool | Mounted at | Purpose |
|---|---|---|
| `cluster-shared` (pool `default`) | `/shared` on cluster nodes | homes (`/shared/home/<user>`), frameworks venv |
| `scratch-<tray>` (pool `local-nvme`, btrfs, 50 GiB quota) | `/scratch` on each tray | node-local scratch, JuiceFS cache |
| `rustfs-data` (pool `default`) | `/var/lib/rustfs` on `sched-storage` | object data |
| `juicefs-meta` (pool `default`) | `/var/lib/redis` on `sched-storage` | JuiceFS metadata (Redis), kept with the object data |

## Limitations

What the lab models and what it does not is in the README's [What is real and what is modelled](../README.md#what-is-real-and-what-is-modelled); each component page lists its own limitations.

## References

- [Incus documentation](https://linuxcontainers.org/incus/docs/main/): instances, projects, storage volumes, restricted certificates
- [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/): the system family the lab models
- Component pages linked from the [README](../README.md#documentation)
