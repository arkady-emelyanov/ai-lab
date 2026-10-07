[← README](../README.md) · **Topology discovery**

# Topology discovery (InfiniBand fabric and topograph)

## Overview

Slurm's `topology/block` places jobs inside NVLink domains. Instead of a hand-written `topology.conf`, the lab generates it from the live system with [topograph](https://github.com/dsx-ai-factory/topograph) (NVIDIA, Apache-2.0), using its bare-metal provider `infiniband-bm`:

| Source | Command on the trays (through pdsh) | Gives |
|---|---|---|
| InfiniBand compute fabric (emulated) | `sudo ibnetdiscover` | switch tree: spine `LAB-IBSPINE-01` → leaf `LAB-IBLEAF-01` → 4 NDR HCAs (`mlx5_0-3`) per tray |
| NVLink domain | `nvidia-smi --query-gpu=fabric.clusterUuid,fabric.cliqueId` | NVLink partition per node |

Each NVLink partition (cluster UUID + clique) becomes one Slurm block. topograph assigns each node exactly one NVLink domain (its output is per node: a block is a set of whole nodes, a label sits on the Node), so all GPUs of a tray must be in the same partition; see [Limitations](#limitations).

In k3s mode the same discovery feeds topograph's Kubernetes engine instead: it labels the trays `accelerator.topograph.run/domain=<cluster UUID>.<clique>`, `fabric.topograph.run/tier-0=<leaf>` and `tier-1=<spine>`, which Kueue's topology-aware scheduling uses ([Kubernetes](kubernetes.md#how-it-works)). topograph still runs on the controller; its unit provides the in-cluster credentials its Kubernetes engine expects, and the trays carry the `topograph.run/instance`/`region` annotations it maps nodes with.

## Usage

```
bin/ssh sched-control update-topology --dry-run    # print what topograph generates
bin/ssh sched-control update-topology              # Slurm: install /etc/slurm/topology.conf and reconfigure; k3s: relabel the trays
bin/ssh sched-control scontrol show topology        # Slurm mode
bin/kubectl get nodes -L accelerator.topograph.run/domain,fabric.topograph.run/tier-0   # k3s mode
```

A systemd timer (`update-topology.timer`) runs it every minute; an unchanged topology is left alone, so partition changes made through the [partition controller](nvlink-partitions.md) reach the scheduler within a minute of the GPUs' reset (a GPU reports its new clique only after a reset, as on GB200) (in k3s mode `update-topology --dry-run` prints the trays' labels). Example after moving tray 2 into its own partition:

```
# block001=7f3c2a10-5b4e-4d6a-9c1e-2b8f0e6d4a91.1
BlockName=block001 Nodes=sched-worker1
# block002=7f3c2a10-5b4e-4d6a-9c1e-2b8f0e6d4a91.7
BlockName=block002 Nodes=sched-worker2
```

topograph's API is also available directly on `sched-control:49021` (`POST /v1/generate`, `GET /v1/topology?uid=…`).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ib_spine`, `ib_leaf`, `ib_hcas_per_node` | `LAB-IBSPINE-01`, `LAB-IBLEAF-01`, 4 | emulated fabric (`/etc/fakeib.json` on the trays) |
| `topograph_ref` | pinned commit | topograph source built by `make init` |
| `topograph_port` | 49021 | API port |

`roles/slurm` writes only the initial `topology.conf`; afterwards topograph owns it. pdsh runs as root on the controller with its own key (`.secrets/ssh/controller_ed25519`), authorised for root on the trays.

## Verification

```
bin/ssh sched-control 'PDSH_RCMD_TYPE=ssh pdsh -w sched-worker[1-2] "sudo ibnetdiscover | grep -c mlx5"'   # 16 per node (fabric-wide view)
bin/ssh sched-control update-topology --dry-run
bin/ssh sched-control systemctl list-timers update-topology.timer
bin/ssh sched-control 'systemctl is-failed update-topology.service; journalctl -u update-topology -n 5'   # failed runs and why
```

`make test` checks that topograph generates a block containing both trays (Slurm mode) or labels both trays with the NVLink domain and the switch tiers (k3s mode).

## Limitations

- topograph emits a `BlockSizes` line that Slurm 23.11 does not support; `update-topology` drops it.
- A tray that does not answer (powered off through its BMC, or down) is left out of the generated topology until it answers again, so its block shrinks or disappears within a minute. Slurm keeps scheduling on the tray until `SlurmdTimeout` marks it down.
- **A tray split across NVLink partitions stops topology updates.** topograph deduplicates the per-GPU `ClusterUUID.CliqueId` values of each node and requires exactly one (`ParseNvidiaSMIOutput` in its `pkg/accelerator/nvidia_smi.go`); otherwise the whole run fails with `ambiguous NVL partition IDs` (`update-topology.service` fails). Nothing is regenerated: `topology.conf` (or the k3s labels) keeps its last good state, and the scheduler keeps placing jobs on it, for example two GPUs of one tray that are in different partitions and cannot reach each other over NVLink. The failed unit and its log are the only signal. Partition along tray boundaries; NVIDIA's NMX-C and the hardware allow splitting a tray, the scheduler-side tooling does not.
- The InfiniBand fabric exists only as `ibnetdiscover` output: there is no RDMA traffic, and the fabric has one leaf and one spine.

## References

- [topograph](https://github.com/dsx-ai-factory/topograph): [Slurm engine](https://github.com/dsx-ai-factory/topograph/blob/main/docs/engines/slurm.md), [Kubernetes engine](https://github.com/dsx-ai-factory/topograph/blob/main/docs/engines/k8s.md), [InfiniBand provider](https://github.com/dsx-ai-factory/topograph/blob/main/docs/providers/infiniband.md), [API](https://github.com/dsx-ai-factory/topograph/blob/main/docs/api.md)
- [Slurm topology.conf (block)](https://slurm.schedmd.com/topology.conf.html), [Kueue topology-aware scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/)
- Sources: `fakeib/ibnetdiscover`; roles `roles/fakeib`, `roles/topograph` (`update-topology`, timer)
