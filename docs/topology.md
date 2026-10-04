[← README](../README.md) · **Topology discovery**

# Topology discovery (InfiniBand fabric and topograph)

## Overview

Slurm's `topology/block` places jobs inside NVLink domains. Instead of a hand-written `topology.conf`, the lab generates it from the live system with [topograph](https://github.com/dsx-ai-factory/topograph) (NVIDIA, Apache-2.0), using its bare-metal provider `infiniband-bm`:

| Source | Command on the trays (through pdsh) | Gives |
|---|---|---|
| InfiniBand compute fabric (emulated) | `sudo ibnetdiscover` | switch tree: spine `LAB-IBSPINE-01` → leaf `LAB-IBLEAF-01` → 4 NDR HCAs (`mlx5_0-3`) per tray |
| NVLink domain | `nvidia-smi --query-gpu=fabric.clusterUuid,fabric.cliqueId` | NVLink partition per node |

Each NVLink partition (cluster UUID + clique) becomes one Slurm block.

## Usage

```
bin/ssh slurm-control update-topology --dry-run    # print what topograph generates
bin/ssh slurm-control update-topology              # install /etc/slurm/topology.conf and reconfigure Slurm
bin/ssh slurm-control scontrol show topology
```

A systemd timer (`update-topology.timer`) runs it every minute; an unchanged topology is left alone, so partition changes made through the [partition controller](nvlink-partitions.md) reach Slurm within a minute. Example after moving tray 2 into its own partition:

```
# block001=7f3c2a10-5b4e-4d6a-9c1e-2b8f0e6d4a91.1
BlockName=block001 Nodes=slurm-worker1
# block002=7f3c2a10-5b4e-4d6a-9c1e-2b8f0e6d4a91.7
BlockName=block002 Nodes=slurm-worker2
```

topograph's API is also available directly on `slurm-control:49021` (`POST /v1/generate`, `GET /v1/topology?uid=…`).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ib_spine`, `ib_leaf`, `ib_hcas_per_node` | `LAB-IBSPINE-01`, `LAB-IBLEAF-01`, 4 | emulated fabric (`/etc/fakeib.json` on the trays) |
| `topograph_ref` | pinned commit | topograph source built by `make init` |
| `topograph_port` | 49021 | API port |

`roles/slurm` writes only the initial `topology.conf`; afterwards topograph owns it. pdsh runs as root on the controller with its own key (`.secrets/ssh/controller_ed25519`), authorised for root on the trays.

## Verification

```
bin/ssh slurm-control 'PDSH_RCMD_TYPE=ssh pdsh -w slurm-worker[1-2] "sudo ibnetdiscover | grep -c mlx5"'   # 16 per node (fabric-wide view)
bin/ssh slurm-control update-topology --dry-run
bin/ssh slurm-control systemctl list-timers update-topology.timer
```

`make test` checks that topograph generates a block containing both trays.

## Limitations

- topograph emits a `BlockSizes` line that Slurm 23.11 does not support; `update-topology` drops it.
- The InfiniBand fabric exists only as `ibnetdiscover` output: there is no RDMA traffic, and the fabric has one leaf and one spine.

## References

- [topograph](https://github.com/dsx-ai-factory/topograph): [Slurm engine](https://github.com/dsx-ai-factory/topograph/blob/main/docs/engines/slurm.md), [InfiniBand provider](https://github.com/dsx-ai-factory/topograph/blob/main/docs/providers/infiniband.md), [API](https://github.com/dsx-ai-factory/topograph/blob/main/docs/api.md)
- [Slurm topology.conf (block)](https://slurm.schedmd.com/topology.conf.html)
- Sources: `fakeib/ibnetdiscover`; roles `roles/fakeib`, `roles/topograph` (`update-topology`, timer)
