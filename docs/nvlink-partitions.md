[← README](../README.md) · **NVLink partitions and fabric telemetry**

# NVLink partitions and fabric telemetry (`fakenmxc/`)

## Overview

On GB200-class systems the NVLink domain is managed from the switch trays: NVIDIA's NMX Controller (NMX-C) owns the partitions (which GPUs may talk over NVLink), and NMX Telemetry exposes the fabric's state and counters. The lab's `sched-nvswitch` host runs `fakenmxc`, which plays both roles:

| Interface | Port | Purpose |
|---|---|---|
| gRPC (`nmxlab.v1.NMXController`, server reflection) | 9370 | partitions, domain/GPU/switch/topology inventory |
| HTTP `/metrics` (Prometheus text) | 9372 | partitions, per-GPU NVLink state and traffic, per-switch port state and throughput |

The API is modelled on NMX-C's public documentation (Hello handshake, default partition 32766, partition ids 1–32765, GPUs addressed by UID or location) but uses its own `.proto` (`fakenmxc/proto/nmxc.proto`); NVIDIA's `.proto` is proprietary and not reused: it ships with the switch tray firmware ("the NMX-Controller gRPC API Guide, which is a part of the Switch Tray firmware package", [GB200 NVL Partition User's Guide](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf) §4), and the copy in NVIDIA's open-source infra-controller is marked [`LicenseRef-NvidiaProprietary`](https://github.com/dsx-ai-factory/infra-controller/blob/755d11644ce46ef3a7244830b6bef8fe3b386110/crates/rpc/proto/nmx_c.proto#L2-L10) (use or redistribution without an NVIDIA licence agreement prohibited). Clients written against NVIDIA's NMX-C therefore cannot talk to the lab's controller.

## How it works

- All GPUs start in the **default partition 32766**; user partitions use ids 1–32765 and a GPU belongs to at most one partition.
- Partition membership is published to each tray's sideband (`fabric-clique`) and becomes the **clique** the GPU reports through NVML (`nvidia-smi --query-gpu=fabric.cliqueId`): the configured default clique for the default partition, the partition id for user partitions, 0 for GPUs in no partition.
- **A GPU takes a new partition at a GPU reset**, as on GB200 ("We need to reset the GPUs (or reboot the nodes) in order for the Clique ID to update", Mission Control). Until then it reports its previous clique, `nvidia-smi -q` shows `GPU Recovery Action : GPU_RESET`, and the controller reports `reset_pending` (`bin/nvlink gpus`, metric `nvlink_gpu_reset_pending`). A reset (`nvidia-smi --gpu-reset`, root, only GPUs no process uses), a tray reboot, or the Slurm epilog after a job applies it ([Fake GPUs](fake-gpu.md#how-it-works), [Slurm](slurm.md#how-it-works)). topograph turns cliques into Slurm blocks or, in k3s mode, node labels for Kueue ([Topology](topology.md)); there GPU Feature Discovery also publishes each tray's clique as `nvidia.com/gpu.clique`.
- **Health** comes from the sideband link state written by the BMCs. As in NVIDIA's GB200 NVL Partition User's Guide (§6.2), a GPU with an access link down (GPU to NVSwitch, disabled on either BMC) is `NMX_GPU_HEALTH_NO_NVLINK`, and its partition's health is unchanged. The guide's other GPU states (`DEGRADED`, `DEGRADED_BW`) and partition states (`DEGRADED_BANDWIDTH`, `DEGRADED`, `UNHEALTHY`) come from trunk-link failures between switch trays, which a single switch tray does not have; they are in the API but never reported.
- **Telemetry** reads each tray's occupancy file through a read-only telemetry volume: busy time and NVLink bytes per GPU, counted by the fake CUDA stack (NCCL collectives and GPU-to-GPU copies). Per-switch throughput attributes each GPU's traffic to its active links on that switch.
- Partition state is persisted in `/var/lib/fakenmxc/partitions.json`.

## Usage

`bin/nvlink` from your machine wraps the controller's gRPC API: it says `Hello`, names GPUs by tray and index instead of `Location` messages, and prints tables (`--json` for the raw responses):

```
bin/nvlink domain                                 # UUID, sizes, control plane state
bin/nvlink gpus sched-worker2                     # uuid, partition, clique, pending reset, NVLinks, health
bin/nvlink topology                               # active links per GPU and NVSwitch
bin/nvlink partitions                             # partitions and their GPUs, GPUs in none
bin/nvlink remove default sched-worker2           # a GPU belongs to one partition: out of the default first
bin/nvlink create tray2 --id 7 sched-worker2      # GPUs as tray (all), tray:0-1 or slot:index; prints the reset to run
bin/ssh sched-worker2 nvidia-smi --query-gpu=fabric.cliqueId --format=csv     # still 1: no reset yet
bin/ssh sched-worker2 nvidia-smi --gpu-reset      # the GPUs take partition 7 (they must be idle)
bin/ssh sched-worker2 nvidia-smi --query-gpu=fabric.cliqueId --format=csv     # now 7
bin/nvlink delete tray2                           # its GPUs end up in no partition
bin/nvlink add default sched-worker2              # back to the default partition
bin/ssh sched-worker2 nvidia-smi --gpu-reset      # and reset again to take it
```

Partitions can take single GPUs (`sched-worker2:1`), but keep each tray's GPUs in one partition: topograph needs one partition per node and stops updating the scheduler's topology when a tray is split ([Topology](topology.md#limitations)); `bin/nvlink` warns when a change leaves a tray split.

The underlying RPCs, for clients of your own (`bin/grpcurl`, [grpcurl](https://github.com/fullstorydev/grpcurl) built with Go into `.cache/tools` on first use, calls them directly):

| RPC | Purpose |
|---|---|
| `Hello` | identifies the client (gateway); other calls return `NMX_ST_NOT_HELLO` before it |
| `GetDomainProperties`, `GetDomainStateInfo` | domain UUID, sizes, control plane state |
| `GetComputeNodeInfoList`, `GetSwitchNodeInfoList` | trays (slot, GPUs), switch tray (switches, ports, ports down) |
| `GetGpuInfoList` | GPUs, optionally filtered by `slot_ids` and/or `partition_id` |
| `GetTopologyInfo` | every GPU link with its switch, port and state |
| `GetPartitionCount`, `GetPartitionIdList`, `GetPartitionInfoList` | partitions |
| `CreatePartition`, `DeletePartition`, `AddGpusToPartition`, `RemoveGpusFromPartition` | partition management; errors `NMX_ST_GPU_IN_USE`, `NMX_ST_NOT_FOUND`, `NMX_ST_ALREADY_EXISTS`, `NMX_ST_INVALID_ARGUMENT` |

`bin/grpcurl -plaintext 10.107.111.34:9370 describe nmxlab.v1.NMXController` lists every RPC and message; a raw call needs `Hello` first with the same `gateway_id`, e.g. `bin/grpcurl -plaintext -d '{"gateway_id": "me"}' 10.107.111.34:9370 nmxlab.v1.NMXController/Hello`.

**Fabric metrics** (`curl http://10.107.111.34:9372/metrics`, scraped by Prometheus as job `nvlink`):

| Metric | Labels | Meaning |
|---|---|---|
| `nvlink_domain_info` | `domain`, `cluster_uuid` | the domain |
| `nvlink_partition_gpus` | `partition_id`, `name`, `default` | GPUs per partition |
| `nvlink_gpu_partition_id`, `nvlink_gpu_clique_id`, `nvlink_gpu_reset_pending` | `host`, `gpu`, `uuid`, `slot` | partition and reported clique per GPU; 1 while a partition change waits for a GPU reset |
| `nvlink_gpu_links`, `nvlink_gpu_active_links`, `nvlink_gpu_healthy` | same | NVLink state per GPU |
| `nvlink_gpu_tx_bytes_total`, `nvlink_gpu_rx_bytes_total`, `nvlink_gpu_busy_seconds_total` | same | traffic and busy time (counters) |
| `nvswitch_port_up` | `switch`, `port`, `host`, `gpu`, `link` | every cabled switch port, labelled with its GPU end |
| `nvswitch_ports`, `nvswitch_ports_up` | `switch`, `host` | ports per switch, ports up |
| `nvswitch_tx_bytes_total`, `nvswitch_rx_bytes_total` | `switch`, `host` | traffic carried by each switch: each scrape adds the GPUs' new bytes, split over the links up at that moment, so a link going down or up changes how new traffic is split and never decreases the counter (they start at 0 when the controller starts) |

## Verification

```
bin/nvlink domain                                 # domain UUID, CONFIGURED
curl -s http://10.107.111.34:9372/metrics | grep -E '^nvswitch_ports_up|^nvlink_partition_gpus'
bin/ssh sched-worker1 nvidia-smi --query-gpu=fabric.clusterUuid,fabric.cliqueId --format=csv
```

`make test` checks the gRPC port, that every GPU reports the domain UUID and a clique, and that Prometheus has the fabric series. During the DDP example, `rate(nvlink_gpu_tx_bytes_total[1m])` rises on the GPUs in use.

## Limitations

- Not wire-compatible with NVIDIA NMX-C clients (different `.proto`); semantics follow the public documentation. Of the return codes, the guide names only `NMX_ST_SUCCESS`, `NMX_ST_NOT_READY` and `NMX_ST_NOT_CONFIGURED`; the others (`NMX_ST_GPU_IN_USE` for a GPU already in a partition, `NMX_ST_NOT_FOUND`, ...) are the lab's own.
- No authentication or TLS on the gRPC and metrics ports.
- A partition change does not stop running jobs: the GPUs of a running job keep their clique until the job ends and the epilog resets them, or until someone resets them; the lab does not tear down NVLink traffic of a GPU leaving a partition.

## References

- [NMX-C gRPC API](https://docs.nvidia.com/networking/display/nmxcv100/grpc+api), [NMX-C v1.3.0](https://docs.nvidia.com/networking/display/nmx-controller-nmx-c-documentation-v1-3-0.0.pdf)
- [GB200 NVL Partition User Guide](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf), [Mission Control: NVLink partition management](https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.3.0/nvlink-partition-management.html)
- Sources: `fakenmxc/proto/nmxc.proto`, `controller.go`, `telemetry.go`; role `roles/fakenmxc`
