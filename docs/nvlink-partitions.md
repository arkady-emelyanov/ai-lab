[← README](../README.md) · **NVLink partitions and fabric telemetry**

# NVLink partitions and fabric telemetry (`fakenmxc/`)

## Overview

On GB200-class systems the NVLink domain is managed from the switch trays: NVIDIA's NMX Controller (NMX-C) owns the partitions (which GPUs may talk over NVLink), and NMX Telemetry exposes the fabric's state and counters. The lab's `slurm-nvswitch` host runs `fakenmxc`, which plays both roles:

| Interface | Port | Purpose |
|---|---|---|
| gRPC (`nmxlab.v1.NMXController`, server reflection) | 9370 | partitions, domain/GPU/switch/topology inventory |
| HTTP `/metrics` (Prometheus text) | 9372 | partitions, per-GPU NVLink state and traffic, per-switch port state and throughput |

The API is modelled on NMX-C's public documentation (Hello handshake, default partition 32766, partition ids 1–32765, GPUs addressed by UID or location) but uses its own `.proto` (`fakenmxc/proto/nmxc.proto`); NVIDIA's `.proto` is proprietary and not reused.

## How it works

- All GPUs start in the **default partition 32766**; user partitions use ids 1–32765 and a GPU belongs to at most one partition.
- Partition membership is published to each tray's sideband (`fabric-clique`) and becomes the **clique** the GPU reports through NVML (`nvidia-smi --query-gpu=fabric.cliqueId`): the configured default clique for the default partition, the partition id for user partitions, 0 for GPUs in no partition. topograph turns cliques into Slurm blocks ([Topology](topology.md)).
- **Health** comes from the sideband link state written by the BMCs: a GPU with any NVLink down is `DEGRADED_BANDWIDTH`.
- **Telemetry** reads each tray's occupancy file through a read-only telemetry volume: busy time and NVLink bytes per GPU, counted by the fake CUDA stack (NCCL collectives and GPU-to-GPU copies). Per-switch throughput attributes each GPU's traffic to its active links on that switch.
- Partition state is persisted in `/var/lib/fakenmxc/partitions.json`.

## Usage

With [grpcurl](https://github.com/fullstorydev/grpcurl) (`.cache/tools/bin/grpcurl` after the build tools are fetched, or any installation) from your machine:

```
nmx() { grpcurl -plaintext -d "$2" 10.107.111.34:9370 nmxlab.v1.NMXController/$1; }
nmx Hello '{"gateway_id": "me"}'                                    # required first, per gateway id
nmx GetDomainProperties '{"gateway_id": "me"}'
nmx GetGpuInfoList '{"gateway_id": "me", "slot_ids": [2]}'           # tray 2: uuid, location, partition, clique, links, health
gpus='[{"slot_id":2,"gpu_id":0},{"slot_id":2,"gpu_id":1},{"slot_id":2,"gpu_id":2},{"slot_id":2,"gpu_id":3}]'
nmx RemoveGpusFromPartition "{\"gateway_id\": \"me\", \"partition_id\": 32766, \"locations\": $gpus}"
nmx CreatePartition "{\"gateway_id\": \"me\", \"partition_name\": \"tray2\", \"partition_id\": 7, \"locations\": $gpus}"
bin/ssh slurm-worker2 nvidia-smi --query-gpu=fabric.cliqueId --format=csv     # now 7
```

| RPC | Purpose |
|---|---|
| `Hello` | identifies the client (gateway); other calls return `NMX_ST_NOT_HELLO` before it |
| `GetDomainProperties`, `GetDomainStateInfo` | domain UUID, sizes, control plane state |
| `GetComputeNodeInfoList`, `GetSwitchNodeInfoList` | trays (slot, GPUs), switch tray (switches, ports, ports down) |
| `GetGpuInfoList` | GPUs, optionally filtered by `slot_ids` and/or `partition_id` |
| `GetTopologyInfo` | every GPU link with its switch, port and state |
| `GetPartitionCount`, `GetPartitionIdList`, `GetPartitionInfoList` | partitions |
| `CreatePartition`, `DeletePartition`, `AddGpusToPartition`, `RemoveGpusFromPartition` | partition management; errors `NMX_ST_GPU_IN_USE`, `NMX_ST_NOT_FOUND`, `NMX_ST_ALREADY_EXISTS`, `NMX_ST_INVALID_ARGUMENT` |

`grpcurl -plaintext 10.107.111.34:9370 describe nmxlab.v1.NMXController` lists every RPC and message.

**Fabric metrics** (`curl http://10.107.111.34:9372/metrics`, scraped by Prometheus as job `nvlink`):

| Metric | Labels | Meaning |
|---|---|---|
| `nvlink_domain_info` | `domain`, `cluster_uuid` | the domain |
| `nvlink_partition_gpus` | `partition_id`, `name`, `default` | GPUs per partition |
| `nvlink_gpu_partition_id`, `nvlink_gpu_clique_id` | `host`, `gpu`, `uuid`, `slot` | partition and clique per GPU |
| `nvlink_gpu_links`, `nvlink_gpu_active_links`, `nvlink_gpu_healthy` | same | NVLink state per GPU |
| `nvlink_gpu_tx_bytes_total`, `nvlink_gpu_rx_bytes_total`, `nvlink_gpu_busy_seconds_total` | same | traffic and busy time (counters) |
| `nvswitch_port_up` | `switch`, `port`, `host`, `gpu`, `link` | every cabled switch port, labelled with its GPU end |
| `nvswitch_ports`, `nvswitch_ports_up` | `switch`, `host` | ports per switch, ports up |
| `nvswitch_tx_bytes_total`, `nvswitch_rx_bytes_total` | `switch`, `host` | traffic carried by each switch |

## Verification

```
grpcurl -plaintext -d '{"gateway_id":"me"}' 10.107.111.34:9370 nmxlab.v1.NMXController/Hello     # domainUuid
curl -s http://10.107.111.34:9372/metrics | grep -E '^nvswitch_ports_up|^nvlink_partition_gpus'
bin/ssh slurm-worker1 nvidia-smi --query-gpu=fabric.clusterUuid,fabric.cliqueId --format=csv
```

`make test` checks the gRPC port, that every GPU reports the domain UUID and a clique, and that Prometheus has the fabric series. During the DDP example, `rate(nvlink_gpu_tx_bytes_total[1m])` rises on the GPUs in use.

## Limitations

- Not wire-compatible with NVIDIA NMX-C clients (different `.proto`); semantics follow the public documentation.
- No authentication or TLS on the gRPC and metrics ports.
- Partition changes take effect immediately; GPUs are not reset and running jobs are not interrupted.

## References

- [NMX-C gRPC API](https://docs.nvidia.com/networking/display/nmxcv100/grpc+api), [NMX-C v1.3.0](https://docs.nvidia.com/networking/display/nmx-controller-nmx-c-documentation-v1-3-0.0.pdf)
- [GB200 NVL Partition User Guide](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf), [Mission Control: NVLink partition management](https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.3.0/nvlink-partition-management.html)
- Sources: `fakenmxc/proto/nmxc.proto`, `controller.go`, `telemetry.go`; role `roles/fakenmxc`
