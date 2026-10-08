[← README](../README.md) · **What is real and what is modelled**

# What is real and what is modelled

AI lab models what the hardware shows to software (APIs, topology, telemetry, timing, failures), not the silicon. Everything above that boundary is real software, unmodified.

| Component | Real or modelled | Faithful | Not modelled |
|---|---|---|---|
| Slurm, k3s, Kueue, JobSet, topograph, GPU Feature Discovery | real | configuration, scheduling, accounting, topology-aware placement | Slurm jobs are not confined by cgroups (containers share the host kernel); Slurm is 23.11, without `--segment` and `BlockSizes`; k3s gets its GPUs from the lab's own device plugin, not NVIDIA's, and users may mount host paths in their pods |
| PyTorch, Ray, your applications | real, unmodified | initialisation, process groups, control flow, the errors the APIs below return | numerical results |
| CUDA driver and runtime, cuBLAS, cuDNN | modelled ([`fakegpu`](fake-gpu.md)) | device properties, contexts, streams, events, allocations up to the GPU's memory (out-of-memory beyond it), timing | kernels do not run; copies over 64 MiB are timed, not performed |
| NVML, `nvidia-smi` | modelled ([`fakegpu`](fake-gpu.md)) | identity, PCIe, NVLink state, fabric clique, NUMA layout, processes, GPU reset; utilisation, power and temperature from the load | power and thermals follow a model of load, not measured curves |
| NCCL | modelled ([`fakegpu`](fake-gpu.md)) | communicators and splits, collectives timed over NVLink inside a partition and InfiniBand across partitions | data is not exchanged; a failed peer or link does not fail collectives |
| NVLink and NVSwitch | modelled ([partition controller](nvlink-partitions.md), BMCs) | 18 links per GPU to two 72-port switches, partitions and cliques applied at GPU reset, disabled links, fabric telemetry | NMX-C's wire protocol; one NVL8 domain only |
| BMCs | modelled ([Redfish](bmc-redfish.md)) | GB200 resource layout, power actions that stop and start the tray, GPU sensors, firmware inventory | IPMI, BlueField DPUs |
| InfiniBand | modelled ([topology](topology.md)) | `ibnetdiscover` topology for topograph, NIC byte counters, 400 Gb/s in NCCL timing | packets, subnet manager; one leaf switch |
| Grace CPUs | modelled (NUMA layout only) | two Grace NUMA nodes per tray in `nvidia-smi topo -m` and `numactl -H` | the trays run on the host's x86-64 cores, not Grace (aarch64): `uname -m` and `scontrol show node` say `x86_64` |
| Prometheus, Grafana, LDAP, JuiceFS, RustFS (S3) | real | everything | the lab runs LDAP without TLS |

**Timing is a behavioural model.** Each operation's duration comes from its size and NVIDIA's published GB200 figures (compute per precision, memory, NVLink and InfiniBand bandwidth). Jobs take plausible time and put plausible load on the GPUs. The model is not calibrated and does not predict real GB200 performance. Details: [Emulated GPUs](fake-gpu.md#how-it-works).

Each component page lists its own limitations ([Documentation](README.md)).
