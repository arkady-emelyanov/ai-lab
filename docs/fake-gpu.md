[← README](../README.md) · **Emulated GPUs**

# Emulated GPUs (`fakegpu/`)

## Overview

Each GPU tray presents four NVIDIA GB200 GPUs that do not exist. A set of stub NVIDIA userspace libraries makes them real to the operating system, the scheduler (Slurm or Kubernetes), monitoring and frameworks:

| Library / tool | Replaces | Notes |
|---|---|---|
| `libcuda.so.1` | CUDA driver | devices, contexts, memory, streams, events, launches; the undocumented "export tables" `libcudart` requires (ported from [ZLUDA](https://github.com/vosen/ZLUDA)) |
| `libnvidia-ml.so.1` | NVML | identity, PCIe, NVLink (18 links to NVSwitch), fabric cluster UUID and clique, telemetry, processes, events |
| `libcublas`, `libcublasLt` (`.so.13` and `.so.12`), `libnccl.so.2`, `libcudnn.so.9` | CUDA libraries bundled with frameworks | preloaded through `/etc/ld.so.preload` so they win over the copies in pip wheels |
| `nvidia-smi` | `nvidia-smi` | Python over NVML: table, `-L`, `-q`, `topo -m`, `nvlink`, `dmon`, `--query-gpu`, `--query-compute-apps`, `-l` |
| `numactl` | `numactl` (on the trays) | Python; shows the GB200 NUMA layout (`-H`, `-s`) and pins commands to a Grace node's CPUs (`--cpunodebind`, `--physcpubind`); memory policies are accepted but not applied ([Limitations](#limitations)) |
| `/dev/nvidia0-3`, `/dev/nvidiactl` | device nodes | character devices (major 195) created by Incus |

**The emulation boundary:** an application launched through Slurm or Kubernetes starts, initialises PyTorch, Ray or NCCL, and runs: every call succeeds and takes modelled time, and the GPUs report modelled load. GPU kernels never actually execute, so there is no real GPU math.

An application that does not check its numbers believes everything worked.

## How it works

**Unmodified programs run.** The libraries offer every function NVIDIA's do, so CUDA programs load and run without changes.

What frameworks rely on (devices, memory, streams, GEMMs, NCCL, NVML) behaves like the real thing, except that there is no real GPU math. Rarely used functions only report success, without doing anything ([Limitations](#limitations)).

**Simulated time.** Each CUDA stream has its own timeline:

- GPU work (kernels, copies, cuBLAS and cuDNN calls, NCCL collectives) is added to the timeline and the call returns at once.
- Synchronising calls (`cudaStreamSynchronize`, `cudaDeviceSynchronize`, events, blocking copies) wait until the timeline catches up.
- A stream holds at most 1024 pending operations, like the hardware's queue, so a tight launch loop eventually waits.
- `cudaEventElapsedTime` reports the simulated durations.

The rates are NVIDIA's published GB200 figures per GPU (dense): 2.5 PFLOP/s BF16/FP16, 1.25 PFLOP/s TF32, 5 PFLOP/s FP8, 40 TFLOP/s FP64. Two are the lab's own:

- FP32 without tensor cores: about 60 TFLOP/s, an estimate (NVIDIA publishes no figure).
- Memory-bound work: half the published 8 TB/s, since a copy both reads and writes.

All of them are settings ([Configuration](#configuration)).

| Operation | Cost (GB200-like, × `latency_scale`) |
|---|---|
| Kernel launch | 3 µs + 2.5 µs per wave of blocks over 148 SMs |
| GEMM (cuBLAS, cuBLASLt) | 2·m·n·k FLOPs at the rate of its precision: ~2.5 PFLOP/s BF16/FP16 (and other 16-bit types), ~1.25 PFLOP/s TF32 (`CUBLAS_COMPUTE_32F_FAST_TF32`, or FP32 with `CUBLAS_TF32_TENSOR_OP_MATH`, as PyTorch's `allow_tf32` sets), ~5 PFLOP/s FP8, ~60 TFLOP/s FP32, ~40 TFLOP/s FP64 |
| Copy host↔GPU / GPU↔GPU / device | ~400 GB/s / ~900 GB/s (NVLink) / ~4 TB/s |
| NCCL collective, ranks in one NVLink partition | 10 µs + bytes × bus factor (all-reduce 2(n−1)/n) at ~900 GB/s |
| NCCL collective across partitions | 20 µs + bytes × bus factor(m) at ~900 GB/s (NVLink, inside each partition) + bytes × bus factor(g) / m at 50 GB/s (InfiniBand, one 400 Gb/s NIC per GPU), for g partitions of at least m ranks |
| NCCL send/recv | 10 µs + bytes at ~900 GB/s to a peer in the same partition, 50 GB/s otherwise |
| cuDNN convolution or graph | 40 µs |

**GPU load.** Every CUDA process on a tray records what it uses in a shared file (`state_path`): its memory on each GPU, and each GPU's busy time and NVLink traffic. From that, NVML reports:

- utilisation: the share of time the GPU was busy
- power: from 140 W idle to about 1 kW at full load
- temperature: from 32 °C idle towards about 75 °C, following the load with a 20 s lag
- clocks, P-state, energy, and the processes on each GPU, under the PIDs the caller sees (also inside a pod)

The tray's BMC reads the same file, so its sensors match `nvidia-smi` ([BMCs](bmc-redfish.md#how-it-works)).

**GPU memory** counts against the GPU's 186 GB: an allocation beyond it fails with out-of-memory. Allocations are reserved, not filled, so large GPU buffers cost no host RAM.

**Management inputs.** The BMCs and the partition controller tell the GPUs about changes through shared files (the tray's sideband):

- An NVLink disabled on the tray BMC or the switch BMC shows as inactive (`nvidia-smi topo -m` shows e.g. `NV16`).
- Each GPU reports the NVLink partition (clique) it had at its last reset.

**GPU reset.** `nvidia-smi --gpu-reset` works like the real one: it needs root and refuses a GPU that a process still uses (`-i` picks GPUs). A reset:

- returns the GPU to idle (utilisation, temperature)
- applies a pending NVLink partition change: as on GB200, a GPU joins its new partition only at a reset or a reboot

Until the reset, `nvidia-smi -q` shows `GPU Recovery Action : GPU_RESET`. That is the lab's choice: NVIDIA documents this field for faults, not for partition changes.

Trays reset their GPUs at every boot and, in Slurm mode, when a job ends ([Slurm](slurm.md#how-it-works)).

There is no GPU memory to clear: the lab keeps "GPU memory" in the process's own RAM, which the kernel zeroes for each new allocation. On real hardware, the reset is what clears GPU memory between tenants (GB200 NVL Partition User's Guide, §4.1).

**Visible GPUs.** As with the real driver, a process sees the GPUs whose device files (`/dev/nvidia<n>`) it has:

- on the tray: all four
- in a Kubernetes pod: only the ones allocated to it
- in a Slurm job: all four files, but CUDA uses only the GPUs in `CUDA_VISIBLE_DEVICES`, which Slurm sets to the job's GPUs

`nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`, like the real one, so inside a Slurm job it lists all four GPUs. Real clusters hide the others with device cgroups (`ConstrainDevices=yes`), which the lab's containers cannot use.

A GPU's UUID and serial come from the tray's name, so a pod sees the same GPUs as its tray. The PCI identity is a GB200 compute tray's (public `lspci` output from real trays): device `10de:2941`, subsystem `10de:2046`, each GPU in a PCI domain of its own on bus 1, `0008:01:00.0`, `0009:01:00.0` (on Grace 0), `0018:01:00.0`, `0019:01:00.0` (on Grace 1).

## Usage

On a tray (`bin/ssh sched-worker1`), inside a Slurm job or inside a GPU pod (the CDI device brings `nvidia-smi` along):

```
nvidia-smi                       # table with processes
nvidia-smi topo -m               # NV18 between every GPU pair
nvidia-smi -q                    # includes GPU Recovery Action and Fabric: State, CliqueId, ClusterUUID
nvidia-smi --gpu-reset -i 0      # root, GPU without processes
nvidia-smi dmon                  # per second: power, temperature, utilisation, clocks
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu,fabric.cliqueId --format=csv -l 1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
numactl -H                       # GB200 NUMA nodes: Grace CPUs 0-1, GPUs 2, 10, 18, 26 (tray and Slurm jobs)
numactl -N 1 -m 1 python train.py   # rank of GPU 2 or 3 on its Grace CPU's cores
```

PyTorch, Ray and NCCL code runs unmodified ([Frameworks and examples](frameworks-and-examples.md)).

**NCCL topology.** As in real NCCL, the ranks of a job first find out which NVLink partition each GPU is in:

- Each rank asks NVML for its GPU's partition (cluster UUID and clique).
- The ranks swap that information through files on `/shared` (`fakegpu_nccl_dir`). Split communicators (`ncclCommSplit`) do the same.
- If a rank's peers do not report within 60 s (`nccl_timeout_s`), it warns once and assumes they share its partition. NCCL calls never fail.

Then the collectives are timed:

- Inside one partition, over NVLink.
- Across partitions, as NCCL does across nodes: NVLink inside each partition and InfiniBand between them. A job split across partitions slows down, and its traffic shows on the trays' InfiniBand counters ([Monitoring](monitoring.md)).
- A GPU in no partition (clique 0) has no NVLink peers.

The link rates are published figures (NVLink5 per GPU, NDR 400 per NIC), not measurements. On real hardware, message sizes, NCCL's algorithm choice and overlap with computation make the results vary.

## Configuration

`inventory/group_vars/all.yml` (override them in `local.yml`), applied with `make configure` (written to `/etc/fakegpu.conf` on the trays; every key can be overridden per process with `FAKEGPU_<KEY>`):

| Variable | Default | Read by |
|---|---|---|
| `fakegpu_count`, `fakegpu_type` | 4, `gb200` | GPUs per tray (fixed size of the lab), Slurm GRES type |
| `fakegpu_name` | `NVIDIA GB200` | NVML name: `nvidia-smi`, GFD's `nvidia.com/gpu.product`, the BMCs' GPU processors, chassis and `HGX_GPU_<n>` model |
| `fakegpu_mem_mb` | 189471 | GPU memory: NVML, `nvidia-smi`, GFD's `nvidia.com/gpu.memory`, the BMCs' `MemorySummary` |
| `fakegpu_nvlinks` | 18 | NVLinks per GPU (GB200: 18; leave as is unless you approximate another GPU generation): NVML and `nvidia-smi` (`nvlink -s`, `topo -m` shows `NV<links>`), the tray BMCs' ports, the partition controller and switch cabling ([NVLink partitions](nvlink-partitions.md#configuration)) |
| `fakegpu_nvlink_link_gbs` | 50 | GB/s per NVLink and direction: NVML (`NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON`, `nvlink -s`), the BMCs' port speeds (2 lanes), GPU-to-GPU copies and NCCL (links × rate = 900 GB/s per GPU) |
| `fakegpu_power_limit_w` | 1200 | NVML power limit (`nvidia-smi`, GPU exporter), the BMCs' `PowerLimitWatts` |
| `fakegpu_idle_power_w`, `fakegpu_max_power_w` | 140, 1000 | power draw idle and at full utilisation (NVML, `nvidia-smi`, the BMCs' sensors, which use the same model) |
| `fakegpu_sm_count` | 148 | multiprocessor count (CUDA device attribute), kernel duration (waves of blocks) |
| `fakegpu_tensor_tflops`, `fakegpu_tf32_tflops`, `fakegpu_fp8_tflops`, `fakegpu_fp32_tflops`, `fakegpu_fp64_tflops` | 2500, 1250, 5000, 60, 40 | dense GEMM throughput per precision (cuBLAS, cuBLASLt): BF16/FP16 and other 16-bit types, TF32, FP8 on tensor cores; FP32 without tensor cores; FP64 |
| `fakegpu_host_link_gbs`, `fakegpu_hbm_gbs` | 400, 4000 | host-to-GPU copies (NVLink-C2C) and device-to-device copies |
| `fakegpu_coherent_gpu_memory` | `os` | whether GPU memory is a NUMA node, like the driver option `NVreg_CoherentGPUMemoryMode`. `os` (the driver's default): yes, as on GB200, GPUs on nodes 2, 10, 18, 26 after the Grace CPUs' 0 and 1 (`nvidia-smi topo -m`, `numactl -H`). `driver` (CDMM, which NVIDIA recommends for Kubernetes): no, GPU NUMA ID N/A. Either way each GPU's NUMA affinity is its Grace CPU: 0 for GPUs 0–1, 1 for GPUs 2–3 |
| `fakegpu_driver_version` | `580.95.05` | `nvidia-smi` "Driver Version", NVML (`nvmlSystemGetDriverVersion`, `nvmlSystemGetNVMLVersion`), GFD's `nvidia.com/cuda.driver-version.*` labels |
| `fakegpu_cuda_version` | `13.0` | highest CUDA version the driver supports (`MAJOR.MINOR`): `nvidia-smi` "CUDA Version", `cuDriverGetVersion`, `nvmlSystemGetCudaDriverVersion`, GFD's `nvidia.com/cuda.runtime-version.*` labels |
| `fakegpu_vbios_version` | `97.00.82.00.0F` | GPU VBIOS: `nvidia-smi` "VBIOS Version", NVML, the tray BMC's firmware inventory (`HGX_FW_GPU_<n>`); can be set per tray ([BMCs](bmc-redfish.md#configuration)) |
| `nvl_cluster_uuid`, `nvl_clique_id` | `7f3c2a10-…`, 1 | NVLink domain UUID and the default partition's clique: NVML fabric info, GFD's `nvidia.com/gpu.clique`, topograph's blocks and labels, the BMCs' `FabricClique`, the partition controller |
| `ib_link_rate` | `NDR` | InfiniBand link rate (`ib_link_gbps`: NDR 400 Gb/s, XDR 800, HDR 200, ...): `ibnetdiscover` (`4xNDR`) and `ib_gbps_per_gpu`, NCCL's bandwidth between NVLink partitions |
| `fakegpu_latency_scale` | 1.0 | multiplier for all simulated times; 0 disables delays |
| `fakegpu_copy_max_mb` | 64 | copies above this are timed but not performed |
| `fakegpu_nccl_dir` | `/shared/.fakegpu/nccl` | where NCCL ranks exchange their NVLink partitions (empty: no exchange, one partition assumed) |

**Changing them** needs only `make configure`. It rewrites the configuration of the GPUs, BMCs, partition controller and InfiniBand fabric, and restarts the services that read it.

Running jobs keep the values they started with: processes read `/etc/fakegpu.conf` when they start.

**Validation.** `make up` and `make configure` check the settings first and stop with a message if they don't fit together:

- `fakegpu_nvlinks` is 1–64 and a multiple of `nvswitch_count`, and every GPU link fits a switch port (trays × GPUs × links ≤ `nvswitch_count` × `nvswitch_ports`).
- Idle power is below maximum power, and maximum power is not above the limit.
- Rates and counts are positive; `fakegpu_cuda_version` is `MAJOR.MINOR`.
- `nvl_cluster_uuid` is a lower-case UUID, and `nvl_clique_id` is 1–32765.
- `ib_link_rate` is one of `ib_link_gbps`.

`/etc/fakegpu.conf` also gets `host`, the tray's name, which GPU identities derive from. Debugging: `FAKEGPU_DEBUG=1` logs CUDA entry points resolved to no-ops and unknown export tables.

## Verification

```
bin/ssh sched-worker1 nvidia-smi -L                          # 4 × NVIDIA GB200 with UUIDs
bin/ssh login 'srun -N1 --gpus-per-node=2 bash -c "echo \$CUDA_VISIBLE_DEVICES; nvidia-smi -L -i \$CUDA_VISIBLE_DEVICES"'   # Slurm: the job's 2 GPUs
bin/ssh login 'cd examples/kubernetes && ./submit --wait nvl8-hello.yaml'   # k3s: 8 pods, one GPU each
bin/ssh login sbatch < examples/slurm/gpu-topology.sbatch    # topology, NVLink matrix, fabric per tray
make test-fakegpu                                            # the libraries on this machine, no lab: NCCL across partitions, NIC counters
```

While a GPU job runs, `nvidia-smi` on its tray shows its processes with memory, utilisation, power and rising temperature; the same values appear in Prometheus ([Monitoring](monitoring.md)).

## Limitations

- Kernels do not run: tensors computed on the GPU hold zeros or garbage; NCCL collectives behave as if every rank contributed the same data (so cross-rank consistency checks pass); ranks exchange only their NVLink partitions at initialisation, so a failed link or rank does not fail its peers' collectives.
- Data larger than `fakegpu_copy_max_mb` does not round-trip between host and GPU.
- The GB200 NUMA layout exists only in what the lab's driver and `numactl` report. The tray's kernel has the host's NUMA layout, so:
  - `numactl` splits the tray's own CPUs and memory between the two Grace nodes; it pins CPUs but not memory, and is not in pods (on real clusters it comes with the container image).
  - `nvidia-smi topo -m` shows the same split as "CPU Affinity": the first half of the tray's cores for GPUs 0–1, the second half for GPUs 2–3.
  - Tools that read the kernel's NUMA layout directly (`lscpu`, `hwloc`) see the host's.
  - CUDA's NUMA device attribute (`cudaDeviceProp.deviceNumaId`) is not reported.
- Unusual APIs reach generated no-ops that return success without filling outputs; mainstream PyTorch, Ray, NCCL and NVML paths are implemented.
- Only x86-64; the stubs mimic CUDA 13 (and the CUDA 12 SONAMEs for cuBLAS).

## References

- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/), [NVML API](https://docs.nvidia.com/deploy/nvml-api/), [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/)
- [ZLUDA](https://github.com/vosen/ZLUDA) (Apache-2.0): source of the `libcudart` export-table layouts and integrity check (`fakegpu/dark_api.c`)
- Sources: `fakegpu/cuda_stub.c`, `timing.c`, `nvml_stub.c`, `occupancy.h`, `cublas_stub.c`, `nccl_stub.c`, `cudnn_stub.c`, `nvidia-smi`; role `roles/fakegpu`
