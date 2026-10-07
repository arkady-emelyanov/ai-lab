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
| `numactl` | `numactl` (on the trays) | Python, numactl 2.0.18's options and output on the GB200 NUMA layout: `-H` and `-s` show it, `--cpunodebind` and `--physcpubind` pin the command to the tray CPUs of those Grace nodes, memory policies (`--membind`, `--preferred`, `--interleave`, ...) are checked and shown by a nested `numactl -s` but not applied; shared memory policies (`--shm`, `--file`) and device node specifiers (`netdev:`, `pci:`, ...) fail with a message |
| `/dev/nvidia0-3`, `/dev/nvidiactl` | device nodes | character devices (major 195) created by Incus |

**The emulation boundary:** an application launched through Slurm or as a Kubernetes pod starts, initialises its framework (PyTorch, Ray, NCCL, ...), every call succeeds and takes modelled time, and the GPUs report modelled load; GPU kernels never actually execute, so there is no real GPU math. An application that does not check numerical results believes everything worked.

## How it works

**Every symbol exists.** `symbols/*.syms` list every function the real libraries export (taken from real NVIDIA binaries). Each gets a weak "return success" definition placed in its own ELF section; hand-written implementations override them for calls whose outputs matter. `cuGetProcAddress` prefers hand-written implementations over generated no-ops.

**Simulated time.** Every CUDA stream has a timeline. Asynchronous work (kernels, copies, cuBLAS/cuDNN calls, NCCL collectives) is appended and returns at once; `cudaStreamSynchronize`, `cudaDeviceSynchronize`, events and blocking copies wait until the timeline catches up. A stream holds at most 1024 pending operations, like the hardware's push buffer, so tight launch loops block the CPU. `cudaEventElapsedTime` reports simulated durations.

Rates follow NVIDIA's published GB200 figures per GPU (dense): 2.5 PFLOP/s BF16/FP16, 1.25 PFLOP/s TF32, 5 PFLOP/s FP8 and 40 TFLOP/s FP64 on tensor cores; NVIDIA publishes no FP32 figure without tensor cores, so ~60 TFLOP/s is an estimate. Memory-bound work uses half the published 8 TB/s HBM bandwidth, since a copy reads and writes. All are configurable ([Configuration](#configuration)).

| Operation | Cost (GB200-like, × `latency_scale`) |
|---|---|
| Kernel launch | 3 µs + 2.5 µs per wave of blocks over 148 SMs |
| GEMM (cuBLAS, cuBLASLt) | 2·m·n·k FLOPs at the rate of its precision: ~2.5 PFLOP/s BF16/FP16 (and other 16-bit types), ~1.25 PFLOP/s TF32 (`CUBLAS_COMPUTE_32F_FAST_TF32`, or FP32 with `CUBLAS_TF32_TENSOR_OP_MATH`, as PyTorch's `allow_tf32` sets), ~5 PFLOP/s FP8, ~60 TFLOP/s FP32, ~40 TFLOP/s FP64 |
| Copy host↔GPU / GPU↔GPU / device | ~400 GB/s / ~900 GB/s (NVLink) / ~4 TB/s |
| NCCL collective, ranks in one NVLink partition | 10 µs + bytes × bus factor (all-reduce 2(n−1)/n) at ~900 GB/s |
| NCCL collective across partitions | 20 µs + bytes × bus factor(m) at ~900 GB/s (NVLink, inside each partition) + bytes × bus factor(g) / m at 50 GB/s (InfiniBand, one 400 Gb/s NIC per GPU), for g partitions of at least m ranks |
| NCCL send/recv | 10 µs + bytes at ~900 GB/s to a peer in the same partition, 50 GB/s otherwise |
| cuDNN convolution or graph | 40 µs |

**Occupancy.** Every CUDA process registers in a shared table (`state_path`, on the tray's telemetry volume): its PID and memory per GPU, each GPU's busy time and NVLink bytes. Processes in containers (Kubernetes pods) use the same table: each slot records its owner's PID namespace and is held by a file lock the kernel drops when the process exits, so liveness is judged correctly from any namespace, and NVML lists each process under the PID the caller can see. NVML derives utilisation from busy time, power from utilisation (140 W idle to ~1 kW), energy, temperature (32 °C idle towards ~75 °C with a 20 s thermal lag), clocks and P-state, and lists processes. The tray's BMC reads the same table and applies the same model, so its sensors agree with NVML ([BMCs](bmc-redfish.md#how-it-works)). GPU memory is accounted against the 186 GB capacity (allocations beyond it fail with out-of-memory) but is lazily backed, so large GPU buffers cost no host RAM.

**Management inputs.** NVML reads the tray's sideband (written by the BMCs and the partition controller): NVLinks disabled by the tray BMC or switch BMC report inactive (`nvidia-smi topo -m` shows e.g. `NV16`), and each GPU reports the clique of its NVLink partition as of its last reset.

**GPU reset.** `nvidia-smi --gpu-reset` (`-r`, optionally `-i`) needs root and refuses a GPU any process still uses, like the real one. A reset returns the GPU to idle (utilisation, temperature) and applies a pending NVLink partition change: as on GB200, a GPU takes its new partition's clique only at a reset or a node reboot, and until then the lab shows `GPU Recovery Action : GPU_RESET` in `nvidia-smi -q` (NVML field `NVML_FI_DEV_GET_GPU_RECOVERY_ACTION`; the lab's modelling, NVIDIA documents the field for faults). Each tray resets its GPUs at boot (`fakegpu-boot-reset.service`), and in Slurm mode the epilog resets a job's GPUs when it ends ([Slurm](slurm.md#how-it-works)). There is no memory to scrub: "GPU memory" is the allocating process's own memory, which the kernel zeroes for every new allocation and reclaims at exit; on real hardware the reset is what clears GPU memory between tenants (GB200 NVL Partition User's Guide, §4.1).

**Visible GPUs.** As with the real driver, a process sees only the GPUs whose device node `/dev/nvidia<n>` exists in its mount namespace: all of them on the tray, only the allocated ones in a Kubernetes pod (CDI injects one node per GPU). On top of that, CUDA honours `CUDA_VISIBLE_DEVICES` (indices and UUIDs), so a Slurm job's CUDA programs see exactly the GPUs Slurm gave it. NVML, and therefore `nvidia-smi`, ignores `CUDA_VISIBLE_DEVICES`, as the real one does: inside a Slurm job it lists all four GPUs of the tray. Real clusters hide the others from `nvidia-smi` with device cgroups (`ConstrainDevices=yes`), which unprivileged containers cannot use; in a Kubernetes pod only the allocated device nodes exist, so `nvidia-smi` lists only those. GPU identities (UUID, serial, PCI address) derive from the tray name in `/etc/fakegpu.conf` (`host`), so a container, whose hostname differs, sees the tray's GPUs.

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

**NCCL topology.** Like real NCCL at initialisation, every rank learns where its peers' GPUs are: each finds its own GPU's NVLink partition through NVML (PCI bus id, then fabric info: cluster UUID and clique, as of the GPU's last reset), and the ranks of a communicator exchange them through files in `fakegpu_nccl_dir` on `/shared`, keyed by the communicator's unique id (`ncclCommSplit` exchanges colours the same way, so split communicators know their members). Collectives inside one partition run over NVLink; across partitions they are hierarchical, as NCCL does across nodes: NVLink inside each partition and InfiniBand between them, so a job split across partitions slows down and its traffic shows on the trays' NIC counters ([Monitoring](monitoring.md)). A GPU in no partition (clique 0) has no NVLink peer. If a peer does not report within `nccl_timeout_s` (60 s), the rank warns once on stderr and assumes one partition; NCCL calls never fail. The link rates are documented figures (NVLink5 per GPU, NDR 400 per NIC), not measurements; message sizes, NCCL's algorithm choice and overlap with computation make real ratios vary.

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
| `fakegpu_coherent_gpu_memory` | `os` | like the driver option `NVreg_CoherentGPUMemoryMode`: `os` (the driver's default) has each GPU's coherent memory as a NUMA node, as on GB200: GPUs on nodes 2, 10, 18, 26 after the Grace CPUs' nodes 0 and 1 (`nvidia-smi topo -m` "GPU NUMA ID", `topo -gnid`, `nvmlDeviceGetNumaNodeId`, `numactl -H`); `driver` is CDMM (driver-managed GPU memory, which NVIDIA recommends for Kubernetes): GPU NUMA ID N/A, `nvmlDeviceGetNumaNodeId` not supported. Either way "NUMA Affinity" (`nvmlDeviceGetMemoryAffinity`) is the GPU's Grace CPU: 0 for GPUs 0–1, 1 for GPUs 2–3 |
| `fakegpu_driver_version` | `580.95.05` | `nvidia-smi` "Driver Version", NVML (`nvmlSystemGetDriverVersion`, `nvmlSystemGetNVMLVersion`), GFD's `nvidia.com/cuda.driver-version.*` labels |
| `fakegpu_cuda_version` | `13.0` | highest CUDA version the driver supports (`MAJOR.MINOR`): `nvidia-smi` "CUDA Version", `cuDriverGetVersion`, `nvmlSystemGetCudaDriverVersion`, GFD's `nvidia.com/cuda.runtime-version.*` labels |
| `fakegpu_vbios_version` | `97.00.82.00.0F` | GPU VBIOS: `nvidia-smi` "VBIOS Version", NVML, the tray BMC's firmware inventory (`HGX_FW_GPU_<n>`); can be set per tray ([BMCs](bmc-redfish.md#configuration)) |
| `nvl_cluster_uuid`, `nvl_clique_id` | `7f3c2a10-…`, 1 | NVLink domain UUID and the default partition's clique: NVML fabric info, GFD's `nvidia.com/gpu.clique`, topograph's blocks and labels, the BMCs' `FabricClique`, the partition controller |
| `ib_link_rate` | `NDR` | InfiniBand link rate (`ib_link_gbps`: NDR 400 Gb/s, XDR 800, HDR 200, ...): `ibnetdiscover` (`4xNDR`) and `ib_gbps_per_gpu`, NCCL's bandwidth between NVLink partitions |
| `fakegpu_latency_scale` | 1.0 | multiplier for all simulated times; 0 disables delays |
| `fakegpu_copy_max_mb` | 64 | copies above this are timed but not performed |
| `fakegpu_nccl_dir` | `/shared/.fakegpu/nccl` | where NCCL ranks exchange their NVLink partitions (empty: no exchange, one partition assumed) |

**Changing them** needs only `make configure`: it rewrites `/etc/fakegpu.conf`, the BMCs' and the partition controller's configuration and `/etc/fakeib.json`, restarts the BMCs, the partition controller, the GPU exporter and, in k3s mode, GPU Feature Discovery. Processes read `/etc/fakegpu.conf` when they start, so running jobs keep the values they started with. The cuBLAS and cuDNN stubs keep reporting the CUDA 13 runtime they mimic.

**Validation.** `make up` and `make configure` first check the settings (`playbooks/tasks/check-settings.yml`) and stop with a message if they are inconsistent: `fakegpu_nvlinks` must be 1–64 and a multiple of `nvswitch_count`, and trays × GPUs × links must fit `nvswitch_count` × `nvswitch_ports` (every GPU link is cabled to a switch port); idle power below maximum power, maximum power not above the limit; rates and counts positive; `fakegpu_cuda_version` `MAJOR.MINOR`; `nvl_cluster_uuid` a lower-case UUID, `nvl_clique_id` 1–32765; `ib_link_rate` one of `ib_link_gbps`. The lab's size (trays, GPUs per tray, switch trays, InfiniBand switches) is fixed.

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
- The GB200 NUMA layout exists only in what the fake driver and `numactl` report: the tray's kernel has the host's NUMA layout, so `numactl` splits the tray's own CPUs and memory between the two Grace nodes, binds CPUs but not memory, and is not in pods (on real clusters it comes with the container image). Tools reading `/sys/devices/system/node` directly (`lscpu`, `hwloc`) see the host. `nvidia-smi topo -m`'s "CPU Affinity" (`nvmlDeviceGetCpuAffinity`) lists the cores of the GPU's Grace node in that split: the first half of the tray's own cores for GPUs 0–1, the second half for GPUs 2–3. CUDA's device attributes for NUMA (`cudaDeviceProp.deviceNumaId`) are not reported.
- Unusual APIs reach generated no-ops that return success without filling outputs; mainstream PyTorch, Ray, NCCL and NVML paths are implemented.
- Only x86-64; the stubs mimic CUDA 13 (and the CUDA 12 SONAMEs for cuBLAS).

## References

- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/), [NVML API](https://docs.nvidia.com/deploy/nvml-api/), [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/)
- [ZLUDA](https://github.com/vosen/ZLUDA) (Apache-2.0): source of the `libcudart` export-table layouts and integrity check (`fakegpu/dark_api.c`)
- Sources: `fakegpu/cuda_stub.c`, `timing.c`, `nvml_stub.c`, `occupancy.h`, `cublas_stub.c`, `nccl_stub.c`, `cudnn_stub.c`, `nvidia-smi`; role `roles/fakegpu`
