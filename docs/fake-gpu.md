[← README](../README.md) · **Fake GPUs**

# Fake GPUs (`fakegpu/`)

## Overview

Each GPU tray presents four NVIDIA GB200 GPUs that do not exist. A set of stub NVIDIA userspace libraries makes them real to the operating system, the scheduler (Slurm or Kubernetes), monitoring and frameworks:

| Library / tool | Replaces | Notes |
|---|---|---|
| `libcuda.so.1` | CUDA driver | devices, contexts, memory, streams, events, launches; the undocumented "export tables" `libcudart` requires (ported from [ZLUDA](https://github.com/vosen/ZLUDA)) |
| `libnvidia-ml.so.1` | NVML | identity, PCIe, NVLink (18 links to NVSwitch), fabric cluster UUID and clique, telemetry, processes, events |
| `libcublas`, `libcublasLt` (`.so.13` and `.so.12`), `libnccl.so.2`, `libcudnn.so.9` | CUDA libraries bundled with frameworks | preloaded through `/etc/ld.so.preload` so they win over the copies in pip wheels |
| `nvidia-smi` | `nvidia-smi` | Python over NVML: table, `-L`, `-q`, `topo -m`, `nvlink`, `dmon`, `--query-gpu`, `--query-compute-apps`, `-l` |
| `/dev/nvidia0-3`, `/dev/nvidiactl` | device nodes | character devices (major 195) created by Incus |

**The emulation boundary:** an application launched through Slurm or as a Kubernetes pod starts, initialises its framework (PyTorch, Ray, NCCL, ...), every call succeeds and takes realistic time, and the GPUs report realistic load; nothing is computed. An application that does not check numerical results believes everything worked.

## How it works

**Every symbol exists.** `symbols/*.syms` list every function the real libraries export (taken from real NVIDIA binaries). Each gets a weak "return success" definition placed in its own ELF section; hand-written implementations override them for calls whose outputs matter. `cuGetProcAddress` prefers hand-written implementations over generated no-ops.

**Simulated time.** Every CUDA stream has a timeline. Asynchronous work (kernels, copies, cuBLAS/cuDNN calls, NCCL collectives) is appended and returns at once; `cudaStreamSynchronize`, `cudaDeviceSynchronize`, events and blocking copies wait until the timeline catches up. A stream holds at most 1024 pending operations, like the hardware's push buffer, so tight launch loops block the CPU. `cudaEventElapsedTime` reports simulated durations.

| Operation | Cost (GB200-like, × `latency_scale`) |
|---|---|
| Kernel launch | 3 µs + 2.5 µs per wave of blocks over 148 SMs |
| GEMM (cuBLAS, cuBLASLt) | 2·m·n·k FLOPs at ~1.2 PFLOP/s (tensor), ~60 TFLOP/s (fp32), ~40 TFLOP/s (fp64) |
| Copy host↔GPU / GPU↔GPU / device | ~400 GB/s / ~900 GB/s (NVLink) / ~4 TB/s |
| NCCL collective | 10 µs + bytes × bus factor (all-reduce 2(n−1)/n) at ~900 GB/s |
| cuDNN convolution or graph | 40 µs |

**Occupancy.** Every CUDA process registers in a shared table (`state_path`, on the tray's telemetry volume): its PID and memory per GPU, each GPU's busy time and NVLink bytes. Processes in containers (Kubernetes pods) use the same table: each slot records its owner's PID namespace and is held by a file lock the kernel drops when the process exits, so liveness is judged correctly from any namespace, and NVML lists each process under the PID the caller can see. NVML derives utilisation from busy time, power from utilisation (140 W idle to ~1 kW), energy, temperature (32 °C idle towards ~75 °C with a 20 s thermal lag), clocks and P-state, and lists processes. The tray's BMC reads the same table and applies the same model, so its sensors agree with NVML ([BMCs](bmc-redfish.md#how-it-works)). GPU memory is accounted against the 186 GB capacity (allocations beyond it fail with out-of-memory) but is lazily backed, so large GPU buffers cost no host RAM.

**Management inputs.** NVML reads the tray's sideband (written by the BMCs and the partition controller): NVLinks disabled by the tray BMC or switch BMC report inactive (`nvidia-smi topo -m` shows e.g. `NV16`), and each GPU reports the clique of its NVLink partition as of its last reset.

**GPU reset.** `nvidia-smi --gpu-reset` (`-r`, optionally `-i`) needs root and refuses a GPU any process still uses, like the real one. A reset returns the GPU to idle (utilisation, temperature) and applies a pending NVLink partition change: as on GB200, a GPU takes its new partition's clique only at a reset or a node reboot, and until then `nvidia-smi -q` shows `GPU Recovery Action : GPU_RESET` (NVML field `NVML_FI_DEV_GET_GPU_RECOVERY_ACTION`). Each tray resets its GPUs at boot (`fakegpu-boot-reset.service`), and in Slurm mode the epilog resets a job's GPUs when it ends ([Slurm](slurm.md#how-it-works)). There is no memory to scrub: "GPU memory" is the allocating process's own memory, which the kernel zeroes for every new allocation and reclaims at exit; on real hardware the reset is what clears GPU memory between tenants (GB200 NVL Partition User's Guide, §4.1).

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
```

PyTorch, Ray and NCCL code runs unmodified ([Frameworks and examples](frameworks-and-examples.md)).

## Configuration

`inventory/group_vars/all.yml`, applied with `make configure` (written to `/etc/fakegpu.conf` on the trays; every key can be overridden per process with `FAKEGPU_<KEY>`):

| Variable | Default | Meaning |
|---|---|---|
| `fakegpu_count`, `fakegpu_name`, `fakegpu_mem_mb`, `fakegpu_type` | 4, `NVIDIA GB200`, 189471, `gb200` | GPUs per tray, name, memory, Slurm GRES type |
| `fakegpu_nvlinks` | 18 | NVLinks per GPU (Blackwell: 18 × NVLink5) |
| `nvl_cluster_uuid`, `nvl_clique_id` | | NVLink domain UUID, clique of the default partition |
| `fakegpu_latency_scale` | 1.0 | multiplier for all simulated times; 0 disables delays |
| `fakegpu_copy_max_mb` | 64 | copies above this are timed but not performed |

`/etc/fakegpu.conf` also gets `host`, the tray's name, which GPU identities derive from. Debugging: `FAKEGPU_DEBUG=1` logs CUDA entry points resolved to no-ops and unknown export tables.

## Verification

```
bin/ssh sched-worker1 nvidia-smi -L                          # 4 × NVIDIA GB200 with UUIDs
bin/ssh login 'srun -N1 --gpus-per-node=2 bash -c "echo \$CUDA_VISIBLE_DEVICES; nvidia-smi -L -i \$CUDA_VISIBLE_DEVICES"'   # Slurm: the job's 2 GPUs
bin/ssh login 'cd examples/kubernetes && ./submit --wait nvl8-hello.yaml'   # k3s: 8 pods, one GPU each
bin/ssh login sbatch < examples/slurm/gpu-topology.sbatch    # topology, NVLink matrix, fabric per tray
```

While a GPU job runs, `nvidia-smi` on its tray shows its processes with memory, utilisation, power and rising temperature; the same values appear in Prometheus ([Monitoring](monitoring.md)).

## Limitations

- Kernels do not run: tensors computed on the GPU hold zeros or garbage; NCCL collectives behave as if every rank contributed the same data (so cross-rank consistency checks pass); ranks do not communicate, so a failed link or rank does not fail its peers' collectives.
- Data larger than `fakegpu_copy_max_mb` does not round-trip between host and GPU.
- Unusual APIs reach generated no-ops that return success without filling outputs; mainstream PyTorch, Ray, NCCL and NVML paths are implemented.
- Only x86-64; the stubs mimic CUDA 13 (and the CUDA 12 SONAMEs for cuBLAS).

## References

- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/), [NVML API](https://docs.nvidia.com/deploy/nvml-api/), [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/)
- [ZLUDA](https://github.com/vosen/ZLUDA) (Apache-2.0): source of the `libcudart` export-table layouts and integrity check (`fakegpu/dark_api.c`)
- Sources: `fakegpu/cuda_stub.c`, `timing.c`, `nvml_stub.c`, `occupancy.h`, `cublas_stub.c`, `nccl_stub.c`, `cudnn_stub.c`, `nvidia-smi`; role `roles/fakegpu`
