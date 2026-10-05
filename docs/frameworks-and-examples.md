[← README](../README.md) · **Frameworks and examples**

# Frameworks and examples

## Overview

`make frameworks` installs a Python venv on the shared volume (`/shared/venv`) with PyTorch (CUDA 13 build), Ray, NumPy and boto3, activated automatically in login shells. Because the fake GPU stack implements the CUDA runtime, cuBLAS, cuDNN and NCCL paths these frameworks use, they run unmodified on the emulated GPUs with simulated timing ([Fake GPUs](fake-gpu.md)).

## Usage

The examples are in `examples/`; copy them to the cluster and submit from the login node:

```
bin/scp -r examples slurm:
bin/ssh slurm 'cd examples && sbatch nvl8-hello.sbatch'
bin/ssh slurm 'cat examples/nvl8-hello-*.out'
```

| Example | Resources | What it does |
|---|---|---|
| `gpu-topology.sbatch` | 2 nodes × 4 GPUs | prints the Slurm topology, then per tray the GPUs, NVLink matrix and fabric registration |
| `nvl8-hello.sbatch` | 2 nodes × 4 tasks, 1 GPU per task, `--tmp=10G` | each rank reports its node, `CUDA_VISIBLE_DEVICES`, GPU UUID, clique and scratch |
| `ddp-train.sbatch` + `ddp_train.py` | 2 nodes × 4 GPUs (`torchrun`, NCCL) | data-parallel training of an MLP; prints step time, samples/s, TFLOP/s per GPU; arguments are passed through (`--batch`, `--width`, `--layers`, `--steps`) |
| `ray-cluster.sbatch` + `ray_demo.py` | 2 nodes × 4 GPUs | starts a Ray cluster in the allocation (temp and spill on `$SCRATCH`) and runs one GPU task per GPU |

With `scheduler: k3s` the same four examples exist as Kueue-queued JobSets in `examples/k8s/` (`./submit --wait <example>.yaml`); see [Kubernetes (k3s)](kubernetes.md#usage).

Expected `nvl8-hello` output: eight ranks, four per tray, each bound to a different GPU, all in the same NVLink clique:

```
job 14 on sched-worker[1-2]: 8 tasks
0: rank 0 on sched-worker1 CUDA_VISIBLE_DEVICES=0: NVIDIA GB200, GPU-d026216d-…, 1, scratch  198G
…
7: rank 7 on sched-worker2 CUDA_VISIBLE_DEVICES=3: NVIDIA GB200, GPU-d01e22ed-…, 1, scratch  198G
```

**Writing your own jobs:** build models directly on the GPU (`with torch.device("cuda")`): GPU memory is lazily backed, while a model built on the CPU first occupies the tray's host RAM (8 GiB per tray; 10 GiB in k3s mode). Timing reacts to sizes as on real hardware: larger GEMMs take longer, all-reduce time grows with gradient size.

## Verification

`make test` runs all four examples as `joe` through the configured scheduler (Slurm batch jobs, or JobSets in k3s mode) and checks their output (topology and fabric lines, rank 7, all 8 DDP ranks, 8 Ray tasks); in Slurm mode also that every job is `COMPLETED` in accounting.

## Configuration

| Variable | Default |
|---|---|
| `frameworks_venv` | `/shared/venv` |
| `torch_index` | `https://download.pytorch.org/whl/cu130` |
| `frameworks_packages` | `torch`, `ray[default]`, `numpy`, `boto3` |

## Limitations

- Results are not numerically meaningful (kernels do not compute).
- Each tray has 4 CPUs and 8 GiB RAM (10 GiB in k3s mode): CPU-side work (data loading, Python overhead) is slow compared with the simulated GPU work.

## References

- [PyTorch distributed](https://pytorch.org/docs/stable/distributed.html), [torchrun](https://pytorch.org/docs/stable/elastic/run.html)
- [Ray on Slurm](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)
- Sources: `examples/`, `examples/k8s/`, `playbooks/frameworks.yml`
