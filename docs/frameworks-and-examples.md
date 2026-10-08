[← README](../README.md) · **Frameworks and examples**

# Frameworks and examples

## Overview

`make frameworks` installs a Python venv on the shared volume (`/shared/venv`) with PyTorch (CUDA 13 build), Ray, NumPy and boto3; login shells activate it. The frameworks run unmodified on the emulated GPUs, with modelled timing ([Emulated GPUs](fake-gpu.md)).

## Usage

The examples are in `examples/` ([overview](../examples/README.md)): the same four jobs in `slurm/` and `kubernetes/`, sharing the Python programs. Copy them to the cluster and submit from the login node:

```
bin/scp -r examples login:
bin/ssh login 'cd examples/slurm && sbatch nvl8-hello.sbatch'
bin/ssh login 'cat examples/slurm/nvl8-hello-*.out'
```

| Example | Resources | What it does |
|---|---|---|
| `gpu-topology.sbatch` | 2 nodes × 4 GPUs | prints the Slurm topology, then per tray the GPUs, NVLink matrix and fabric registration |
| `nvl8-hello.sbatch` | 2 nodes × 4 tasks, 1 GPU per task, `--tmp=10G` | each rank reports its node, `CUDA_VISIBLE_DEVICES`, GPU UUID, clique and scratch |
| `ddp-train.sbatch` + `../ddp_train.py` | 2 nodes × 4 GPUs (`torchrun`, NCCL) | data-parallel training of an MLP; prints step time, samples/s, TFLOP/s per GPU; arguments are passed through (`--batch`, `--width`, `--layers`, `--steps`) |
| `ray-cluster.sbatch` + `../ray_demo.py` | 2 nodes × 4 GPUs | starts a Ray cluster in the allocation (temp and spill on `$SCRATCH`) and runs one GPU task per GPU |

With `scheduler: k3s` use the Kueue-queued JobSets in `examples/kubernetes/` (`./submit --wait <example>.yaml`); see [Kubernetes (k3s)](kubernetes.md#usage).

Expected `nvl8-hello` output: eight ranks, four per tray, each bound to a different GPU, all in the same NVLink clique:

```
job 14 on sched-worker[1-2]: 8 tasks
0: rank 0 on sched-worker1 CUDA_VISIBLE_DEVICES=0: NVIDIA GB200, GPU-d026216d-…, 1, scratch  198G
…
7: rank 7 on sched-worker2 CUDA_VISIBLE_DEVICES=3: NVIDIA GB200, GPU-d01e22ed-…, 1, scratch  198G
```

Expected `ddp-train` results (Slurm, the defaults `--batch 64 --width 8192 --layers 8` with `--steps 16000`, measured on the lab at commit `680c432`; the numbers vary a little with the host's CPU):

| Run | Step | 16,000 steps (`sacct` Elapsed) | GPU load during the run |
|---|---|---|---|
| All 8 GPUs in one NVLink partition | ~4.2 ms, ~123k samples/s, 50 TFLOP/s/GPU | 1:18–1:19 | 85–87 % utilisation, 850–890 W, ~56 °C |
| Split across two partitions (one per tray) | ~9.5 ms, ~54k samples/s, 22 TFLOP/s/GPU | 2:39–2:41 | ~93 % utilisation, 925–945 W, ~67 °C |

The step time is mostly NCCL and launch overhead; the model's BF16 GEMMs are a small share of it.

Split across partitions, the gradient all-reduce crosses InfiniBand, so the step takes more than twice as long. The GPUs count the longer collectives as busy time, so utilisation and power go up ([NVLink partitions](nvlink-partitions.md)).

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
- Sources: `examples/` (`slurm/`, `kubernetes/`, shared `ddp_train.py`, `ray_demo.py`), `playbooks/frameworks.yml`
