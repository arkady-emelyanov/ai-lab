# Examples

The same four jobs for each scheduler; use the folder that matches `scheduler` in `inventory/group_vars/all.yml`.

| Job | Slurm (`slurm/`) | Kubernetes (`kubernetes/`) | What it does |
|---|---|---|---|
| GPU topology | `gpu-topology.sbatch` | `gpu-topology.yaml` | one task or pod per tray with its 4 GPUs: `nvidia-smi -L`, `topo -m`, NVLink fabric state |
| NVL8 hello | `nvl8-hello.sbatch` | `nvl8-hello.yaml` | 8 tasks or pods × 1 GPU across the NVLink domain; each reports its tray, GPU UUID and clique |
| DDP training | `ddp-train.sbatch` | `ddp-train.yaml` | PyTorch DDP over NCCL, 2 trays × 4 GPUs with `torchrun`; runs `ddp_train.py` |
| Ray cluster | `ray-cluster.sbatch` | `ray-cluster.yaml` | Ray head and worker, one GPU task per GPU; runs `ray_demo.py` |

`ddp_train.py` and `ray_demo.py` are shared by both versions. The DDP and Ray jobs need the frameworks venv (`make frameworks`).

## Running them

From the repository root on your machine, copy the folder to joe's home and log in to the login node:

```
bin/scp -r examples login:
bin/ssh login
```

The rest runs on the login node as `joe`.

**Slurm:**

```
joe@sched-login:~$ cd examples/slurm
joe@sched-login:~/examples/slurm$ sbatch nvl8-hello.sbatch    # output in nvl8-hello-<job id>.out
joe@sched-login:~/examples/slurm$ squeue; sacct -X
```

**Kubernetes:** the manifests are JobSets queued in Kueue. `submit` fills in your uid, gid and home, gives the run a unique name and prints it; with `--wait` it waits for the run to finish and writes the pods' output to `<name>.out`, like `sbatch --wait`.

```
joe@sched-login:~$ cd examples/kubernetes
joe@sched-login:~/examples/kubernetes$ ./submit --wait nvl8-hello.yaml   # prints the run name, writes <name>.out
joe@sched-login:~/examples/kubernetes$ kubectl get jobsets,workloads      # joe's kubeconfig, his namespace
```

From your machine, `bin/kubectl -n joe get jobsets,workloads` shows the same as cluster admin.

More: [Frameworks and examples](../docs/frameworks-and-examples.md), [Slurm](../docs/slurm.md#usage), [Kubernetes (k3s)](../docs/kubernetes.md#usage).
