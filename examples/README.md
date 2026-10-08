# Examples

The same four jobs for each scheduler; use the folder that matches `scheduler` in `local.yml`.

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

**vLLM (Kubernetes only):** `vllm-serve.yaml` runs vLLM's OpenAI-compatible server from the official image, unmodified, on one GPU, as a Deployment and Service queued in Kueue (Qwen2.5-0.5B-Instruct). The emulated GPUs compute nothing, so the model never picks a real token: the sampler's output holds leftover bytes (with this image, id 1065353216, the bit pattern of float 1.0, outside any vocabulary), so every answer is empty text with exactly `max_tokens` tokens and `finish_reason: length`. Everything around the text is real: the API, batching, KV cache and prefix caching, vLLM's `/metrics` and GPU load. The first start downloads the model and pulls the image (~9 GB).

**Limits of vLLM on the emulated GPUs.** vLLM's model runner (V2, the default in this image) keeps part of its per-step bookkeeping on the GPU: how many tokens each request sampled, positions, sequence lengths. No GPU kernel runs, so vLLM reads leftover bytes there too, and depending on them a request can:

- stall: it stays running at 0 tokens/s (seen with 32 concurrent long requests and, occasionally, with a single one)
- crash the engine: an assertion in vLLM's scheduler fails (seen with `prompt_logprobs` on a repeated prompt) and the pod restarts

Short single requests mostly work, so the example suits trying the deployment, the API and the Kueue queueing; it is not reliable for load tests. Tensor parallelism (`--tensor-parallel-size 4` over a tray's GPUs) starts with `--disable-custom-all-reduce` and `VLLM_ALLREDUCE_USE_SYMM_MEM=0` (vLLM's custom all-reduce needs CUDA IPC, PyTorch's symmetric memory CUDA virtual memory management, which the emulated driver lacks) and trays with more RAM (six processes with PyTorch, ~10 GiB), but runs into the same limits.

Submit as joe on the login node, under the fixed name `vllm` (the Deployment and its Service; submitting again updates it in place):

```
bin/ssh login 'cd examples/kubernetes && ./submit --name vllm vllm-serve.yaml'
```

The rest runs on your machine, through `bin/kubectl` (cluster admin). Wait until it is ready (about a minute, longer on the first start):

```
bin/kubectl -n joe rollout status deploy/vllm
```

Forward its port, in a second terminal (Ctrl-C ends it):

```
bin/kubectl -n joe port-forward svc/vllm 8000
```

Query it; any OpenAI client works against `http://localhost:8000/v1`:

```
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "prompt": "Hello", "max_tokens": 64}'
```

Stop it:

```
bin/kubectl -n joe delete deploy,svc vllm
```

More: [Frameworks and examples](../docs/frameworks-and-examples.md), [Slurm](../docs/slurm.md#usage), [Kubernetes (k3s)](../docs/kubernetes.md#usage).
