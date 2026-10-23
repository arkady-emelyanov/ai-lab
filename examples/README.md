# Examples

The same four jobs for each scheduler; use the folder that matches `scheduler` in `local.yml`.

| Job | Slurm (`slurm/`) | Kubernetes (`kubernetes/`) | What it does |
|---|---|---|---|
| GPU topology | `gpu-topology.sbatch` | `gpu-topology.yaml` | one task or pod per tray with its 4 GPUs: `nvidia-smi -L`, `topo -m`, NVLink fabric state |
| NVL8 hello | `nvl8-hello.sbatch` | `nvl8-hello.yaml` | 8 tasks or pods × 1 GPU across the NVLink domain; each reports its tray, GPU UUID and clique |
| DDP training | `ddp-train.sbatch` | `ddp-train.yaml` | PyTorch DDP over NCCL, 2 trays × 4 GPUs with `torchrun`; runs `ddp_train.py` |
| Ray cluster | `ray-cluster.sbatch` | `ray-cluster.yaml` | Ray head and worker, one GPU task per GPU; runs `ray_demo.py` |
| Ray job (KubeRay) | – | `ray-job.yaml` | the same as a KubeRay `RayJob`: KubeRay starts the Ray cluster, submits `ray_demo.py` and deletes the cluster when it ends; shows in the KubeRay dashboard ([KubeRay](../docs/kuberay.md)) |

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

**vLLM (Kubernetes only):** for inference on the lab, prefer SGLang (below); `vllm-serve.yaml` runs vLLM's OpenAI-compatible server from the official image, unmodified, on one GPU, as a Deployment and Service queued in Kueue (Qwen2.5-0.5B-Instruct). The emulated GPUs compute nothing, so the model never picks a real token: the sampler's output holds leftover bytes (with this image, id 1065353216, the bit pattern of float 1.0, outside any vocabulary), so every answer is empty text with exactly `max_tokens` tokens and `finish_reason: length`. Everything around the text is real: the API, batching, KV cache and prefix caching, vLLM's `/metrics` and GPU load. The first start downloads the model and pulls the image (~9 GB).

**Limits of vLLM on the emulated GPUs.** vLLM's model runner (V2, the default in this image) keeps part of its per-step bookkeeping on the GPU: how many tokens each request sampled, positions, sequence lengths. No GPU kernel runs, so vLLM reads leftover bytes there too, and depending on them a request can:

- stall: it stays running at 0 tokens/s (seen with 32 concurrent long requests and, occasionally, with a single one)
- crash the engine: an assertion in vLLM's scheduler fails (seen with `prompt_logprobs` on a repeated prompt) and the pod restarts

Short single requests mostly work, so the example suits trying the deployment, the API and the Kueue queueing; it is not reliable for load tests. Tensor parallelism starts but runs into the same limits. A one-change patched vLLM image removes them, inside a tray and across both trays: [Inference servers](../docs/inference.md).

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

Stop it (it holds a GPU while it runs, and jobs that need all eight, such as `make test`'s, wait in Kueue until it is gone):

```
bin/kubectl -n joe delete deploy,svc vllm
```

**SGLang (Kubernetes only), the recommended inference server:** `sglang-serve.yaml` runs SGLang's OpenAI-compatible server from the official image (`lmsysorg/sglang:v0.5.21-cu130-runtime`), unmodified, as a Deployment and Service queued in Kueue (Qwen3-0.6B), on one GPU or, with `--tp` and `nvidia.com/gpu` set to 2 or 4, with tensor parallelism inside a tray. `sglang-tp8.yaml` runs TP=8 across both trays: two pods with four GPUs each, rank 0 serving the API. The emulated GPUs compute nothing, so every answer is `"!!!..."` (token 0) with exactly `max_tokens` tokens and `finish_reason: length`; the pods set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` so that outputs no kernel computed read as zeros (valid token ids) rather than leftover bytes. Unlike vLLM, SGLang does not rely on GPU-computed bookkeeping here: load tests run clean (832 requests at up to 64 concurrent, on one GPU, with TP=4 and with TP=8 across both trays), including prompt logprobs (`"echo": true, "logprobs": 1`). Inside a tray SGLang uses its own all-reduce over symmetric memory shared between the GPUs; across trays it uses NCCL. The first start pulls the image (~14 GB) and downloads the model. Tensor parallelism over a tray's four GPUs (TP=4, and `sglang-tp8.yaml`) needs bigger trays than the default: [Inference servers](../docs/inference.md#requirements).

Submit as joe on the login node:

```
bin/ssh login 'cd examples/kubernetes && ./submit --name sglang sglang-serve.yaml'
bin/kubectl -n joe rollout status deploy/sglang
bin/kubectl -n joe port-forward svc/sglang 30000            # second terminal
curl -s localhost:30000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Hello", "max_tokens": 64}'
bin/kubectl -n joe delete deploy,svc sglang
```

**SGLang across the trays: NVLink vs InfiniBand.** With both trays in one NVLink partition (the default), TP=8's all-reduces stay on NVLink; with the trays in separate partitions they cross InfiniBand. Measured with 256 requests (32 concurrent) per run:

| Trays | NVLink | InfiniBand |
|---|---|---|
| one partition (default) | 320 GB | 0 |
| separate partitions | 262 GB | 23 GB per tray |

```
bin/ssh login 'cd examples/kubernetes && ./submit --name sglang-tp8 sglang-tp8.yaml'
bin/kubectl -n joe port-forward $(bin/kubectl -n joe get pod -o name \
    -l jobset.sigs.k8s.io/jobset-name=sglang-tp8,batch.kubernetes.io/job-completion-index=0) 30000
bin/kubectl -n joe delete jobset sglang-tp8
```

Traffic per fabric (Prometheus, `<bridge>` as in [Lab endpoints](../docs/endpoints.md)): `sum by (instance) (rate(nvlink_gpu_tx_bytes_total[1m]))` and `sum by (instance) (rate(node_infiniband_port_data_transmitted_bytes_total[1m]))`. To run across partitions, split them (`bin/nvlink remove default sched-worker2 && bin/nvlink create tray2 --id 7 sched-worker2 && bin/ssh root@sched-worker2 nvidia-smi --gpu-reset`; restore with `bin/nvlink delete tray2 && bin/nvlink add default sched-worker2 && bin/ssh root@sched-worker2 nvidia-smi --gpu-reset`). Kueue then refuses the job as written, because no single NVLink domain has eight GPUs; change `podset-required-topology` to `podset-preferred-topology` in a copy of `sglang-tp8.yaml` to let it span them.

More: [Frameworks and examples](../docs/frameworks-and-examples.md), [Slurm](../docs/slurm.md#usage), [Kubernetes (k3s)](../docs/kubernetes.md#usage).
