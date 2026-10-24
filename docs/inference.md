[← README](../README.md) · **Inference servers**

# Inference servers (vLLM, SGLang)

## Overview

The lab runs the official images of two LLM inference servers on its emulated GB200s, under Kubernetes (k3s mode): SGLang and vLLM. The emulated GPUs compute nothing ([Emulated GPUs](fake-gpu.md)), so answers are meaningless text of exactly `max_tokens` tokens; what the lab tests is everything around the math: deployment, scheduling and queueing in Kueue, startup, the OpenAI-compatible API, batching and caching, communication setup, load handling, and how tensor parallelism uses NVLink and InfiniBand.

**Use SGLang** to run inference on the lab: its official image works unmodified, on one GPU and with tensor parallelism inside a tray and across both trays. vLLM's official image is reliable only for short single requests; it needs a patched image for anything more ([vLLM](#vllm)).

| | SGLang | vLLM |
|---|---|---|
| Image | `lmsysorg/sglang:v0.5.21-cu130-runtime`, unmodified | `vllm/vllm-openai:v0.31.0-x86_64`, unmodified |
| One GPU | 832 of 832 requests (up to 64 concurrent) | short single requests work; under load, requests stall or the engine crashes |
| Tensor parallelism in a tray (TP=4) | 832 of 832 (up to 64 concurrent) | same limits as one GPU |
| Tensor parallelism across trays (TP=8) | 832 of 832 (up to 64 concurrent) | same limits as one GPU |
| vLLM with a one-change patched image | – | TP=4 and TP=8: 832 of 832 each |
| Examples | `sglang-serve.yaml`, `sglang-tp8.yaml` | `vllm-serve.yaml` |

How to run the examples: [examples](../examples/README.md).

## SGLang

The recommended inference server on the lab. Works unmodified. The pods set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (see [What the GPUs return](#what-the-gpus-return)); with that, every answer is `"!!!..."` (token 0) with exactly `max_tokens` tokens and `finish_reason: length`.

Measured (Qwen2.5-0.5B on one GPU, Qwen3-0.6B with tensor parallelism; 832 requests per configuration, in batches of 64, 256 and 512 at 16, 32 and 64 concurrent (the split-partition run: 256 at 32), of random length, every seventh with prompt logprobs through `"echo": true, "logprobs": 1`):

| Run | Result |
|---|---|
| one GPU | 832 of 832 requests complete |
| TP=4, one tray | 832 of 832 |
| TP=8, both trays, one NVLink partition | 832 of 832; with 256 requests, 320 GB on NVLink, nothing on InfiniBand |
| TP=8, trays in separate NVLink partitions | 256 of 256; 262 GB on NVLink, 23 GB on InfiniBand per tray |

Inside a tray SGLang's all-reduce runs over symmetric memory shared between the GPUs (the log shows `All Reduce config: symmetric_memory = 40.01 MB`), built on the emulated driver's shared GPU memory ([Emulated GPUs](fake-gpu.md)). Across trays it uses NCCL, whose traffic the lab models per fabric.

## vLLM

### Unmodified

The official image starts and serves on one GPU and with tensor parallelism, but not reliably under load:

- a request can stall at 0 tokens per second;
- the engine can crash on a scheduler assertion (`assert not prompt_logprobs_tensors`, typically with prompt logprobs), and the pod restarts;
- with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` requests stall (the count reads 0; seen on the first request).

Short single requests mostly work, so `vllm-serve.yaml` suits trying the deployment, the API and the Kueue queueing; it is not reliable for load tests.

**Why.** After each step vLLM's model runner (V2, the default) tells the scheduler how many tokens each request produced: 1, or 0 while the request is still in the middle of its prompt. It computes that number on the GPU and copies it back. No GPU kernel runs here, so the number is whatever the GPU memory holds: stale bytes (for example `1065353216`, the bit pattern of float 1.0), or 0 when memory is zeroed. A request told "0 tokens" makes no progress; one that also asked for prompt logprobs trips the scheduler's check that a request without new tokens has no prompt logprobs, which is fatal to the engine. SGLang shows no such dependence: its load tests run clean (inferred from its behaviour, not from its code).

### With a patched image

One small change to vLLM makes it work: for steps without speculative decoding, report that count from the CPU, which already knows it, instead of copying it from the GPU. On a real GPU the change gives identical outputs (token ids, logprobs, prompt logprobs). It is not upstream yet; until it is, vLLM in the lab needs an image built with it (vLLM's `docker/Dockerfile` with `VLLM_USE_PRECOMPILED=1` builds one from a patched checkout in under an hour, reusing the published kernels), imported into each tray's k3s (`k3s ctr -n k8s.io images import`) and used with `imagePullPolicy: Never`.

Measured with the patched v0.31.0 image and `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (Qwen2.5-1.5B for TP=4, Qwen3-0.6B for TP=8; the same requests as for SGLang):

| Run | Result |
|---|---|
| TP=4, one tray | 832 of 832 requests complete |
| TP=8, both trays, one NVLink partition | 832 of 832; with 256 requests, 407 GB on NVLink, nothing on InfiniBand |
| TP=8, trays in separate NVLink partitions | 256 of 256; 279 GB on NVLink, 26 and 22 GB on InfiniBand |

vLLM then uses its GB200 communication paths: FlashInfer's MNNVL all-reduce (`Initialized FlashInfer Allreduce norm fusion workspace with backend=mnnvl`) and its custom all-reduce (`Using ['FLASHINFER', 'CUSTOM', 'PYNCCL'] all-reduce backends`), on the emulated driver's shared GPU memory and CUDA IPC, without `--disable-custom-all-reduce`.

Run it with `--enforce-eager` (nothing to gain from CUDA graphs or `torch.compile` without real kernels).

## What the GPUs return

Outputs no kernel computed hold whatever their memory held. Fresh GPU allocations are zeroed, but PyTorch's caching allocator reuses blocks, so by default they hold stale data from earlier tensors: token ids outside the vocabulary, logprobs that are garbage. `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (a PyTorch setting) makes every tensor a fresh allocation, so these outputs read as zeros: token 0 and logprob 0.0, valid and repeatable. Side effect: PyTorch's CUDA tensor sharing between processes (`torch.multiprocessing`) needs the caching allocator, as on real GPUs; neither server needs it.

## Requirements

- **Tray size.** The default k3s trays (4 CPU, 10 GiB) fit one-GPU servers. Tensor parallelism over a tray's four GPUs runs a server process per GPU plus the API server, and the TP=4 and TP=8 examples request 4 CPU and 14 GiB per pod. Raise the trays in `local.yml` and run `make provision`, which applies the new limits to the running trays:

  ```yaml
  instance_limits_k3s:
    controller: { cpu: 2, memory: 3GiB }
    workers: { cpu: 8, memory: 24GiB }
  ```

- **Heads.** The tensor-parallel size must divide the model's attention heads: Qwen2.5-0.5B has 14 (TP 1, 2, 7), Qwen2.5-1.5B 12 (TP 1, 2, 4), Qwen3-0.6B 16 (TP 1, 2, 4, 8).
- **First start.** Pulls the image (SGLang ~14 GB, vLLM ~9 GB per tray) and downloads the model to `~/.cache/huggingface` on `/shared`.
- **Readiness.** SGLang's `/health` goes through its scheduler and can take more than a second under load; the example's probe allows 10 seconds.

## Topology

TP=8 spans both trays. With both trays in one NVLink partition (the default) its all-reduces cross trays over NVLink; split into two partitions, the cross-tray part goes over InfiniBand, as NCCL does across NVLink domains on GB200. Kueue keeps the two pods in one NVLink domain (`podset-required-topology`); with the trays split it refuses the job, and `podset-preferred-topology` lets it span partitions. Commands: [examples](../examples/README.md); partitions: [NVLink partitions](nvlink-partitions.md); metrics: `nvlink_gpu_tx_bytes_total`, `node_infiniband_port_data_transmitted_bytes_total` ([Monitoring](monitoring.md)).

All-reduces inside a tray that run as the servers' own kernels (SGLang's symmetric-memory all-reduce, vLLM's custom and FlashInfer all-reduces) do not go through the emulated NCCL, so their NVLink traffic does not show in the metrics; NCCL traffic does.

## Limitations

- No real math: outputs are token 0 (or stale data without `PYTORCH_NO_CUDA_MEMORY_CACHING=1`), so nothing about answer quality, accuracy or numerical behaviour can be tested.
- Speculative decoding (not tested): how many draft tokens are accepted depends on the model's outputs, which are never computed, so it is not expected to work, at any tensor-parallel size, on either server.
- Structured output (grammars, JSON schemas; not tested): every token is id 0, which a grammar may reject, so such requests are expected to end with an error.
- Mixture-of-experts models with expert parallelism: routing decisions come from the GPU; untested.
- vLLM needs the patched image for anything beyond short single requests (see [vLLM](#vllm)).

More: [Emulated GPUs](fake-gpu.md), [Kubernetes (k3s)](kubernetes.md), [examples](../examples/README.md).
