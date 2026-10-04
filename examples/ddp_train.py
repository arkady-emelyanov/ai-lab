"""Synthetic data-parallel training with PyTorch DDP over NCCL.

Each rank trains a small transformer-ish MLP on random data and reports step
time and throughput. Under the emulated GPUs the numbers come from the
simulated kernel, GEMM and all-reduce times, so changing the model size or
batch changes them the way it would on real hardware.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--width", type=int, default=8192)
    p.add_argument("--layers", type=int, default=8)
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    # Build directly on the GPU: no host-RAM copy of the weights (GPU memory on
    # the emulated trays is lazily backed, so large models cost no host RAM).
    with torch.device(dev):
        model = nn.Sequential(*[nn.Sequential(nn.Linear(args.width, args.width, dtype=torch.bfloat16), nn.GELU())
                                for _ in range(args.layers)])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"world={world} params={params / 1e6:.0f}M batch/rank={args.batch} width={args.width}", flush=True)

    x = torch.randn(args.batch, args.width, device=dev, dtype=torch.bfloat16)
    for step in range(1, args.steps + 1):
        t = time.perf_counter()
        loss = model(x).float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        if rank == 0 and (step == 1 or step % 10 == 0):
            # forward + backward = 3 x 2 x params FLOPs per sample
            tflops = 6 * params * args.batch / dt / 1e12
            print(f"step {step:4d}  {dt * 1e3:7.1f} ms  {args.batch * world / dt:8.0f} samples/s  "
                  f"{tflops:6.0f} TFLOP/s/GPU", flush=True)

    print(f"rank {rank}/{world} on {os.uname().nodename} cuda:{local} "
          f"({torch.cuda.get_device_name(local)}) peak mem {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
