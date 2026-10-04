"""Ray on a Slurm allocation: one GPU task per GPU, each running a short
matmul loop. Prints the cluster's resources and where every task ran."""
import collections
import socket
import time

import ray


@ray.remote(num_gpus=1)
def gpu_task(i, seconds=3.0):
    import torch

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end, n = time.time() + seconds, 0
    while time.time() < end:
        x @ x
        n += 1
    torch.cuda.synchronize()
    return i, socket.gethostname(), ray.get_gpu_ids()[0], torch.cuda.get_device_name(0), n


def main():
    ray.init(address="auto")
    res = ray.cluster_resources()
    print(f"cluster: {int(res.get('GPU', 0))} GPUs, {int(res.get('CPU', 0))} CPUs, "
          f"accelerator types {[k for k in res if k.startswith('accelerator_type')]}", flush=True)
    t = time.time()
    results = ray.get([gpu_task.remote(i) for i in range(int(res["GPU"]))])
    per_node = collections.Counter(host for _, host, *_ in results)
    for i, host, gpu, name, n in sorted(results):
        print(f"task {i}: {host} GPU {gpu} ({name}) {n} matmuls", flush=True)
    print(f"{len(results)} tasks on {dict(per_node)} in {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
