"""GPU fraction probe (make test, Run:ai): what a container given part of a GPU
sees through CUDA. Prints key=value lines, then stays up so a second fraction
shares the GPU with it."""
import ctypes
import os
import time

cu = ctypes.CDLL("libcuda.so.1")
cu.cuInit(0)
dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
cu.cuCtxSetCurrent(ctx)
free, total = ctypes.c_size_t(), ctypes.c_size_t()
cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
print(f"gpu={os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
print(f"limit_mib={os.environ.get('NVIDIA_GPU_MEMORY_LIMIT')}")
print(f"total_mib={total.value >> 20}")
ptr = ctypes.c_uint64()
print(f"alloc_within={cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(total.value * 9 // 10))}")
print(f"alloc_beyond={cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(total.value // 5))}", flush=True)
time.sleep(300)
