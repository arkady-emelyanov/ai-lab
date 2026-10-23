"""What a container given a MIG slice sees (make test, MIG): its CUDA
devices with name, memory and SMs, and whether an allocation beyond the
slice's memory fails (2: out of memory)."""
import ctypes

cu = ctypes.CDLL("libcuda.so.1")
cu.cuInit(0)
n = ctypes.c_int()
cu.cuDeviceGetCount(ctypes.byref(n))
print(f"cuda devices: {n.value}")
for i in range(n.value):
    name = ctypes.create_string_buffer(96)
    cu.cuDeviceGetName(name, 96, i)
    mem = ctypes.c_size_t()
    cu.cuDeviceTotalMem_v2(ctypes.byref(mem), i)
    sms = ctypes.c_int()
    cu.cuDeviceGetAttribute(ctypes.byref(sms), 16, i)
    print(f"cuda device {i}: {name.value.decode()}, {mem.value >> 20} MiB, {sms.value} SMs")
ctx = ctypes.c_void_p()
cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0)
cu.cuCtxSetCurrent(ctx)
ptr = ctypes.c_ulonglong()
within = cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(20 << 30))
beyond = cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(4 << 30))
print(f"alloc within: {within}, beyond: {beyond}")
