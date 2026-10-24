"""GPU memory shared over NVLink between two trays (multi-node NVLink), as
NCCL, PyTorch's symmetric memory and MNNVL all-reduces do on GB200: rank 0
creates memory that may be shared as a fabric handle, writes a pattern into
it and hands the 64-byte handle over through a file; rank 1, on the other
tray, imports and maps it and reads the pattern back.

The driver allows fabric handles only to a process with an IMEX channel
(/dev/nvidia-caps-imex-channels/channel<n>): in Kubernetes a pod of a
ComputeDomain gets one (imex-share.yaml); without it, cuMemCreate fails with
CUDA_ERROR_NOT_PERMITTED. Needs no frameworks, only the driver (ctypes).

The ranks meet in RENDEZVOUS_DIR, which rank 0 removes at the end. A rank
whose peer failed, or did not get there within 5 minutes, stops with an error.

    python3 imex_share.py RANK RENDEZVOUS_DIR
"""
import ctypes
import glob
import os
import shutil
import socket
import sys
import time

RANK, RV = int(sys.argv[1]), sys.argv[2]
SIZE, FABRIC = 2 << 20, 0x8
PATTERN = bytes(range(256)) * 16
HANDLE, DONE, FAILED = (os.path.join(RV, f) for f in ("handle", "done", "failed"))
DEADLINE = time.monotonic() + 300  # for the other rank to start and reach each step
u64 = ctypes.c_ulonglong


class MemProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("loc_type", ctypes.c_int),
                ("loc_id", ctypes.c_int), ("win32", ctypes.c_void_p), ("allocFlags", ctypes.c_ubyte * 8)]


class AccessDesc(ctypes.Structure):
    _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_int)]


cu = ctypes.CDLL("libcuda.so.1")
cu.cuMemMap.argtypes = [u64, ctypes.c_size_t, ctypes.c_size_t, u64, u64]
cu.cuMemSetAccess.argtypes = [u64, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
cu.cuMemExportToShareableHandle.argtypes = [ctypes.c_void_p, u64, ctypes.c_int, u64]
cu.cuMemImportFromShareableHandle.argtypes = [ctypes.POINTER(u64), ctypes.c_void_p, ctypes.c_int]
cu.cuMemcpyHtoD_v2.argtypes = [u64, ctypes.c_void_p, ctypes.c_size_t]
cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, u64, ctypes.c_size_t]
cu.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]


def check(name, rc):
    if rc:  # tell the other rank, so that it does not wait for us
        err = ctypes.c_char_p()
        cu.cuGetErrorName(rc, ctypes.byref(err))
        os.makedirs(RV, exist_ok=True)
        open(FAILED, "w").close()
        sys.exit(f"rank {RANK} on {socket.gethostname()}: {name} failed: {(err.value or b'?').decode()}")


def wait_for(path):
    while not os.path.exists(path):
        failed = os.path.exists(FAILED)
        if failed or time.monotonic() > DEADLINE:
            shutil.rmtree(RV, ignore_errors=True)  # the other rank is gone: we are the last
            sys.exit(f"rank {RANK} on {socket.gethostname()}: "
                     + ("the other rank failed" if failed else f"gave up waiting for {path} after 300 s"))
        time.sleep(0.2)


def mapped(h):
    ptr = u64()
    check("cuMemAddressReserve", cu.cuMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(SIZE), ctypes.c_size_t(0),
                                                         u64(0), u64(0)))
    check("cuMemMap", cu.cuMemMap(ptr.value, SIZE, 0, h, 0))
    check("cuMemSetAccess", cu.cuMemSetAccess(ptr.value, SIZE, ctypes.byref(AccessDesc(1, 0, 3)), 1))
    return ptr.value


check("cuInit", cu.cuInit(0))
ctx = ctypes.c_void_p()
check("cuDevicePrimaryCtxRetain", cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0))
check("cuCtxSetCurrent", cu.cuCtxSetCurrent(ctx))
channels = sorted(os.path.basename(c) for c in glob.glob("/dev/nvidia-caps-imex-channels/channel*"))
print(f"rank {RANK} on {socket.gethostname()}: IMEX channels {channels or 'none'}", flush=True)

if RANK == 0:
    h = u64()
    check("cuMemCreate", cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(SIZE), ctypes.byref(MemProp(1, FABRIC, 1, 0)),
                                        u64(0)))
    ptr = mapped(h.value)
    buf = ctypes.create_string_buffer(PATTERN, len(PATTERN))
    check("cuMemcpyHtoD", cu.cuMemcpyHtoD_v2(ptr, buf, len(PATTERN)))
    handle = ctypes.create_string_buffer(64)
    check("cuMemExportToShareableHandle", cu.cuMemExportToShareableHandle(handle, h.value, FABRIC, u64(0)))
    os.makedirs(RV, exist_ok=True)
    with open(HANDLE + ".tmp", "wb") as f:
        f.write(handle.raw)
    os.rename(HANDLE + ".tmp", HANDLE)
    print(f"rank 0 on {socket.gethostname()}: exported {SIZE >> 20} MiB as a fabric handle", flush=True)
    wait_for(DONE)  # the memory lives as long as its creator holds it
    shutil.rmtree(RV)
else:
    wait_for(HANDLE)
    with open(HANDLE, "rb") as f:
        handle = ctypes.create_string_buffer(f.read(), 64)
    h = u64()
    check("cuMemImportFromShareableHandle", cu.cuMemImportFromShareableHandle(ctypes.byref(h), handle, FABRIC))
    ptr = mapped(h.value)
    buf = ctypes.create_string_buffer(len(PATTERN))
    check("cuMemcpyDtoH", cu.cuMemcpyDtoH_v2(buf, ptr, len(PATTERN)))
    ok = buf.raw == PATTERN
    print(f"rank 1 on {socket.gethostname()}: imported the fabric handle, read {'the pattern' if ok else 'wrong data'}",
          flush=True)
    open(DONE, "w").close()
    sys.exit(0 if ok else 1)
