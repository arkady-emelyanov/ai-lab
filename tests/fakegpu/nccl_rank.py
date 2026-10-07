"""One NCCL rank on an emulated tray of its own (FAKEGPU_* environment):
runs one operation and prints what it cost, as JSON.

  nccl_rank.py LIBDIR NRANKS RANK IDFILE OP BYTES [ARG]
  OP: allreduce | send (ARG: peer) | recv (ARG: peer) | split (ARG: colour) | uniqueid
"""
import ctypes
import json
import os
import struct
import sys
import time

lib, nranks, rank, idfile, op, nbytes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
arg = int(sys.argv[7]) if len(sys.argv) > 7 else None
cu = ctypes.CDLL(f"{lib}/libcuda.so.1")
# A build directory has it next to libcuda; an install, under preload/.
nccl = ctypes.CDLL(f"{lib}/libnccl.so.2" if os.path.exists(f"{lib}/libnccl.so.2") else f"{lib}/preload/libnccl.so.2")


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


# struct fg_occupancy (fakegpu/occupancy.h): gpu[8] x 56 bytes, proc[128] x 16, pidns[128] x 8,
# reset[8] x 16, net[8] x 16 (ib_tx, ib_rx)
NET = 8 * 56 + 128 * 16 + 128 * 8 + 8 * 16


def counters():
    with open(os.environ["FAKEGPU_STATE_PATH"], "rb") as f:
        data = f.read()
    busy, = struct.unpack_from("<Q", data, 0)
    nvlink_tx, = struct.unpack_from("<Q", data, 40)
    ib_tx, ib_rx = struct.unpack_from("<QQ", data, NET)
    return {"busy_ns": busy, "nvlink_tx": nvlink_tx, "ib_tx": ib_tx, "ib_rx": ib_rx}


if op == "uniqueid":
    ids = []
    for _ in range(2):
        uid = UniqueId()
        nccl.ncclGetUniqueId(ctypes.byref(uid))
        ids.append(ctypes.string_at(ctypes.addressof(uid), 16).hex())
    print(json.dumps({"ids": ids}))
    sys.exit()

cu.cuInit(0)
dev, ctx, buf = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
cu.cuDeviceGet(ctypes.byref(dev), 0)
cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
cu.cuCtxSetCurrent(ctx)
cu.cuMemAlloc_v2(ctypes.byref(buf), ctypes.c_size_t(max(nbytes, 4)))

uid = UniqueId()
if rank == 0:
    nccl.ncclGetUniqueId(ctypes.byref(uid))
    with open(idfile + ".tmp", "wb") as f:
        f.write(ctypes.string_at(ctypes.addressof(uid), 128))  # bytes(char array) stops at a NUL
    os.rename(idfile + ".tmp", idfile)
else:
    while not os.path.exists(idfile):
        time.sleep(0.01)
    with open(idfile, "rb") as f:
        ctypes.memmove(ctypes.addressof(uid), f.read(), 128)

nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
comm = ctypes.c_void_p()
assert nccl.ncclCommInitRank(ctypes.byref(comm), nranks, uid, rank) == 0
parent = ctypes.c_void_p(comm.value)
out = {}
if op == "split":
    child = ctypes.c_void_p()
    assert nccl.ncclCommSplit(comm, arg, rank, ctypes.byref(child), None) == 0
    if child.value:
        n, r = ctypes.c_int(), ctypes.c_int()
        nccl.ncclCommCount(child, ctypes.byref(n))
        nccl.ncclCommUserRank(child, ctypes.byref(r))
        out.update(child_nranks=n.value, child_rank=r.value)
        comm = child
    else:
        out.update(child_nranks=0)
before = counters()
if op in ("allreduce", "split") and comm.value:
    nccl.ncclAllReduce(buf, buf, ctypes.c_size_t(nbytes // 4), 7, 0, comm, None)  # float, sum
elif op == "send":
    nccl.ncclSend(buf, ctypes.c_size_t(nbytes), 0, arg, comm, None)
elif op == "recv":
    nccl.ncclRecv(buf, ctypes.c_size_t(nbytes), 0, arg, comm, None)
cu.cuCtxSynchronize()
after = counters()
out.update({k: after[k] - before[k] for k in after})
if comm.value != parent.value and comm.value:
    nccl.ncclCommDestroy(comm)
nccl.ncclCommDestroy(parent)
print(json.dumps(out))
