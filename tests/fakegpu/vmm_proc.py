"""One process on an emulated tray using the fake driver's virtual memory
management (test_vmm.py starts it): creates, exports, imports and maps
GPU memory and multicast objects through ctypes, prints a JSON result.

    vmm_proc.py FAKEGPU local
    vmm_proc.py FAKEGPU export fd|fabric mem|mc RENDEZVOUS
    vmm_proc.py FAKEGPU import fd|fabric mem|mc RENDEZVOUS
    vmm_proc.py FAKEGPU ipc-export|ipc-import RENDEZVOUS

The exporter writes a pattern into its memory and hands the handle over at
RENDEZVOUS (a Unix socket for fds, a file for fabric handles); the importer
maps it and reads the pattern back, and reports each driver result code.
The ipc- pair is vLLM's P2P check over legacy CUDA IPC: the exporter's
1 KiB cudaMalloc holds 1s, the importer sets it to 2s, both read 2s."""
import ctypes
import json
import os
import socket
import sys
import time

FAKEGPU, CMD = sys.argv[1], sys.argv[2]
cu = ctypes.CDLL(os.path.join(FAKEGPU, "libcuda.so.1"))
SIZE = 2 << 20
FD, FABRIC = 0x1, 0x8
PATTERN = bytes(range(256)) * 16


class MemProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("loc_type", ctypes.c_int),
                ("loc_id", ctypes.c_int), ("win32", ctypes.c_void_p), ("allocFlags", ctypes.c_ubyte * 8)]


class McProp(ctypes.Structure):
    _fields_ = [("numDevices", ctypes.c_uint), ("size", ctypes.c_size_t), ("handleTypes", ctypes.c_ulonglong),
                ("flags", ctypes.c_ulonglong)]


class AccessDesc(ctypes.Structure):
    _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_int)]


u64 = ctypes.c_ulonglong
cu.cuMemMap.argtypes = [u64, ctypes.c_size_t, ctypes.c_size_t, u64, u64]
cu.cuMemUnmap.argtypes = [u64, ctypes.c_size_t]
cu.cuMemRelease.argtypes = [u64]
cu.cuMemAddressFree.argtypes = [u64, ctypes.c_size_t]
cu.cuMemSetAccess.argtypes = [u64, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
cu.cuMemExportToShareableHandle.argtypes = [ctypes.c_void_p, u64, ctypes.c_int, u64]
cu.cuMemImportFromShareableHandle.argtypes = [ctypes.POINTER(u64), ctypes.c_void_p, ctypes.c_int]
cu.cuMemcpyHtoD_v2.argtypes = [u64, ctypes.c_void_p, ctypes.c_size_t]
cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, u64, ctypes.c_size_t]
cu.cuMulticastAddDevice.argtypes = [u64, ctypes.c_int]
cu.cuMulticastBindMem.argtypes = [u64, ctypes.c_size_t, u64, ctypes.c_size_t, ctypes.c_size_t, u64]
cu.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, u64]
cu.cuInit(0)
ctx = ctypes.c_void_p()
cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0)
cu.cuCtxSetCurrent(ctx)
rc = {}


def call(name, *args):
    rc[name] = getattr(cu, name)(*args)
    return rc[name]


def attr(a):
    v = ctypes.c_int()
    cu.cuDeviceGetAttribute(ctypes.byref(v), a, 0)
    return v.value


def create(kind, types):
    h = u64()
    if kind == "mem":
        call("cuMemCreate", ctypes.byref(h), ctypes.c_size_t(SIZE), ctypes.byref(MemProp(1, types, 1, 0)), u64(0))
    else:
        call("cuMulticastCreate", ctypes.byref(h), ctypes.byref(McProp(2, SIZE, types, 0)))
    return h.value


def map_(h):
    ptr = u64()
    call("cuMemAddressReserve", ctypes.byref(ptr), ctypes.c_size_t(SIZE), ctypes.c_size_t(0), u64(0), u64(0))
    call("cuMemMap", ptr.value, SIZE, 0, h, 0)
    call("cuMemSetAccess", ptr.value, SIZE, ctypes.byref(AccessDesc(1, 0, 3)), 1)
    return ptr.value


def write(ptr):
    buf = ctypes.create_string_buffer(PATTERN, len(PATTERN))
    call("cuMemcpyHtoD_v2", ptr, buf, len(PATTERN))


def read(ptr):
    buf = ctypes.create_string_buffer(len(PATTERN))
    call("cuMemcpyDtoH_v2", buf, ptr, len(PATTERN))
    return buf.raw == PATTERN


def done(ptr, h):
    call("cuMemUnmap", ptr, SIZE)
    call("cuMemAddressFree", ptr, SIZE)
    call("cuMemRelease", h)


class IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


cu.cuMemsetD8_v2.argtypes = [u64, ctypes.c_ubyte, ctypes.c_size_t]
cu.cuIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcHandle), u64]
cu.cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(u64), IpcHandle, ctypes.c_uint]
cu.cuIpcCloseMemHandle.argtypes = [u64]
cu.cuMemFree_v2.argtypes = [u64]


def all_bytes(ptr, value, n=1024):
    buf = ctypes.create_string_buffer(n)
    cu.cuMemcpyDtoH_v2(buf, ptr, n)
    return buf.raw == bytes([value]) * n


def wait_for(path):
    while not os.path.exists(path):
        time.sleep(0.02)


out = {"rc": rc}
if CMD in ("ipc-export", "ipc-import"):
    rv = sys.argv[3]
    if CMD == "ipc-export":
        ptr, h = u64(), IpcHandle()
        call("cuMemAlloc_v2", ctypes.byref(ptr), ctypes.c_size_t(1024))
        call("cuMemsetD8_v2", ptr.value, 1, 1024)
        call("cuIpcGetMemHandle", ctypes.byref(h), ptr.value)
        own = u64()
        out["open_own"] = cu.cuIpcOpenMemHandle_v2(ctypes.byref(own), h, 1)
        out["still_mine"] = all_bytes(ptr.value, 1)  # sharing kept the contents
        open(rv + ".tmp", "wb").write(bytes(h))  # all 64 bytes (.reserved stops at a NUL)
        os.rename(rv + ".tmp", rv)
        print(json.dumps({"ready": True}), flush=True)
        wait_for(rv + ".written")
        out["sees_peer_write"] = all_bytes(ptr.value, 2)
        call("cuMemFree_v2", ptr.value)
    else:
        ptr = u64()
        h = IpcHandle.from_buffer_copy(open(rv, "rb").read())
        call("cuIpcOpenMemHandle_v2", ctypes.byref(ptr), h, 1)
        if rc["cuIpcOpenMemHandle_v2"] == 0:
            out["sees_peer_data"] = all_bytes(ptr.value, 1)
            call("cuMemsetD8_v2", ptr.value, 2, 1024)
            out["reads_back"] = all_bytes(ptr.value, 2)
            call("cuIpcCloseMemHandle", ptr.value)
        open(rv + ".written", "w").close()
    print(json.dumps(out))
    sys.exit(0)
if CMD == "local":
    out["attrs"] = {a: attr(a) for a in (102, 103, 128, 132)}
    h = create("mem", FD)
    ptr = map_(h)
    write(ptr)
    out["roundtrip"] = read(ptr)
    mtype = ctypes.c_uint()
    cu.cuPointerGetAttribute(ctypes.byref(mtype), 2, ptr)  # MEMORY_TYPE
    out["memory_type"] = mtype.value
    big = u64()
    out["oom"] = cu.cuMemCreate(ctypes.byref(big), ctypes.c_size_t(1 << 40), ctypes.byref(MemProp(1, FD, 1, 0)), u64(0))
    done(ptr, h)
else:
    how, kind, rendezvous = sys.argv[3], sys.argv[4], sys.argv[5]
    types = FD if how == "fd" else FABRIC
    if CMD == "export":
        h = create(kind, types)
        if rc.get("cuMemCreate", rc.get("cuMulticastCreate")):  # nothing to share
            print(json.dumps({"ready": True}), flush=True)
            print(json.dumps(out))
            sys.exit(0)
        ptr = map_(h) if kind == "mem" else None
        if ptr:
            write(ptr)
        if how == "fd":
            fd = ctypes.c_int()
            call("cuMemExportToShareableHandle", ctypes.byref(fd), h, FD, 0)
            srv = socket.socket(socket.AF_UNIX)
            srv.bind(rendezvous)
            srv.listen(1)
            print(json.dumps({"ready": True}), flush=True)
            conn, _ = srv.accept()
            socket.send_fds(conn, [b"h"], [fd.value])
            conn.recv(1)  # the importer has mapped it
        else:
            fh = ctypes.create_string_buffer(64)
            call("cuMemExportToShareableHandle", fh, h, FABRIC, 0)
            with open(rendezvous + ".tmp", "wb") as f:
                f.write(fh.raw)
            os.rename(rendezvous + ".tmp", rendezvous)
            print(json.dumps({"ready": True}), flush=True)
            while not os.path.exists(rendezvous + ".done"):
                time.sleep(0.05)
        if ptr:
            done(ptr, h)
        else:
            call("cuMemRelease", h)
    else:
        h = u64()
        if how == "fd":
            conn = socket.socket(socket.AF_UNIX)
            conn.connect(rendezvous)
            _, fds, _, _ = socket.recv_fds(conn, 1, 1)
            call("cuMemImportFromShareableHandle", ctypes.byref(h), ctypes.c_void_p(fds[0]), FD)
            os.close(fds[0])
        else:
            fh = ctypes.create_string_buffer(open(rendezvous, "rb").read(), 64)
            call("cuMemImportFromShareableHandle", ctypes.byref(h), fh, FABRIC)
        if rc["cuMemImportFromShareableHandle"] == 0:
            if kind == "mem":
                ptr = map_(h.value)
                out["read"] = read(ptr)
                done(ptr, h.value)
            else:
                call("cuMulticastAddDevice", h.value, 0)
                ptr = map_(h.value)
                mem = create("mem", types)
                call("cuMulticastBindMem", h.value, 0, mem, 0, SIZE, 0)
                done(ptr, h.value)
                call("cuMemRelease", mem)
        if how == "fd":
            conn.send(b"x")
        else:
            open(rendezvous + ".done", "w").close()
print(json.dumps(out))
