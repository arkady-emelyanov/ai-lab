"""ctypes bindings for NVML's MIG API, for the MIG tests (test_mig_*.py),
which run their checks in subprocesses on an emulated tray: signatures and
structures as in nvml.h (go-nvml v0.13), each call returning NVML's return
code (and its results). Tests import it with PYTHONPATH set to this
directory and print JSON."""
import ctypes
import os
from pathlib import Path

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"

SUCCESS, INVALID_ARGUMENT, NOT_SUPPORTED, NO_PERMISSION, NOT_FOUND = 0, 2, 3, 4, 6
INSUFFICIENT_RESOURCES, IN_USE, RESET_REQUIRED = 23, 19, 16

u32, u64, dev_t = ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_void_p


class ProfileInfo(ctypes.Structure):  # nvmlGpuInstanceProfileInfo_t (v1, no version field)
    _fields_ = [("id", u32), ("isP2pSupported", u32), ("sliceCount", u32), ("instanceCount", u32),
                ("multiprocessorCount", u32), ("copyEngineCount", u32), ("decoderCount", u32), ("encoderCount", u32),
                ("jpegCount", u32), ("ofaCount", u32), ("memorySizeMB", u64)]


class ProfileInfoV2(ctypes.Structure):
    _fields_ = [("version", u32)] + ProfileInfo._fields_ + [("name", ctypes.c_char * 96)]


class ProfileInfoV3(ctypes.Structure):
    _fields_ = [("version", u32), ("id", u32), ("sliceCount", u32), ("instanceCount", u32),
                ("multiprocessorCount", u32), ("copyEngineCount", u32), ("decoderCount", u32), ("encoderCount", u32),
                ("jpegCount", u32), ("ofaCount", u32), ("memorySizeMB", u64), ("name", ctypes.c_char * 96),
                ("capabilities", u32)]


class Placement(ctypes.Structure):
    _fields_ = [("start", u32), ("size", u32)]


class GpuInstanceInfo(ctypes.Structure):
    _fields_ = [("device", dev_t), ("id", u32), ("profileId", u32), ("placement", Placement)]


class CiProfileInfo(ctypes.Structure):  # nvmlComputeInstanceProfileInfo_t (v1)
    _fields_ = [("id", u32), ("sliceCount", u32), ("instanceCount", u32), ("multiprocessorCount", u32),
                ("sharedCopyEngineCount", u32), ("sharedDecoderCount", u32), ("sharedEncoderCount", u32),
                ("sharedJpegCount", u32), ("sharedOfaCount", u32)]


class CiProfileInfoV2(ctypes.Structure):
    _fields_ = [("version", u32)] + CiProfileInfo._fields_ + [("name", ctypes.c_char * 96)]


class ComputeInstanceInfo(ctypes.Structure):
    _fields_ = [("device", dev_t), ("gpuInstance", dev_t), ("id", u32), ("profileId", u32),
                ("placement", Placement)]


class DeviceAttributes(ctypes.Structure):
    _fields_ = [("multiprocessorCount", u32), ("sharedCopyEngineCount", u32), ("sharedDecoderCount", u32),
                ("sharedEncoderCount", u32), ("sharedJpegCount", u32), ("sharedOfaCount", u32),
                ("gpuInstanceSliceCount", u32), ("computeInstanceSliceCount", u32), ("memorySizeMB", u64)]


class Memory(ctypes.Structure):
    _fields_ = [("total", u64), ("free", u64), ("used", u64)]


def version(struct, v):
    return ctypes.sizeof(struct) | (v << 24)


nvml = ctypes.CDLL(str(FAKEGPU / "libnvidia-ml.so.1"))
assert nvml.nvmlInit_v2() == SUCCESS


def _info(struct):
    return {f: (getattr(struct, f).decode() if isinstance(getattr(struct, f), bytes) else getattr(struct, f))
            for f, _ in struct._fields_ if f not in ("version", "device", "gpuInstance", "placement")}


def count():
    n = u32()
    nvml.nvmlDeviceGetCount_v2(ctypes.byref(n))
    return n.value


def dev(i):
    h = dev_t()
    assert nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == SUCCESS
    return h


def mig_mode(d):
    cur, pend = u32(9), u32(9)
    ret = nvml.nvmlDeviceGetMigMode(d, ctypes.byref(cur), ctypes.byref(pend))
    return ret, cur.value, pend.value


def set_mig(d, mode):
    act = u32(0xdead)
    ret = nvml.nvmlDeviceSetMigMode(d, u32(mode), ctypes.byref(act))
    return ret, act.value


def profile(d, index):
    i = ProfileInfo()
    ret = nvml.nvmlDeviceGetGpuInstanceProfileInfo(d, u32(index), ctypes.byref(i))
    return ret, _info(i)


def profile_v(d, index, v=2):
    i = (ProfileInfoV2 if v == 2 else ProfileInfoV3)()
    i.version = version(type(i), v)
    ret = nvml.nvmlDeviceGetGpuInstanceProfileInfoV(d, u32(index), ctypes.byref(i))
    return ret, _info(i)


def profile_by_id(d, profile_id):
    i = ProfileInfoV2()
    i.version = version(ProfileInfoV2, 2)
    ret = nvml.nvmlDeviceGetGpuInstanceProfileInfoByIdV(d, u32(profile_id), ctypes.byref(i))
    return ret, _info(i)


def placements(d, profile_id):
    """go-nvml's two-call protocol: the count first, then the list."""
    n = u32(0)
    ret = nvml.nvmlDeviceGetGpuInstancePossiblePlacements_v2(d, u32(profile_id), None, ctypes.byref(n))
    if ret != SUCCESS:
        return ret, []
    buf = (Placement * max(n.value, 1))()
    ret = nvml.nvmlDeviceGetGpuInstancePossiblePlacements_v2(d, u32(profile_id), buf, ctypes.byref(n))
    return ret, [(p.start, p.size) for p in buf[:n.value]]


def remaining(d, profile_id):
    n = u32(99)
    ret = nvml.nvmlDeviceGetGpuInstanceRemainingCapacity(d, u32(profile_id), ctypes.byref(n))
    return ret, n.value


def create_gi(d, profile_id, placement=None):
    gi = dev_t()
    if placement is None:
        ret = nvml.nvmlDeviceCreateGpuInstance(d, u32(profile_id), ctypes.byref(gi))
    else:
        p = Placement(*placement)
        ret = nvml.nvmlDeviceCreateGpuInstanceWithPlacement(d, u32(profile_id), ctypes.byref(p), ctypes.byref(gi))
    return ret, gi


def gi_info(gi):
    i = GpuInstanceInfo()
    ret = nvml.nvmlGpuInstanceGetInfo(gi, ctypes.byref(i))
    return ret, {"id": i.id, "profileId": i.profileId, "placement": (i.placement.start, i.placement.size),
                 "device": i.device}


def gis(d, profile_id, size=16):
    buf = (dev_t * size)()
    n = u32(size)
    ret = nvml.nvmlDeviceGetGpuInstances(d, u32(profile_id), buf, ctypes.byref(n))
    return ret, [dev_t(h) for h in buf[:n.value]] if ret == SUCCESS else []


def gi_by_id(d, gi_id):
    gi = dev_t()
    ret = nvml.nvmlDeviceGetGpuInstanceById(d, u32(gi_id), ctypes.byref(gi))
    return ret, gi


def destroy_gi(gi):
    return nvml.nvmlGpuInstanceDestroy(gi)


def ci_profile(gi, index, engine=0):
    i = CiProfileInfo()
    ret = nvml.nvmlGpuInstanceGetComputeInstanceProfileInfo(gi, u32(index), u32(engine), ctypes.byref(i))
    return ret, _info(i)


def ci_profile_v(gi, index, engine=0):
    i = CiProfileInfoV2()
    i.version = version(CiProfileInfoV2, 2)
    ret = nvml.nvmlGpuInstanceGetComputeInstanceProfileInfoV(gi, u32(index), u32(engine), ctypes.byref(i))
    return ret, _info(i)


def create_ci(gi, profile_id, placement=None):
    ci = dev_t()
    if placement is None:
        ret = nvml.nvmlGpuInstanceCreateComputeInstance(gi, u32(profile_id), ctypes.byref(ci))
    else:
        p = Placement(*placement)
        ret = nvml.nvmlGpuInstanceCreateComputeInstanceWithPlacement(gi, u32(profile_id), ctypes.byref(p),
                                                                     ctypes.byref(ci))
    return ret, ci


def ci_placements(gi, profile_id):
    """GetComputeInstancePossiblePlacements: the count first, then the list."""
    n = u32(0)
    ret = nvml.nvmlGpuInstanceGetComputeInstancePossiblePlacements(gi, u32(profile_id), None, ctypes.byref(n))
    if ret != SUCCESS:
        return ret, []
    buf = (Placement * max(n.value, 1))()
    ret = nvml.nvmlGpuInstanceGetComputeInstancePossiblePlacements(gi, u32(profile_id), buf, ctypes.byref(n))
    return ret, [(p.start, p.size) for p in buf[:n.value]]


def ci_info(ci):
    i = ComputeInstanceInfo()
    ret = nvml.nvmlComputeInstanceGetInfo_v2(ci, ctypes.byref(i))
    return ret, {"id": i.id, "profileId": i.profileId, "placement": (i.placement.start, i.placement.size),
                 "device": i.device, "gpuInstance": i.gpuInstance}


def cis(gi, profile_id, size=8):
    buf = (dev_t * size)()
    n = u32(size)
    ret = nvml.nvmlGpuInstanceGetComputeInstances(gi, u32(profile_id), buf, ctypes.byref(n))
    return ret, [dev_t(h) for h in buf[:n.value]] if ret == SUCCESS else []


def ci_by_id(gi, ci_id):
    ci = dev_t()
    ret = nvml.nvmlGpuInstanceGetComputeInstanceById(gi, u32(ci_id), ctypes.byref(ci))
    return ret, ci


def destroy_ci(ci):
    return nvml.nvmlComputeInstanceDestroy(ci)


def max_mig(d):
    n = u32(99)
    ret = nvml.nvmlDeviceGetMaxMigDeviceCount(d, ctypes.byref(n))
    return ret, n.value


def mig_handle(d, index):
    h = dev_t()
    ret = nvml.nvmlDeviceGetMigDeviceHandleByIndex(d, u32(index), ctypes.byref(h))
    return ret, h


def is_mig(h):
    v = u32(9)
    ret = nvml.nvmlDeviceIsMigDeviceHandle(h, ctypes.byref(v))
    return ret, v.value


def parent(h):
    p = dev_t()
    ret = nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(h, ctypes.byref(p))
    return ret, p


def gi_id(h):
    v = u32(99)
    return nvml.nvmlDeviceGetGpuInstanceId(h, ctypes.byref(v)), v.value


def ci_id(h):
    v = u32(99)
    return nvml.nvmlDeviceGetComputeInstanceId(h, ctypes.byref(v)), v.value


def uuid(h):
    b = ctypes.create_string_buffer(96)
    return nvml.nvmlDeviceGetUUID(h, b, 96), b.value.decode()


def name(h):
    b = ctypes.create_string_buffer(96)
    return nvml.nvmlDeviceGetName(h, b, 96), b.value.decode()


def memory(h):
    m = Memory()
    return nvml.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)), m.total


def attributes(h):
    a = DeviceAttributes()
    return nvml.nvmlDeviceGetAttributes_v2(h, ctypes.byref(a)), _info(a)


def index(h):
    v = u32(99)
    return nvml.nvmlDeviceGetIndex(h, ctypes.byref(v)), v.value


def minor(h):
    v = u32(99)
    return nvml.nvmlDeviceGetMinorNumber(h, ctypes.byref(v)), v.value


def by_uuid(u):
    h = dev_t()
    return nvml.nvmlDeviceGetHandleByUUID(u.encode(), ctypes.byref(h)), h


def mig_devices(d):
    """MIG handles of a GPU as go-nvlib's VisitMigDevices walks them."""
    ret, n = max_mig(d)
    out = []
    for i in range(n if ret == SUCCESS else 0):
        r, h = mig_handle(d, i)
        if r == SUCCESS:
            out.append(h)
    return out


def setup_all(d, profile_id, ci_profile_index=None):
    """Fill a GPU with GPU instances of one profile, a full compute instance
    in each (nvidia-smi mig -cgi <id>,... -C); returns the created GIs."""
    created = []
    while True:
        ret, gi = create_gi(d, profile_id)
        if ret != SUCCESS:
            return created, ret
        if ci_profile_index is not None:
            r, info = ci_profile(gi, ci_profile_index)
            assert r == SUCCESS, r
            r, _ = create_ci(gi, info["id"])
            assert r == SUCCESS, r
        created.append(gi)


if os.environ.get("MIG_NVML_SELFTEST"):
    print(count())
