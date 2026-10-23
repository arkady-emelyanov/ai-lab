"""MIG through CUDA (R570 and later, as the MIG user guide describes it):
a MIG-enabled GPU shows as its MIG devices, one per GPU instance, not as
itself, and none while it has no compute instance; CUDA_VISIBLE_DEVICES
takes MIG- UUIDs; a MIG device has its GPU instance's memory and its compute
instance's SMs, as a container given a MIG slice sees it."""
import pytest

from mig_tray import Tray

DEVICES = """
n = ctypes.c_int(); cu.cuDeviceGetCount(ctypes.byref(n))
out = []
for i in range(n.value):
    name = ctypes.create_string_buffer(96); cu.cuDeviceGetName(name, 96, i)
    mem = ctypes.c_size_t(); cu.cuDeviceTotalMem_v2(ctypes.byref(mem), i)
    sms = ctypes.c_int(); cu.cuDeviceGetAttribute(ctypes.byref(sms), 16, i)
    u = (ctypes.c_ubyte * 16)(); cu.cuDeviceGetUuid_v2(u, i)
    h = bytes(u).hex()
    out.append([name.value.decode(), mem.value >> 20, sms.value, f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"])
"""


@pytest.fixture
def tray(tmp_path):
    return Tray(tmp_path)


def mig_uuids(tray, gpu=0):
    return tray.nvml(f"d = m.dev({gpu}); out = [m.uuid(h)[1] for h in m.mig_devices(d)]")


def test_mig_gpu_without_instances_shows_no_device(tmp_path):
    """Without GPU and compute instances CUDA cannot run on a MIG-enabled
    GPU; other GPUs still show."""
    tray = Tray(tmp_path, count=2)
    tray.enable_mig(0)
    out = tray.cuda(DEVICES)
    assert len(out) == 1 and out[0][0] == "NVIDIA GB200" and out[0][1] == 189471


def test_one_device_per_gpu_instance(tmp_path):
    """Without CUDA_VISIBLE_DEVICES: a device per GPU instance, then the GPUs
    without MIG; each with its MIG UUID, name, memory and SMs."""
    tray = Tray(tmp_path, count=2)
    tray.smi("-i", "0", "-mig", "1")
    tray.smi("mig", "-i", "0", "-cgi", "9,19", "-C")
    uuids = mig_uuids(tray)
    out = tray.cuda(DEVICES)
    assert [d[:3] for d in out] == [["NVIDIA GB200 MIG 3g.93gb", 93696, 72], ["NVIDIA GB200 MIG 1g.23gb", 22528, 18],
                                    ["NVIDIA GB200", 189471, 148]]
    assert ["MIG-" + d[3] for d in out[:2]] == uuids


def test_visible_devices_by_mig_uuid(tray):
    """CUDA_VISIBLE_DEVICES=MIG-<uuid>: that MIG device only (a container
    given a MIG slice)."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "19,19", "-C")
    uuids = mig_uuids(tray)
    out = tray.cuda(DEVICES, env={"CUDA_VISIBLE_DEVICES": uuids[1]})
    assert len(out) == 1 and "MIG-" + out[0][3] == uuids[1] and out[0][1] == 22528


def test_partial_compute_instance_has_its_sms(tray):
    """A container sees its compute instance's SMs: on 1g.47gb with CI 0, 18
    (dra-driver-nvidia-gpu#1454), with the GPU instance's memory."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "15")
    tray.smi("mig", "-cci", "0")  # in every GPU instance: the one 1g.47gb
    out = tray.cuda(DEVICES)
    assert out[0][:3] == ["NVIDIA GB200 MIG 1g.47gb", 46592, 18]


def test_memory_is_the_slices(tray):
    """Allocations beyond the MIG device's memory fail with out of memory."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "19", "-C")
    out = tray.cuda("""
c = ctypes.c_void_p(); cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), 0); cu.cuCtxSetCurrent(c)
p = ctypes.c_ulonglong()
fits = cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(20 << 30))
beyond = cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(4 << 30))
free, total = ctypes.c_size_t(), ctypes.c_size_t(); cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
out = [fits, beyond, total.value >> 20]
""")
    assert out == [0, 2, 22528]


def test_slices_are_slower(tray):
    """Work on a MIG device takes longer, in proportion to its SMs."""
    timing = """
import time
c = ctypes.c_void_p(); cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), 0); cu.cuCtxSetCurrent(c)
t = time.monotonic()
for _ in range(100):
    cu.cuLaunchKernel(None, 1 << 18, 1, 1, 256, 1, 1, 0, None, None, None)
cu.cuCtxSynchronize()
out = time.monotonic() - t
"""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "0", "-C")  # 7g: the whole GPU's SMs
    whole = tray.cuda(timing)
    tray.smi("mig", "-dci")
    tray.smi("mig", "-dgi")
    tray.smi("mig", "-cgi", "19", "-C")
    slice_ = tray.cuda(timing)
    assert slice_ > 4 * whole


def test_no_fabric_memory_on_a_mig_device(tray):
    """MIG devices have no NVLink peers and no fabric handles (GB200 lists
    every profile without P2P)."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "19", "-C")
    out = tray.cuda("v = ctypes.c_int(9); cu.cuDeviceGetAttribute(ctypes.byref(v), 128, 0); out = v.value",
                    env={"FAKEGPU_FABRIC_DIR": str(tray.dir / "fabric")})
    assert out == 0
