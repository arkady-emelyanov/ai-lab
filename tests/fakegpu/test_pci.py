"""The GPUs' PCI identity is a GB200 compute tray's (lspci in NVIDIA/mig-parted#227
and NVIDIA/kubevirt-gpu-device-plugin#154): device 10de:2941, subsystem
10de:2046, each GPU in a PCI domain of its own on bus 1, the same through
NVML, the CUDA driver API and nvidia-smi."""
import os
import subprocess
import sys
from pathlib import Path

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"
ADDRESSES = ["00000008:01:00.0", "00000009:01:00.0", "00000018:01:00.0", "00000019:01:00.0"]


def run(tmp_path, *args):
    (tmp_path / "dev").mkdir(exist_ok=True)
    for i in range(4):
        (tmp_path / "dev" / f"nvidia{i}").touch()
    env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(tmp_path / "none.conf"), FAKEGPU_COUNT="4",
               FAKEGPU_DEV_DIR=str(tmp_path / "dev"), FAKEGPU_STATE_PATH=str(tmp_path / "occ"))
    return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True, check=True).stdout


APIS = f"""
import ctypes
class Pci(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]
nvml = ctypes.CDLL({str(FAKEGPU / 'libnvidia-ml.so.1')!r}); cu = ctypes.CDLL({str(FAKEGPU / 'libcuda.so.1')!r})
nvml.nvmlInit_v2(); cu.cuInit(0)
for i in range(4):
    h = ctypes.c_void_p(); nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
    p = Pci(); nvml.nvmlDeviceGetPciInfo_v3(h, ctypes.byref(p))
    by_bus = ctypes.c_void_p(); nvml.nvmlDeviceGetHandleByPciBusId_v2(p.busId, ctypes.byref(by_bus))
    b = ctypes.create_string_buffer(32); cu.cuDeviceGetPCIBusId(b, 32, i)
    dom, bus, dev = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(-1)
    cu.cuDeviceGetAttribute(ctypes.byref(dom), 50, i); cu.cuDeviceGetAttribute(ctypes.byref(bus), 33, i)
    cu.cuDeviceGetByPCIBusId(ctypes.byref(dev), b.value)
    print(p.busId.decode(), p.busIdLegacy.decode(), hex(p.domain), p.bus, hex(p.pciDeviceId), hex(p.pciSubSystemId),
          by_bus.value == h.value, b.value.decode(), hex(dom.value), bus.value, dev.value)
"""


def test_nvml_and_cuda_agree(tmp_path):
    rows = [line.split() for line in run(tmp_path, "-c", APIS).splitlines()]
    assert [r[0] for r in rows] == ADDRESSES
    assert [r[1] for r in rows] == [a[4:] for a in ADDRESSES]                    # legacy form, 4-digit domain
    assert [r[2] for r in rows] == ["0x8", "0x9", "0x18", "0x19"]
    assert {(r[3], r[4], r[5], r[6]) for r in rows} == {("1", "0x294110de", "0x204610de", "True")}
    assert [r[7] for r in rows] == ADDRESSES                                     # cuDeviceGetPCIBusId
    assert [(r[8], r[9]) for r in rows] == [("0x8", "1"), ("0x9", "1"), ("0x18", "1"), ("0x19", "1")]
    assert [r[10] for r in rows] == ["0", "1", "2", "3"]                         # cuDeviceGetByPCIBusId


def test_nvidia_smi(tmp_path):
    out = run(tmp_path, str(FAKEGPU / "nvidia-smi"), "--query-gpu=pci.bus_id", "--format=csv,noheader")
    assert out.split() == ADDRESSES
