"""A GPU fraction's memory limit (NVIDIA_GPU_MEMORY_LIMIT, MiB), which NVIDIA's
GPU fractioning gives a container sharing a GPU and the driver enforces: each
process allocates up to it and gets out of memory beyond it, cuMemGetInfo
reports it as the total, and processes without a limit keep the whole GPU."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"
OUT_OF_MEMORY = 2
MIB = 1 << 20

# Allocates the given sizes (MiB) in turn, then prints cuMemGetInfo and each
# allocation's result code.
PROG = f"""
import ctypes, sys
cu = ctypes.CDLL({str(FAKEGPU / 'libcuda.so.1')!r})
cu.cuInit(0); dev = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p(); cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)
rcs = []
for mib in map(int, sys.argv[1:]):
    p = ctypes.c_uint64()
    rcs.append(cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(mib << 20)))
free, total = ctypes.c_size_t(), ctypes.c_size_t()
cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
print(free.value, total.value, *rcs)
"""


@pytest.fixture(scope="module", autouse=True)
def build():
    subprocess.run(["make", "-s", "-C", str(FAKEGPU)], check=True)


@pytest.fixture
def gpu(tmp_path):
    """gpu(limit_mib, *alloc_mib) -> (free, total, [result codes]) in one process on a 1 GiB GPU."""
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "nvidia0").touch()

    def run(limit, *allocs):
        env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(tmp_path / "none.conf"),
                   FAKEGPU_COUNT="1", FAKEGPU_DEV_DIR=str(tmp_path / "dev"), FAKEGPU_STATE_PATH=str(tmp_path / "occ"),
                   FAKEGPU_MEM_MB="1024")
        env.pop("NVIDIA_GPU_MEMORY_LIMIT", None)
        if limit is not None:
            env["NVIDIA_GPU_MEMORY_LIMIT"] = str(limit)
        out = subprocess.run([sys.executable, "-c", PROG, *map(str, allocs)], env=env, capture_output=True,
                             text=True, check=True, timeout=30).stdout.split()
        return int(out[0]), int(out[1]), [int(rc) for rc in out[2:]]

    return run


def test_allocations_stop_at_the_limit(gpu):
    free, total, rcs = gpu(256, 100, 100, 100)
    assert rcs == [0, 0, OUT_OF_MEMORY]
    assert total == 256 * MIB
    assert free == 56 * MIB


def test_without_a_limit_the_whole_gpu(gpu):
    free, total, rcs = gpu(None, 900)
    assert rcs == [0]
    assert total == 1024 * MIB


def test_a_limit_above_the_gpu_is_the_gpu(gpu):
    _, total, rcs = gpu(4096, 900, 200)
    assert total == 1024 * MIB
    assert rcs == [0, OUT_OF_MEMORY]
