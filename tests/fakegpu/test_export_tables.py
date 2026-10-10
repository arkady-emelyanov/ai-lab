"""The driver's undocumented export tables ("dark API") that CUDA runtimes
fetch at initialisation: without one they need, every runtime call fails."""
import ctypes
from pathlib import Path

import pytest

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"


@pytest.fixture(scope="module")
def cuda():
    return ctypes.CDLL(str(FAKEGPU / "libcuda.so.1"))


def table(cuda, uuid):
    t = ctypes.c_void_p()
    rc = cuda.cuGetExportTable(ctypes.byref(t), (ctypes.c_ubyte * 16)(*bytes.fromhex(uuid)))
    return rc, t.value


def test_cuda_13_4_runtime_table(cuda):
    # Requested by CUDA 13.4 runtimes (cuda-bindings 13.4's static cudart):
    # a size header and 94 slots, as driver 595.99.02 serves it.
    rc, t = table(cuda, "21318c60971432488ca641ff7324c8f2")
    assert rc == 0 and t
    slots = (ctypes.c_uint64 * 95).from_address(t)
    assert slots[0] == 95 * 8
    assert all(slots[1:])


def test_unknown_table_is_refused(cuda):
    assert table(cuda, "00" * 16) == (1, None)
