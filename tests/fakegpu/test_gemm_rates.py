"""GEMMs take 2·m·n·k FLOPs at the rate of their precision, as NVIDIA
publishes for GB200 (dense): BF16/FP16 2.5 PFLOP/s, TF32 1.25, FP8 5,
FP32 without tensor cores ~60 TFLOP/s, FP64 40 TFLOP/s."""
import os
import subprocess
import sys
from pathlib import Path

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"

# cudaDataType and cublasComputeType values (cuda/library_types.h, cublas_api.h).
R_32F, R_64F, R_16BF, R_8F_E4M3 = 0, 1, 14, 28
COMPUTE_32F, COMPUTE_64F, COMPUTE_32F_FAST_TF32 = 68, 70, 77
TF32_TENSOR_OP_MATH = 3

PROBE = r"""
import ctypes, sys, time
cuda = ctypes.CDLL("libcuda.so.1")
blas = ctypes.CDLL("libcublas.so.13")
cuda.cuInit(0)
dev, ctx = ctypes.c_int(), ctypes.c_void_p()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
h = ctypes.c_void_p()
blas.cublasCreate_v2(ctypes.byref(h))
atype, compute, mode, n = (int(a) for a in sys.argv[1:5])
blas.cublasSetMathMode(h, mode)
one = ctypes.c_double(1)
t = time.monotonic()
blas.cublasGemmEx(h, 0, 0, n, n, n, ctypes.byref(one), None, atype, n, None, atype, n, ctypes.byref(one), None, atype, n, compute, -1)
cuda.cuCtxSynchronize()
print(time.monotonic() - t)
"""


def gemm_seconds(tmp_path, atype, compute, mode=0, n=16384):
    (tmp_path / "dev").mkdir(exist_ok=True)
    (tmp_path / "dev" / "nvidia0").touch()
    env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(tmp_path / "none.conf"), FAKEGPU_COUNT="1",
               FAKEGPU_DEV_DIR=str(tmp_path / "dev"), FAKEGPU_STATE_PATH=str(tmp_path / "occ"))
    out = subprocess.run([sys.executable, "-c", PROBE, str(atype), str(compute), str(mode), str(n)],
                         env=env, capture_output=True, text=True, check=True).stdout
    return float(out)


def rate(tmp_path, *args, n=16384):
    """Measured TFLOP/s of one n x n x n GEMM."""
    return 2 * n ** 3 / gemm_seconds(tmp_path, *args, n=n) / 1e12


def test_gemm_rate_follows_precision(tmp_path):
    near = lambda measured, want: want * 0.8 < measured < want * 1.05  # launch and wake-up overheads
    assert near(rate(tmp_path, R_16BF, COMPUTE_32F), 2500)
    assert near(rate(tmp_path, R_32F, COMPUTE_32F_FAST_TF32), 1250)
    assert near(rate(tmp_path, R_32F, COMPUTE_32F, TF32_TENSOR_OP_MATH), 1250)  # PyTorch's allow_tf32
    assert near(rate(tmp_path, R_8F_E4M3, COMPUTE_32F), 5000)
    assert near(rate(tmp_path, R_32F, COMPUTE_32F, n=4096), 60)
    assert near(rate(tmp_path, R_64F, COMPUTE_64F, n=4096), 40)
