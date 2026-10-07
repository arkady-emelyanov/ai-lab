"""The driver, CUDA and VBIOS versions the fake driver reports are settings
(fakegpu_driver_version, fakegpu_cuda_version, fakegpu_vbios_version), seen
the same through nvidia-smi, NVML and the CUDA driver API."""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"


def run(tmp_path, code, **versions):
    (tmp_path / "dev").mkdir(exist_ok=True)
    (tmp_path / "dev" / "nvidia0").touch()
    env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(tmp_path / "none.conf"), FAKEGPU_COUNT="1",
               FAKEGPU_DEV_DIR=str(tmp_path / "dev"), FAKEGPU_STATE_PATH=str(tmp_path / "occ"),
               **{f"FAKEGPU_{k.upper()}": v for k, v in versions.items()})
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout


SMI = f"import runpy, sys; sys.argv = ['nvidia-smi', '-q']; runpy.run_path({str(FAKEGPU / 'nvidia-smi')!r}, run_name='__main__')"
APIS = f"""
import ctypes
cu = ctypes.CDLL({str(FAKEGPU / 'libcuda.so.1')!r}); nvml = ctypes.CDLL({str(FAKEGPU / 'libnvidia-ml.so.1')!r})
v = ctypes.c_int(); cu.cuDriverGetVersion(ctypes.byref(v)); n = ctypes.c_int(); nvml.nvmlSystemGetCudaDriverVersion(ctypes.byref(n))
s = ctypes.create_string_buffer(80); nvml.nvmlSystemGetNVMLVersion(s, 80)
print(v.value, n.value, s.value.decode())
"""


def field(out, name):
    return next(line.split(":", 1)[1].strip() for line in out.splitlines() if line.strip().startswith(name))


def test_defaults(tmp_path):
    out = run(tmp_path, SMI)
    assert (field(out, "Driver Version"), field(out, "CUDA Version"), field(out, "VBIOS Version")) == \
        ("580.95.05", "13.0", "97.00.82.00.0F")
    assert run(tmp_path, APIS).split() == ["13000", "13000", "13.580.95.05"]


def test_configured_versions_everywhere(tmp_path):
    versions = dict(driver_version="575.57.08", cuda_version="12.9", vbios_version="96.00.AF.00.01")
    out = run(tmp_path, SMI, **versions)
    assert (field(out, "Driver Version"), field(out, "CUDA Version"), field(out, "VBIOS Version")) == \
        ("575.57.08", "12.9", "96.00.AF.00.01")
    assert run(tmp_path, APIS, **versions).split() == ["12090", "12090", "12.575.57.08"]
