"""The fake libraries load in containers built on older distributions, as
NVIDIA's do: no symbol needs a glibc newer than 2.33 (glibc_compat.h), so
CUDA and NVIDIA images on Ubuntu 22.04 (glibc 2.35) work, e.g. NVIDIA's
DCGM exporter on the fake NVML; the fake driver's programs (nvidia-smi and
the container toolkit's entry points) need no newer than 2.34."""
import re
import subprocess
from pathlib import Path

import pytest

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"
LIBS = ["libcuda.so.1", "libnvidia-ml.so.1", "libcublas.so.13", "libcublasLt.so.13", "libcublas.so.12",
        "libcublasLt.so.12", "libnccl.so.2", "libcudnn.so.9"]
NEWEST = (2, 33)
# Programs that also run in containers (nvidia-smi through the CDI spec and
# nvidia-container-cli), and the toolkit's entry points: every program built
# on current glibc starts through __libc_start_main@GLIBC_2.34, so 2.34 (Ubuntu
# 22.04, Debian 12, RHEL 9) is the oldest that runs them.
PROGRAMS = ["nvidia-smi", "nvidia-container-cli", "nvidia-container-runtime", "nvidia-cuda-mps-control"]
NEWEST_PROGRAM = (2, 34)


@pytest.fixture(scope="module", autouse=True)
def build():
    subprocess.run(["make", "-s", "-C", str(FAKEGPU)], check=True)


@pytest.mark.parametrize("lib", LIBS)
def test_needs_no_newer_glibc(lib):
    symbols = subprocess.run(["objdump", "-T", str(FAKEGPU / lib)], capture_output=True, text=True,
                             check=True).stdout
    too_new = sorted({f"{name} (GLIBC_{major}.{minor})"
                      for major, minor, name in re.findall(r"\(?GLIBC_(\d+)\.(\d+)\)?\s+(\S+)$", symbols, re.M)
                      if (int(major), int(minor)) > NEWEST})
    assert not too_new, f"{lib} needs a newer glibc for: {', '.join(too_new)}"


@pytest.mark.parametrize("program", PROGRAMS)
def test_programs_need_no_newer_glibc(program):
    symbols = subprocess.run(["objdump", "-T", str(FAKEGPU / program)], capture_output=True, text=True,
                             check=True).stdout
    too_new = sorted({f"{name} (GLIBC_{major}.{minor})"
                      for major, minor, name in re.findall(r"\(?GLIBC_(\d+)\.(\d+)\)?\s+(\S+)$", symbols, re.M)
                      if (int(major), int(minor)) > NEWEST_PROGRAM})
    assert not too_new, f"{program} needs a newer glibc for: {', '.join(too_new)}"
