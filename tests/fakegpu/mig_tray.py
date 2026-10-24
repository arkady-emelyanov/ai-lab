"""An emulated tray for the MIG tests: its own configuration, device nodes and
shared GPU state (what a tray's processes share), and helpers that run NVML
(mig_nvml.py), CUDA and nvidia-smi there in fresh processes, as separate
programs on a tray do. Changing MIG state needs root, as with the real
driver: NVML and nvidia-smi run as root of a user namespace (unshare -Ur)
unless root=False; CUDA runs as the user, as a job or pod does."""
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKEGPU = HERE.parents[1] / "fakegpu"

# GB200 GPU instance profiles: NVML profile index -> what NVML reports
# (nvidia-smi mig -lgip on GB200: NVIDIA/mig-parted#227; profile names as
# go-nvlib derives them and mig-parted's GB200 configurations use).
PROFILES = {
    #   name          id  slices inst  SMs   MiB    CE DEC ENC JPG OFA
    0: ("1g.23gb",    19, 1, 7,  18,  22528, 2, 1, 0, 1, 0),
    7: ("1g.23gb+me", 20, 1, 1,  18,  22528, 2, 1, 0, 1, 1),
    9: ("1g.47gb",    15, 1, 4,  32,  46592, 2, 1, 0, 1, 0),
    1: ("2g.47gb",    14, 2, 3,  38,  46592, 4, 2, 0, 2, 0),
    2: ("3g.93gb",     9, 3, 2,  72,  93696, 6, 3, 0, 3, 0),
    3: ("4g.93gb",     5, 4, 1,  76,  93696, 8, 4, 0, 4, 1),
    4: ("7g.186gb",    0, 7, 1, 152, 188416, 16, 7, 0, 7, 2),
}
BY_NAME = {v[0]: (k, *v[1:]) for k, v in PROFILES.items()}
UNSUPPORTED = [5, 6, 8] + list(range(10, 18))  # 8_SLICE, 6_SLICE, 2_SLICE_REV1, GFX, NO_ME, ALL_ME
PROFILE_COUNT = 18

# Placements (start, size) on the GPU's 8 memory slices, by profile id.
PLACEMENTS = {
    19: [(s, 1) for s in range(7)],
    20: [(s, 1) for s in range(7)],
    15: [(0, 2), (2, 2), (4, 2), (6, 2)],
    14: [(0, 2), (2, 2), (4, 2)],
    9: [(0, 4), (4, 4)],
    5: [(0, 4)],
    0: [(0, 8)],
}

# Compute instance profiles in each GPU instance profile: CI profile index ->
# instances; the full one (C == G) has the GPU instance's SMs. On 1g.47gb
# CI 7 (1_SLICE_REV1, 32 SMs) is the full one and CI 0 has 18 SMs
# (dra-driver-nvidia-gpu#1454).
CI_PROFILES = {
    "1g.23gb": {0: 1},
    "1g.23gb+me": {0: 1},
    "1g.47gb": {0: 1, 7: 1},
    "2g.47gb": {0: 2, 1: 1},
    "3g.93gb": {0: 3, 1: 1, 2: 1},
    "4g.93gb": {0: 4, 1: 2, 3: 1},
    "7g.186gb": {0: 7, 1: 3, 2: 2, 3: 1, 4: 1},
}
FULL_CI = {"1g.23gb": 0, "1g.23gb+me": 0, "1g.47gb": 7, "2g.47gb": 1, "3g.93gb": 2, "4g.93gb": 3, "7g.186gb": 4}
CI_SLICES = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 7: 1}
CI_PROFILE_COUNT = 9

GPU_MEMORY_MIB = 189471


def go_nvlib_name(slices_g, slices_c, memory_mb, suffix=""):
    """go-nvlib's MIG profile name (pkg/nvlib/device/mig_profile.go)."""
    import math
    total = GPU_MEMORY_MIB << 20
    gb = math.floor(math.ceil(memory_mb * (1 << 20) / total * 8) / 8 * math.ceil(total / (1 << 30)) + 0.5)  # Go's math.Round
    base = f"{slices_g}g.{gb}gb" if slices_c == slices_g else f"{slices_c}c.{slices_g}g.{gb}gb"
    return base + suffix


class Tray:
    def __init__(self, tmp_path, count=1, name="tray0"):
        self.dir = tmp_path / name
        (self.dir / "dev").mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (self.dir / "dev" / f"nvidia{i}").touch()
        self.env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), PYTHONPATH=str(HERE),
                        FAKEGPU_CONF=str(self.dir / "none.conf"), FAKEGPU_COUNT=str(count), FAKEGPU_HOST=name,
                        FAKEGPU_DEV_DIR=str(self.dir / "dev"), FAKEGPU_STATE_PATH=str(self.dir / "occ"),
                        FAKEGPU_SIDEBAND_DIR=str(self.dir / "sb"), FAKEGPU_IPC_DIR=str(self.dir / "ipc"),
                        FAKEGPU_FABRIC_DIR="")

    def py(self, code, env=None, check=True, root=False):
        cmd = [sys.executable, "-c", code]
        p = subprocess.run(["unshare", "-Ur", *cmd] if root else cmd, env=dict(self.env, **(env or {})),
                           capture_output=True, text=True, timeout=60)
        if check:
            assert p.returncode == 0, p.stderr
        return p

    def nvml(self, body, env=None, root=True):
        """Runs body with `m` (mig_nvml) and `json`; returns the JSON of `out`."""
        code = "import json\nimport mig_nvml as m\n" + body + "\nprint(json.dumps(out, default=str))"
        return json.loads(self.py(code, env, root=root).stdout.splitlines()[-1])

    def cuda(self, body, env=None):
        """Runs body with `cu` (libcuda) and `json`; returns the JSON of `out`."""
        code = ("import ctypes, json\n"
                f"cu = ctypes.CDLL({str(FAKEGPU / 'libcuda.so.1')!r})\n"
                "assert cu.cuInit(0) == 0\n" + body + "\nprint(json.dumps(out, default=str))")
        return json.loads(self.py(code, env).stdout.splitlines()[-1])

    def smi(self, *args, root=True, check=True, env=None):
        """nvidia-smi on the tray, as root of a user namespace (MIG changes,
        nvidia-smi -r as mig-parted calls it) unless root=False."""
        cmd = [sys.executable, str(FAKEGPU / "nvidia-smi"), *args]
        if root:
            cmd = ["unshare", "-Ur", *cmd]
        p = subprocess.run(cmd, env=dict(self.env, **(env or {})), capture_output=True, text=True, timeout=60)
        if check:
            assert p.returncode == 0, p.stdout + p.stderr
        return p

    @contextlib.contextmanager
    def holding(self, mib=1, env=None):
        """A CUDA process (as the user) holding mib MiB on its CUDA device 0
        while the block runs: a process on the GPU, or on its MIG device."""
        code = ("import ctypes, sys\n"
                "cu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0); c = ctypes.c_void_p()\n"
                "cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), 0); cu.cuCtxSetCurrent(c); p = ctypes.c_ulonglong()\n"
                f"r = cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t({mib} << 20))\n"
                "print('ready' if r == 0 else f'alloc {r}', flush=True); sys.stdin.read()")
        p = subprocess.Popen([sys.executable, "-c", code], env=dict(self.env, **(env or {})), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout.readline().strip() == "ready"
            yield p
        finally:
            p.stdin.close()
            p.wait(timeout=30)

    def enable_mig(self, gpu=0):
        out = self.nvml(f"out = m.set_mig(m.dev({gpu}), 1)")
        assert out == [0, 0], out
