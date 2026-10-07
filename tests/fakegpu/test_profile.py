"""The GPU profile (fakegpu_* settings) drives what the fake driver reports
and how long work takes: NVLink count and speed, power figures, NCCL cost."""
import os
import subprocess
import sys
from pathlib import Path

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"
SMI = FAKEGPU / "nvidia-smi"


def smi(tmp_path, *args, **profile):
    (tmp_path / "dev").mkdir(exist_ok=True)
    (tmp_path / "dev" / "nvidia0").touch()
    env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(tmp_path / "none.conf"), FAKEGPU_COUNT="1",
               FAKEGPU_DEV_DIR=str(tmp_path / "dev"), FAKEGPU_STATE_PATH=str(tmp_path / "occ"),
               **{f"FAKEGPU_{k.upper()}": str(v) for k, v in profile.items()})
    code = f"import runpy, sys; sys.argv = ['nvidia-smi', *{list(args)!r}]; runpy.run_path({str(SMI)!r}, run_name='__main__')"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout


def test_default_profile(tmp_path):
    links = [line for line in smi(tmp_path, "nvlink", "-s").splitlines() if "Link " in line]
    assert len(links) == 18 and all(line.endswith("50 GB/s") for line in links)
    assert smi(tmp_path, "--query-gpu=power.limit", "--format=csv,noheader").strip() == "1200.00 W"


def test_configured_profile(tmp_path):
    profile = dict(nvlinks=12, nvlink_link_gbs=25, power_limit_w=700, idle_power_w=90, max_power_w=650)
    links = [line for line in smi(tmp_path, "nvlink", "-s", **profile).splitlines() if "Link " in line]
    assert len(links) == 12 and all(line.endswith("25 GB/s") for line in links)
    assert smi(tmp_path, "--query-gpu=power.limit", "--format=csv,noheader", **profile).strip() == "700.00 W"
    draw = float(smi(tmp_path, "--query-gpu=power.draw", "--format=csv,noheader,nounits", **profile))
    assert 90 * 0.97 < draw < 90 * 1.03  # idle, within the +-2 % jitter
