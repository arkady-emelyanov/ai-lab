"""MIG in a container given MIG devices: NVML, nvidia-smi and CUDA show
those slices only, as the real driver shows a container the MIG devices
whose capability nodes it has. The fake driver takes them from the MIG-
UUIDs in NVIDIA_VISIBLE_DEVICES or, as the lab's device plugin hands them
out (NVIDIA's volume-mounts strategy), from the files named after them in
/var/run/nvidia-container-devices. And at a driver load (a tray's boot) MIG
mode stays but the instances are gone."""
import json
import re
import subprocess
import sys

import pytest

from mig_tray import Tray
from test_mig_cuda import DEVICES


@pytest.fixture
def tray(tmp_path):
    t = Tray(tmp_path)
    t.smi("-mig", "1")
    t.smi("mig", "-cgi", "3g.93gb,1g.23gb,1g.23gb", "-C")
    t.uuids = t.nvml("d = m.dev(0); out = [m.uuid(h)[1] for h in m.mig_devices(d)]")
    return t


def test_nvml_sees_the_containers_slice(tray):
    env = {"NVIDIA_VISIBLE_DEVICES": tray.uuids[2]}
    out = tray.nvml("d = m.dev(0); hs = m.mig_devices(d); "
                    "out = [[m.uuid(h)[1] for h in hs], m.mig_handle(d, 1)[0], m.by_uuid(%r)[0]]" % tray.uuids[0], env)
    assert out == [[tray.uuids[2]], 6, 6]  # only its own; others NOT_FOUND


def test_nvidia_smi_lists_the_containers_slice(tray):
    lines = tray.smi("-L", env={"NVIDIA_VISIBLE_DEVICES": tray.uuids[1]}).stdout.splitlines()
    assert len(lines) == 2 and lines[0].startswith("GPU 0: NVIDIA GB200")
    assert re.fullmatch(rf"  MIG 1g\.23gb     Device  0: \(UUID: {tray.uuids[1]}\)", lines[1])


def test_cuda_sees_the_containers_slice(tray):
    out = tray.cuda(DEVICES, env={"NVIDIA_VISIBLE_DEVICES": tray.uuids[0]})
    assert len(out) == 1 and out[0][:3] == ["NVIDIA GB200 MIG 3g.93gb", 93696, 72]


def test_several_slices_and_the_tray_sees_all(tray):
    env = {"NVIDIA_VISIBLE_DEVICES": ",".join(tray.uuids[1:])}
    assert len(tray.cuda(DEVICES, env=env)) == 2
    assert len(tray.cuda(DEVICES)) == 3
    assert len(tray.cuda(DEVICES, env={"NVIDIA_VISIBLE_DEVICES": "all"})) == 3


def test_slices_from_volume_mounts(tray):
    """NVIDIA's volume-mounts strategy: a file per device under
    /var/run/nvidia-container-devices (NVIDIA_VISIBLE_DEVICES names that
    directory): those slices only, several adding up. Run in a mount
    namespace with its own /run."""
    script = ("mount -t tmpfs none /run && mkdir /run/nvidia-container-devices && "
              'for u in "$@"; do touch /run/nvidia-container-devices/$u; done && '
              f"exec {sys.executable} -c \"$CODE\"")
    code = ("import ctypes, json\ncu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0)\n" + DEVICES +
            "\nprint(json.dumps(out))")
    p = subprocess.run(["unshare", "-Urm", "sh", "-c", script, "sh", tray.uuids[0], tray.uuids[2]],
                       env=dict(tray.env, CODE=code, NVIDIA_VISIBLE_DEVICES="/var/run/nvidia-container-devices"),
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert [d[0] for d in out] == ["NVIDIA GB200 MIG 3g.93gb", "NVIDIA GB200 MIG 1g.23gb"]
    assert "MIG-" + out[1][3] == tray.uuids[2]


def test_driver_load_keeps_mig_mode_not_instances(tray):
    tray.smi("--fakegpu-driver-load")
    out = tray.nvml("d = m.dev(0); out = [m.mig_mode(d), m.mig_handle(d, 0)[0], m.remaining(d, 19)]")
    assert out == [[0, 1, 1], 6, [0, 7]]
