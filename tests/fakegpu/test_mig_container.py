"""MIG in a container given MIG devices (the lab's device plugin gives a pod
a slice as NVIDIA's does: its parent GPU, and NVIDIA_VISIBLE_DEVICES=MIG-
<uuid>): NVML, nvidia-smi and CUDA show that slice only, as the real driver
shows a container the MIG devices whose capability nodes it has. And at a
driver load (a tray's boot) MIG mode stays but the instances are gone."""
import re

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


def test_driver_load_keeps_mig_mode_not_instances(tray):
    tray.smi("--fakegpu-driver-load")
    out = tray.nvml("d = m.dev(0); out = [m.mig_mode(d), m.mig_handle(d, 0)[0], m.remaining(d, 19)]")
    assert out == [[0, 1, 1], 6, [0, 7]]
