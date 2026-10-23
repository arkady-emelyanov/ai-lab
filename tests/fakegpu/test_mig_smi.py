"""MIG through nvidia-smi, in the formats a GB200 prints them (real output in
mig-parted#227 and dra-driver-nvidia-gpu#907, #1454; the MIG user guide for
the rest), as NVIDIA's DRA driver test suite drives it (nvidia-smi -mig 1,
mig -lgip, mig -cgi <profile> -C, mig -dci, -dgi, -L) and mig-parted resets
GPUs (nvidia-smi -r -i <bus ids>)."""
import re

import pytest

from mig_tray import PLACEMENTS, PROFILES, Tray

BUS0 = "00000008:01:00.0"


@pytest.fixture
def tray(tmp_path):
    return Tray(tmp_path)


def test_enable_and_disable(tray):
    out = tray.smi("-i", "0", "-mig", "1").stdout
    assert out == f"Enabled MIG Mode for GPU {BUS0}\nAll done.\n"
    q = tray.smi("--query-gpu=index,mig.mode.current,mig.mode.pending", "--format=csv,noheader").stdout
    assert q == "0, Enabled, Enabled\n"
    assert tray.smi("-i", "0", "-mig", "0").stdout == f"Disabled MIG Mode for GPU {BUS0}\nAll done.\n"
    q = tray.smi("--query-gpu=mig.mode.current", "--format=csv,noheader").stdout
    assert q == "Disabled\n"


def test_disable_with_instances_is_refused(tray):
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "19")
    p = tray.smi("-mig", "0", check=False)
    assert p.returncode == 19
    assert (f"Unable to disable MIG Mode for GPU {BUS0}: In use by another client\n"
            "Terminating early due to previous errors.\n") == p.stdout + p.stderr


def test_list_gpu_instance_profiles(tray):
    """mig -lgip works with MIG off (the DRA test suite reads the 1g profile
    name from it before enabling MIG); one two-line row per profile."""
    out = tray.smi("mig", "-lgip", "-i", "0").stdout
    assert "GPU instance profiles:" in out
    for name, pid, _slices, inst, sms, mib, ce, dec, enc, jpeg, ofa in PROFILES.values():
        row = rf"\|\s+0\s+MIG {re.escape(name)}\s+{pid}\s+{inst}/{inst}\s+{mib / 1024:.2f}\s+No\s+{sms}\s+{dec}\s+{enc}\s+\|"
        assert re.search(row, out), (name, out)
        assert re.search(rf"\|\s+{ce}\s+{jpeg}\s+{ofa}\s+\|", out)
    assert re.search(r"1g\.[1-9]+gb", out).group(0) == "1g.23gb"


def test_list_placements(tray):
    out = tray.smi("mig", "-lgipp", "-i", "0").stdout
    for pid, places in PLACEMENTS.items():
        starts = ",".join(str(s) for s, _ in places)
        label = "Placement :" if len(places) == 1 else "Placements:"
        assert f"GPU  0 Profile ID {pid:2d} {label} {{{starts}}}:{places[0][1]}\n" in out, pid


def test_create_list_and_destroy(tray):
    """The DRA test suite's static MIG setup and teardown, line by line."""
    tray.smi("-i", "0", "-mig", "1")
    out = tray.smi("mig", "-cgi", "1g.23gb", "-C").stdout
    assert out == ("Successfully created GPU instance ID 13 on GPU  0 using profile MIG 1g.23gb (ID 19)\n"
                   "Successfully created compute instance ID  0 on GPU  0 GPU instance ID 13 using profile "
                   "MIG 1g.23gb (ID  0)\n")
    lgi = tray.smi("mig", "-lgi").stdout
    assert re.search(r"\|\s+0\s+MIG 1g\.23gb\s+19\s+13\s+6:1\s+\|", lgi), lgi
    lci = tray.smi("mig", "-lci").stdout
    assert re.search(r"\|\s+0\s+13\s+MIG 1g\.23gb\s+0\s+0\s+0:1\s+\|", lci), lci
    p = tray.smi("mig", "-dgi", check=False)
    assert p.returncode != 0
    assert ("Unable to destroy GPU instance ID 13 from GPU  0: In use by another client\n"
            "Failed to destroy GPU instances: In use by another client\n") == p.stdout + p.stderr
    assert tray.smi("mig", "-dci").stdout == "Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID 13\n"
    assert tray.smi("mig", "-dgi").stdout == "Successfully destroyed GPU instance ID 13 from GPU  0\n"
    assert tray.smi("-mig", "0").returncode == 0


def test_create_by_id_and_several(tray):
    tray.smi("-mig", "1")
    out = tray.smi("mig", "-cgi", "9,14", "-C").stdout.splitlines()
    assert out[0] == "Successfully created GPU instance ID  2 on GPU  0 using profile MIG 3g.93gb (ID  9)"
    assert out[2] == "Successfully created GPU instance ID  4 on GPU  0 using profile MIG 2g.47gb (ID 14)"
    assert len(out) == 4 and "compute instance ID  0" in out[1] and "GPU instance ID  4" in out[3]


def test_create_when_full(tray):
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "0")
    p = tray.smi("mig", "-cgi", "9", check=False)
    assert p.returncode != 0
    assert ("Unable to create a GPU instance on GPU  0 using profile 9: Insufficient Resources\n"
            "Failed to create GPU instances: Insufficient Resources\n") == p.stdout + p.stderr


def test_list_shows_mig_devices(tray):
    """nvidia-smi -L inside a pod (the DRA test suite checks it): the GPU,
    then its MIG devices."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "19,19", "-C")
    lines = tray.smi("-L").stdout.splitlines()
    assert re.fullmatch(r"GPU 0: NVIDIA GB200 \(UUID: GPU-[0-9a-f-]{36}\)", lines[0])
    assert len(lines) == 3
    for i, line in enumerate(lines[1:]):
        assert re.fullmatch(rf"  MIG 1g\.23gb     Device  {i}: \(UUID: MIG-[0-9a-f-]{{36}}\)", line), line


def test_query_shows_mig_mode_and_devices(tray):
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "19", "-C")
    out = tray.smi("-q", "-i", "0").stdout
    assert re.search(r"MIG Mode\n\s+Current\s+: Enabled\n\s+Pending\s+: Enabled\n", out), out
    assert re.search(r"MIG Device\n\s+Index\s+: 0\n\s+GPU Instance ID\s+: 13\n\s+Compute Instance ID\s+: 0\n", out)
    assert re.search(r"Multiprocessor count\s+: 18\n", out)


def test_list_compute_instance_profiles(tray):
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "15")
    out = tray.smi("mig", "-lcip").stdout
    assert re.search(r"\|\s+0\s+\d+\s+MIG 1g\.47gb\s+0\s+1/1\s+18\s+", out), out
    assert re.search(r"\|\s+0\s+\d+\s+MIG 1g\.47gb\s+7\*\s+1/1\s+32\s+", out), out


def test_reset_by_bus_id(tray):
    """mig-parted resets GPUs with nvidia-smi -r -i <bus id>,<bus id>."""
    out = tray.smi("-r", "-i", BUS0, root=True).stdout
    assert out == f"GPU {BUS0} was successfully reset.\nAll done.\n"
