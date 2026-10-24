"""MIG through nvidia-smi, in the formats a GB200 prints them (real output in
mig-parted#227 and dra-driver-nvidia-gpu#907, #1454; the MIG user guide for
the rest), as NVIDIA's DRA driver test suite drives it (nvidia-smi -mig 1,
mig -lgip, mig -cgi <profile> -C, mig -dci, -dgi, -L) and mig-parted resets
GPUs (nvidia-smi -r -i <bus ids>)."""
import re

import pytest

from mig_tray import PLACEMENTS, PROFILES, Tray

BUS0 = "00000008:01:00.0"

# nvidia-smi mig -lgip on a GB200, verbatim (mig-parted#227).
LGIP = """\
+-------------------------------------------------------------------------------+
| GPU instance profiles:                                                        |
| GPU   Name               ID    Instances   Memory     P2P    SM    DEC   ENC  |
|                                Free/Total   GiB              CE    JPEG  OFA  |
|===============================================================================|
|   0  MIG 1g.23gb         19     7/7        22.00      No     18     1     0   |
|                                                               2     1     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 1g.23gb+me      20     1/1        22.00      No     18     1     0   |
|                                                               2     1     1   |
+-------------------------------------------------------------------------------+
|   0  MIG 1g.47gb         15     4/4        45.50      No     32     1     0   |
|                                                               2     1     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 2g.47gb         14     3/3        45.50      No     38     2     0   |
|                                                               4     2     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 3g.93gb          9     2/2        91.50      No     72     3     0   |
|                                                               6     3     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 4g.93gb          5     1/1        91.50      No     76     4     0   |
|                                                               8     4     1   |
+-------------------------------------------------------------------------------+
|   0  MIG 7g.186gb         0     1/1        184.00     No     152    7     0   |
|                                                              16     7     2   |
+-------------------------------------------------------------------------------+
"""


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


def test_list_gpu_instance_profiles_verbatim(tray):
    """mig -lgip as a GB200 prints it, character for character: profiles in
    nvidia-smi's order (by size), the 81-column box."""
    assert tray.smi("mig", "-lgip", "-i", "0").stdout == LGIP


def test_list_placements(tray):
    """In nvidia-smi's order (the MIG user guide's -lgipp: 19, 20, 15, 14, 9,
    5, 0)."""
    out = tray.smi("mig", "-lgipp", "-i", "0").stdout
    expected = ""
    for pid, places in PLACEMENTS.items():
        starts = ",".join(str(s) for s, _ in places)
        label = "Placement :" if len(places) == 1 else "Placements:"
        expected += f"GPU  0 Profile ID {pid:2d} {label} {{{starts}}}:{places[0][1]}\n"
    assert out == expected


def test_changes_need_root(tray):
    """As a user nvidia-smi passes NVML's NO_PERMISSION on: 'Insufficient
    Permissions', exit code 4 (the real tool's for permissions)."""
    p = tray.smi("-i", "0", "-mig", "1", root=False, check=False)
    assert p.returncode == 4
    assert p.stdout == (f"Unable to enable MIG Mode for GPU {BUS0}: Insufficient Permissions\n"
                        "Terminating early due to previous errors.\n")
    tray.smi("-i", "0", "-mig", "1")
    p = tray.smi("mig", "-cgi", "19", "-C", root=False, check=False)
    assert p.returncode == 4
    assert p.stdout == ("Unable to create a GPU instance on GPU  0 using profile 19: Insufficient Permissions\n"
                        "Failed to create GPU instances: Insufficient Permissions\n")
    tray.smi("mig", "-cgi", "19", "-C")
    p = tray.smi("mig", "-dci", root=False, check=False)
    assert p.returncode == 4 and "Insufficient Permissions" in p.stdout
    assert "MIG 1g.23gb" in tray.smi("mig", "-lgi", root=False).stdout  # listing needs no root


def test_pending_mode_while_in_use_then_reset(tray):
    """-mig 1 on a GPU a process uses: the real tool's pending warning, the
    mode pending; once the process is gone, nvidia-smi -r applies it."""
    with tray.holding():
        p = tray.smi("-i", "0", "-mig", "1")
        assert p.stdout == (f"Warning: MIG mode is in pending enable state for GPU {BUS0}:In use by another client\n"
                            f"{BUS0} is currently being used by one or more other processes (e.g. CUDA application or "
                            "a monitoring application such as another instance of nvidia-smi). Please first kill all "
                            "processes using the device and retry the command or reboot the system to make MIG mode "
                            "effective.\nAll done.\n")
        q = tray.smi("--query-gpu=mig.mode.current,mig.mode.pending", "--format=csv,noheader").stdout
        assert q == "Disabled, Enabled\n"
        assert tray.smi("-r", check=False).returncode != 0  # refused while the process runs
    assert tray.smi("-r", "-i", "0").stdout == f"GPU {BUS0} was successfully reset.\nAll done.\n"
    q = tray.smi("--query-gpu=mig.mode.current,mig.mode.pending", "--format=csv,noheader").stdout
    assert q == "Enabled, Enabled\n"


def test_destroy_in_use(tray):
    """mig -dci on a compute instance a process uses: In use by another
    client; the other instances go."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "19,19", "-C")
    uuid = tray.nvml("d = m.dev(0); out = m.uuid(m.mig_devices(d)[1])[1]")  # GI 13
    with tray.holding(env={"CUDA_VISIBLE_DEVICES": uuid}):
        p = tray.smi("mig", "-dci", check=False)
        assert p.returncode == 19
        assert p.stdout == ("Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID 12\n"
                            "Unable to destroy compute instance ID  0 from GPU  0 GPU instance ID 13: "
                            "In use by another client\n"
                            "Failed to destroy compute instances: In use by another client\n")
    assert tray.smi("mig", "-dci").returncode == 0


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
    """mig -lcip as a GB200 prints it for a 1g.47gb (dra-driver-nvidia-gpu
    #1454, verbatim but for the GPU instance id: 6 here, 4 there)."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "15")
    out = tray.smi("mig", "-lcip").stdout
    assert out == """\
+--------------------------------------------------------------------------------------+
| Compute instance profiles:                                                           |
| GPU     GPU       Name             Profile  Instances   Exclusive       Shared       |
|       Instance                       ID     Free/Total     SM       DEC   ENC   OFA  |
|         ID                                                          CE    JPEG       |
|======================================================================================|
|   0      6       MIG 1g.47gb          0      1/1           18        1     0     0   |
|                                                                      2     1         |
+--------------------------------------------------------------------------------------+
|   0      6       MIG 1g.47gb          7*     1/1           32        1     0     0   |
|                                                                      2     1         |
+--------------------------------------------------------------------------------------+
"""


def test_list_gpu_instances_layout(tray):
    """mig -lgi in the MIG user guide's layout, ordered as profiles are
    listed, then by id."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "14,19,19")
    assert tray.smi("mig", "-lgi").stdout == """\
+----------------------------------------------------+
| GPU instances:                                     |
| GPU   Name          Profile  Instance   Placement  |
|                       ID       ID       Start:Size |
|====================================================|
|   0  MIG 1g.23gb      19       10          3:1     |
+----------------------------------------------------+
|   0  MIG 1g.23gb      19       13          6:1     |
+----------------------------------------------------+
|   0  MIG 2g.47gb      14        5          4:2     |
+----------------------------------------------------+
"""


def test_create_compute_instances(tray):
    """mig -cci names no compute instance id (the MIG user guide's
    output); 1c.3g.93gb has a 1g's SMs."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "9")
    out = tray.smi("mig", "-cci", "0,0,0", "-gi", "2").stdout
    assert out == ("Successfully created compute instance on GPU  0 GPU instance ID  2 using profile "
                   "MIG 1c.3g.93gb (ID  0)\n") * 3
    assert tray.nvml("d = m.dev(0); out = [m.attributes(h)[1]['multiprocessorCount'] for h in m.mig_devices(d)]") \
        == [18, 18, 18]


def mig_section(out):
    """The default output's MIG devices section, its rows from index 7."""
    lines = out.splitlines()
    i = lines.index("| MIG devices:" + " " * 76 + "|") - 1
    return lines[i:lines.index("", i)]


def test_default_table_without_mig(tray):
    out = tray.smi().stdout
    assert "MIG devices:" not in out
    row = [line for line in out.splitlines() if line.startswith("|   0  NVIDIA GB200")][0]
    assert out.splitlines()[out.splitlines().index(row) + 2] == "|" + " " * 41 + "|" + " " * 24 + "|" + "Disabled".rjust(21) + " |"


def test_default_table_with_mig(tray):
    """MIG M. Enabled, GPU-Util N/A, and the MIG devices section as a
    GB200's (dra-driver-nvidia-gpu#1454: a 1g.47gb with its 18-SM compute
    instance, there at GI 4 with 46848 MiB on another driver); compute
    instances of one GPU instance share its memory cell."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "15")
    tray.smi("mig", "-cci", "0", "-gi", "6")
    out = tray.smi().stdout
    lines = out.splitlines()
    gpu = lines.index([line for line in lines if line.startswith("|   0  NVIDIA GB200")][0])
    assert lines[gpu].endswith("|                   On |")
    assert lines[gpu + 1].endswith("|     N/A      Default |")
    assert lines[gpu + 2] == "|                                         |                        |              Enabled |"
    assert mig_section(out) == """\
+-----------------------------------------------------------------------------------------+
| MIG devices:                                                                            |
+------------------+----------------------------------+-----------+-----------------------+
| GPU  GI  CI  MIG |              Shared Memory-Usage |        Vol|        Shared         |
|      ID  ID  Dev |                Shared BAR1-Usage | SM     Unc| CE ENC  DEC  OFA  JPG |
|                  |                                  |        ECC|                       |
|==================+==================================+===========+=======================|
|  0    6   0   0  |              16MiB / 46592MiB    | 18      0 |  2   0    1    0    1 |
|                  |               0MiB / 18225MiB    |           |                       |
+------------------+----------------------------------+-----------+-----------------------+""".splitlines()
    tray.smi("mig", "-cgi", "9")
    tray.smi("mig", "-cci", "0,0", "-gi", "1")  # the 3g at 0:4
    rows = mig_section(tray.smi().stdout)[7:]
    assert rows[:5] == ["|  0    1   0   0  |              16MiB / 93696MiB    | 18      0 |  6   0    3    0    3 |",
                        "|                  |               0MiB / 36650MiB    |           |                       |",
                        "+------------------+                                  +-----------+-----------------------+",
                        "|  0    1   1   1  |                                  | 18      0 |  6   0    3    0    3 |",
                        "|                  |                                  |           |                       |"]


def test_default_table_mig_without_devices(tray):
    tray.smi("-mig", "1")
    rows = mig_section(tray.smi().stdout)[7:]
    assert rows == ["|  No MIG devices found" + " " * 67 + "|", "+" + "-" * 89 + "+"]


def test_processes_on_mig_devices(tray):
    """The process table names a process's GPU and compute instance."""
    tray.smi("-mig", "1")
    tray.smi("mig", "-cgi", "19", "-C")
    uuid = tray.nvml("d = m.dev(0); out = m.uuid(m.mig_devices(d)[0])[1]")
    with tray.holding(mib=100, env={"CUDA_VISIBLE_DEVICES": uuid}) as p:
        out = tray.smi().stdout
        assert re.search(rf"\|    0   13    0 +{p.pid} +C +\S+ +100MiB \|", out), out
        assert "|  0   13   0   0  |             116MiB / 22528MiB    |" in out


def test_reset_by_bus_id(tray):
    """mig-parted resets GPUs with nvidia-smi -r -i <bus id>,<bus id>."""
    out = tray.smi("-r", "-i", BUS0, root=True).stdout
    assert out == f"GPU {BUS0} was successfully reset.\nAll done.\n"
