"""GB200 NUMA layout (NVIDIA Grace Performance Tuning Guide): each Grace
CPU is a NUMA node (0, 1) and, with coherent GPU memory onlined by the OS
(the driver's default), each GPU's memory too (2, 10, 18, 26, each followed
by 7 MIG nodes). With CDMM (NVreg_CoherentGPUMemoryMode=driver) the GPUs
are not NUMA nodes."""
import os
import subprocess
import sys
from pathlib import Path

FAKEGPU = Path(__file__).resolve().parents[2] / "fakegpu"


def smi(tmp_path, *args, **conf):
    (tmp_path / "dev").mkdir(exist_ok=True)
    for i in range(4):
        (tmp_path / "dev" / f"nvidia{i}").touch()
    env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(tmp_path / "none.conf"), FAKEGPU_COUNT="4",
               FAKEGPU_DEV_DIR=str(tmp_path / "dev"), FAKEGPU_STATE_PATH=str(tmp_path / "occ"),
               **{f"FAKEGPU_{k.upper()}": str(v) for k, v in conf.items()})
    code = f"import runpy, sys; sys.argv = ['nvidia-smi', *{list(args)!r}]; runpy.run_path({str(FAKEGPU / 'nvidia-smi')!r}, run_name='__main__')"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout


def columns(out):
    """(NUMA Affinity, GPU NUMA ID) per GPU row of topo -m."""
    return [tuple(line.split()[-2:]) for line in out.splitlines() if line.startswith("GPU") and "\t" in line]


def test_gpus_are_numa_nodes_by_default(tmp_path):
    assert columns(smi(tmp_path, "topo", "-m")) == [("0", "2"), ("0", "10"), ("1", "18"), ("1", "26")]
    assert smi(tmp_path, "topo", "-gnid").split() == ["2", "10", "18", "26"]


def test_cdmm_gpus_are_not_numa_nodes(tmp_path):
    out = smi(tmp_path, "topo", "-m", coherent_gpu_memory="driver")
    assert columns(out) == [("0", "N/A"), ("0", "N/A"), ("1", "N/A"), ("1", "N/A")]


def numactl(tmp_path, *args, **conf):
    """fakegpu's numactl, as a tray's PATH has it (nested numactl calls find it too)."""
    smi(tmp_path, "-L")  # device nodes
    env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(tmp_path / "none.conf"), FAKEGPU_COUNT="4",
               FAKEGPU_DEV_DIR=str(tmp_path / "dev"), FAKEGPU_STATE_PATH=str(tmp_path / "occ"),
               PATH=f"{FAKEGPU}:{os.environ['PATH']}",
               **{f"FAKEGPU_{k.upper()}": str(v) for k, v in conf.items()})
    return subprocess.run([str(FAKEGPU / "numactl"), *args], env=env, capture_output=True, text=True)


def test_numactl_hardware(tmp_path):
    out = numactl(tmp_path, "-H").stdout.splitlines()
    assert out[0] == "available: 34 nodes (0-33)"
    assert "node 2 size: 188416 MB" in out and "node 3 size: 0 MB" in out
    dist = {int(line.split(":")[0]): line.split(":")[1].split() for line in out if line[:4].strip().isdigit()}
    assert dist[0][:4] == ["10", "40", "80", "80"] and dist[0][18] == "120"  # Grace 0: own GPUs 80, other's 120
    assert dist[2][2:5] == ["10", "11", "11"] and dist[2][10] == "40"        # GPU to its MIG nodes, other GPUs
    out = numactl(tmp_path, "-H", coherent_gpu_memory="driver").stdout
    assert out.startswith("available: 2 nodes (0-1)\n")


def test_numactl_binds_cpus_to_grace_nodes(tmp_path):
    hw = numactl(tmp_path, "-H").stdout
    grace1 = hw.split("node 1 cpus:")[1].split("\n")[0].split()
    out = numactl(tmp_path, "-N", "1", "-m", "same", "numactl", "-s").stdout
    assert "policy: bind\n" in out and "membind: 1 \n" in out and "nodebind: 1 \n" in out
    assert out.split("physcpubind:")[1].split("\n")[0].split() == grace1
    assert numactl(tmp_path, "-N", "2", "true").returncode == 1    # GPU node: no CPUs
    assert numactl(tmp_path, "-m", "3", "true").returncode == 1    # MIG node: no memory


def test_numactl_unemulated_options_fail_with_message(tmp_path):
    for args in (["--shm", "/tmp/key", "-m", "0"], ["-N", "netdev:eth0", "true"]):
        r = numactl(tmp_path, *args)
        assert r.returncode == 1 and "not emulated" in r.stderr


def test_cpu_affinity_is_the_gpus_grace_cpu(tmp_path):
    """topo -m CPU Affinity: GPUs 0-1 on the first half of the tray's CPUs (Grace 0), 2-3 on the second (Grace 1), as numactl -H splits them."""
    with open("/sys/devices/system/cpu/online") as f:
        online = [c for part in f.read().strip().split(",") for c in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)]
    half = -(-len(online) // 2)
    first, second = online[:half], online[half:]
    rng = lambda cs: f"{cs[0]}-{cs[-1]}" if len(cs) > 1 else f"{cs[0]}"
    rows = [line.split("\t") for line in smi(tmp_path, "topo", "-m").splitlines() if line.startswith("GPU") and "\t" in line]
    assert [r[-4] for r in rows] == [rng(first), rng(first), rng(second), rng(second)]
    hw = numactl(tmp_path, "-H").stdout
    assert hw.split("node 0 cpus:")[1].split("\n")[0].split() == [str(c) for c in first]
