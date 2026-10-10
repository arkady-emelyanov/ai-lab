"""CUDA virtual memory management on the fake driver (fakegpu/vmm.c), on this
machine: allocations mapped into reserved address ranges, shared between
processes as POSIX file descriptors (one tray) or fabric handles (across
trays of an NVLink partition), and multicast objects. Each process is an
emulated tray with its GPU in a chosen NVLink partition (clique)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FAKEGPU = REPO / "fakegpu"
PROC = Path(__file__).with_name("vmm_proc.py")
UUID = "7f3c2a10-5b4e-4d6a-9c1e-2b8f0e6d4a91"
NOT_PERMITTED = 800


@pytest.fixture(scope="session", autouse=True)
def build():
    subprocess.run(["make", "-s", "-C", str(FAKEGPU)], check=True)


@pytest.fixture
def tray(tmp_path):
    """tray(name, clique) -> environment of a process on that tray."""
    fabric = tmp_path / "fabric"

    def env(name, clique=1):
        t = tmp_path / name
        (t / "dev").mkdir(parents=True, exist_ok=True)
        (t / "dev" / "nvidia0").touch()
        (t / "sb").mkdir(exist_ok=True)
        (t / "sb" / "fabric-clique").write_text(f"0 {clique}\n")
        return dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(t / "none.conf"), FAKEGPU_COUNT="1",
                    FAKEGPU_HOST=name, FAKEGPU_CLUSTER_UUID=UUID, FAKEGPU_DEV_DIR=str(t / "dev"),
                    FAKEGPU_SIDEBAND_DIR=str(t / "sb"), FAKEGPU_STATE_PATH=str(t / "occ"),
                    FAKEGPU_FABRIC_DIR=str(fabric), FAKEGPU_IPC_DIR=str(t / "ipc"), FAKEGPU_MEM_MB="1024")

    env.fabric = fabric
    env.rendezvous = str(tmp_path / "rv")
    return env


def run(env, *args):
    p = subprocess.run([sys.executable, str(PROC), str(FAKEGPU), *args], env=env, capture_output=True, text=True,
                       timeout=30)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.splitlines()[-1])


def share(tray, how, kind, exporter, importer):
    """Exporter on one tray, importer on another: the importer's result."""
    exp = subprocess.Popen([sys.executable, str(PROC), str(FAKEGPU), "export", how, kind, tray.rendezvous],
                           env=exporter, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert json.loads(exp.stdout.readline()) == {"ready": True}
    try:
        result = run(importer, "import", how, kind, tray.rendezvous)
    finally:
        out, err = exp.communicate(timeout=30)
    assert exp.returncode == 0, err
    assert all(v == 0 for v in json.loads(out.splitlines()[-1])["rc"].values())
    return result


def test_supported_and_mapped_memory_is_device_memory(tray):
    r = run(tray("tray0"), "local")
    assert r["attrs"] == {"102": 1, "103": 1, "128": 1, "132": 1}
    assert all(v == 0 for v in r["rc"].values()), r["rc"]
    assert r["roundtrip"] and r["memory_type"] == 2  # CU_MEMORYTYPE_DEVICE
    assert r["oom"] == 2  # beyond the GPU's memory


def test_no_fabric_handles_without_fabric_dir(tray):
    env = dict(tray("tray0"), FAKEGPU_FABRIC_DIR="")
    assert run(env, "local")["attrs"]["128"] == 0


def test_fd_shares_memory_between_processes(tray):
    r = share(tray, "fd", "mem", tray("tray0"), tray("tray0"))
    assert r["rc"]["cuMemImportFromShareableHandle"] == 0 and r["read"]


def test_fabric_handle_reaches_another_tray_of_the_partition(tray):
    r = share(tray, "fabric", "mem", tray("tray0", 1), tray("tray1", 1))
    assert r["rc"]["cuMemImportFromShareableHandle"] == 0 and r["read"]
    assert not any(tray.fabric.iterdir())  # released by its creator: gone


def test_fabric_handle_stops_at_the_partition(tray):
    r = share(tray, "fabric", "mem", tray("tray0", 1), tray("tray1", 7))
    assert r["rc"]["cuMemImportFromShareableHandle"] == NOT_PERMITTED


def test_gpu_in_no_partition_has_no_fabric_memory(tray):
    env = tray("tray0", 0)
    p = subprocess.Popen([sys.executable, str(PROC), str(FAKEGPU), "export", "fabric", "mem", tray.rendezvous],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(timeout=30)
    assert json.loads(out.splitlines()[-1])["rc"]["cuMemCreate"] == NOT_PERMITTED


@pytest.mark.parametrize("how", ["fd", "fabric"])
def test_multicast_object_is_shared_and_bound(tray, how):
    r = share(tray, how, "mc", tray("tray0", 1), tray("tray0" if how == "fd" else "tray1", 1))
    assert all(v == 0 for v in r["rc"].values()), r["rc"]


def test_multicast_stops_at_the_partition(tray):
    r = share(tray, "fabric", "mc", tray("tray0", 1), tray("tray1", 7))
    assert r["rc"]["cuMemImportFromShareableHandle"] == NOT_PERMITTED


def test_memory_of_a_killed_process_is_freed(tray):
    """A process killed holding fabric memory leaves its file; the next
    creation deletes it (nobody holds it), as the driver frees a dead
    process' memory. A live holder's file stays."""
    def exporter(rendezvous):
        p = subprocess.Popen([sys.executable, str(PROC), str(FAKEGPU), "export", "fabric", "mem", rendezvous],
                             env=tray("tray0"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert json.loads(p.stdout.readline()) == {"ready": True}
        return p

    dead, live = exporter(tray.rendezvous + "a"), exporter(tray.rendezvous + "b")
    dead.kill()
    dead.wait()
    for f in tray.fabric.iterdir():  # past the grace period for new files
        os.utime(f, (0, 0))
    assert len(list(tray.fabric.iterdir())) == 2
    third = exporter(tray.rendezvous + "c")
    assert len(list(tray.fabric.iterdir())) == 2  # the dead one's is gone
    for p, rv in ((live, "b"), (third, "c")):
        open(tray.rendezvous + rv + ".done", "w").close()
        p.communicate(timeout=30)
    assert not any(tray.fabric.iterdir())


def ipc(tray, exporter, importer):
    """vLLM's P2P check over legacy CUDA IPC: (exporter, importer) results."""
    exp = subprocess.Popen([sys.executable, str(PROC), str(FAKEGPU), "ipc-export", tray.rendezvous],
                           env=exporter, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert json.loads(exp.stdout.readline()) == {"ready": True}
    try:
        imp = run(importer, "ipc-import", tray.rendezvous)
    finally:
        out, err = exp.communicate(timeout=30)
    assert exp.returncode == 0, err
    return json.loads(out.splitlines()[-1]), imp


def test_cuda_ipc_shares_a_small_allocation(tray):
    exp, imp = ipc(tray, tray("tray0"), tray("tray0"))
    assert imp["rc"]["cuIpcOpenMemHandle_v2"] == 0
    assert imp["sees_peer_data"] and imp["reads_back"] and exp["sees_peer_write"] and exp["still_mine"]
    assert exp["open_own"] == 201  # CUDA_ERROR_INVALID_CONTEXT: not in the exporting process


def test_cuda_ipc_stays_on_its_tray(tray):
    """Any process on the tray opens the handle (on real GPUs, also from
    another container); one on another tray (node) cannot."""
    exp, imp = ipc(tray, tray("tray0"), tray("tray1"))
    assert imp["rc"]["cuIpcOpenMemHandle_v2"] == 400  # CUDA_ERROR_INVALID_HANDLE
    assert not exp["sees_peer_write"]


def test_cuda_ipc_memory_goes_with_its_owner(tray):
    exp, _ = ipc(tray, tray("tray0"), tray("tray0"))
    assert exp["rc"]["cuMemFree_v2"] == 0
    assert not any((tray.fabric.parent / "tray0" / "ipc").iterdir())
