"""Fake NCCL across NVLink partitions, on this machine (no lab needed): each
rank is a process on an emulated tray of its own, with its GPU in a chosen
partition (clique); ranks exchange partitions through a shared directory.
The simulated cost of an operation is read from the GPU's busy time and
NIC counters in the tray's shared state, so the checks are exact.

Cost model (fakegpu/nccl_stub.c): one partition, a ring over NVLink:
10 us + B x bus / 900 B/ns. Across partitions: 20 us + B x bus(m) / 900 +
B x bus(g) / m / 50 B/ns, for g partitions of at least m ranks; bus(n) is
2(n-1)/n for all-reduce. NIC: 400 Gb/s = 50 B/ns per GPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FAKEGPU = REPO / "fakegpu"
RANK = Path(__file__).with_name("nccl_rank.py")
UUID = "7f3c2a10-5b4e-4d6a-9c1e-2b8f0e6d4a91"
B = 1 << 28  # bytes per all-reduce
NVLINK, IB = 900.0, 50.0  # bytes per ns


def allreduce_bus(n):
    return 2 * (n - 1) / n if n > 1 else 0


@pytest.fixture(scope="session", autouse=True)
def build():
    subprocess.run(["make", "-s", "-C", str(FAKEGPU)], check=True)


@pytest.fixture
def lab(tmp_path):
    """Runs ranks: run(cliques, op, nbytes, args) -> per-rank results."""
    shared = tmp_path / "nccl"
    shared.mkdir()

    def run(cliques, op="allreduce", nbytes=B, args=None, ranks=None, exchange=True, timeout=10, profile=None):
        idfile = tmp_path / f"id-{op}-{len(list(tmp_path.iterdir()))}"
        procs = []
        for rank in (range(len(cliques)) if ranks is None else ranks):
            tray = tmp_path / f"tray{rank}"
            (tray / "dev").mkdir(parents=True, exist_ok=True)
            (tray / "dev" / "nvidia0").touch()
            (tray / "sb").mkdir(exist_ok=True)
            (tray / "sb" / "fabric-clique").write_text(f"0 {cliques[rank]}\n")
            (tray / "occ").unlink(missing_ok=True)
            env = dict(os.environ, LD_LIBRARY_PATH=str(FAKEGPU), FAKEGPU_CONF=str(tray / "none.conf"),
                       FAKEGPU_COUNT="1", FAKEGPU_HOST=f"tray{rank}", FAKEGPU_CLUSTER_UUID=UUID,
                       FAKEGPU_DEV_DIR=str(tray / "dev"), FAKEGPU_SIDEBAND_DIR=str(tray / "sb"),
                       FAKEGPU_STATE_PATH=str(tray / "occ"), FAKEGPU_NCCL_TIMEOUT_S=str(timeout),
                       FAKEGPU_NCCL_DIR=str(shared) if exchange else "",
                       **{f"FAKEGPU_{k.upper()}": str(v) for k, v in (profile or {}).items()})
            cmd = [sys.executable, str(RANK), str(FAKEGPU), str(len(cliques)), str(rank), str(idfile), op, str(nbytes)]
            if args is not None:
                cmd.append(str(args[rank]))
            procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        results = []
        for p in procs:
            out, err = p.communicate(timeout=timeout + 30)
            assert p.returncode == 0, err
            results.append(dict(json.loads(out), stderr=err))
        return results

    run.shared = shared
    run.tray = lambda rank: tmp_path / f"tray{rank}"
    return run


def near(value, expected):
    return abs(value - expected) <= max(1.0, expected * 1e-6)


def test_one_partition_runs_at_nvlink_speed(lab):
    for r in lab([1, 1, 1, 1]):
        assert near(r["busy_ns"], 10_000 + B * allreduce_bus(4) / NVLINK)
        assert r["ib_tx"] == 0


def test_two_partitions_cross_infiniband(lab):
    """2 + 2 ranks: NVLink inside each pair, the NICs between the pairs."""
    for r in lab([1, 1, 2, 2]):
        ib = B * allreduce_bus(2) / 2
        assert near(r["busy_ns"], 20_000 + B * allreduce_bus(2) / NVLINK + ib / IB)
        assert near(r["ib_tx"], ib) and near(r["ib_rx"], ib)
    slow = lab([1, 1, 2, 2])[0]["busy_ns"] / lab([1, 1, 1, 1])[0]["busy_ns"]
    assert slow > 2, f"a split all-reduce is only {slow:.1f}x slower"


def test_gpus_in_no_partition_only_use_their_nics(lab):
    """Clique 0: a GPU outside every partition has no NVLink peer at all."""
    for r in lab([0, 0, 0, 0]):
        assert near(r["busy_ns"], 20_000 + B * allreduce_bus(4) / IB)
        assert r["nvlink_tx"] == 0


def test_split_by_partition_is_back_on_nvlink(lab):
    """ncclCommSplit is collective: each child knows its members, here one
    partition each, and runs at NVLink speed; NOCOLOR gets no communicator."""
    results = lab([1, 1, 2, 2], op="split", args=[1, 1, 2, -1])
    for r in results[:2]:
        assert r["child_nranks"] == 2
        assert near(r["busy_ns"], 10_000 + B * allreduce_bus(2) / NVLINK) and r["ib_tx"] == 0
    assert results[2]["child_nranks"] == 1 and results[3]["child_nranks"] == 0
    assert sorted(r["child_rank"] for r in results[:2]) == [0, 1]


def test_send_to_another_partition_uses_the_nic(lab):
    same = lab([1, 1, 2, 2], op="send", args=[1, 0, 3, 2])[0]
    other = lab([1, 1, 2, 2], op="send", args=[2, 3, 0, 1])[0]
    assert near(same["busy_ns"], 10_000 + B / NVLINK) and same["ib_tx"] == 0
    assert near(other["busy_ns"], 10_000 + B / IB) and near(other["ib_tx"], B)


def test_missing_peers_fall_back_to_nvlink_and_never_fail(lab):
    """A rank whose peers never report waits nccl_timeout_s, warns once, and
    assumes one partition, as before; NCCL calls never fail."""
    r, = lab([1, 2], ranks=[0], timeout=1)
    assert near(r["busy_ns"], 10_000 + B * allreduce_bus(2) / NVLINK)
    assert "not every rank reported its NVLink partition" in r["stderr"]


def test_without_an_exchange_directory_ranks_share_one_partition(lab):
    for r in lab([1, 1, 2, 2], exchange=False):
        assert near(r["busy_ns"], 10_000 + B * allreduce_bus(4) / NVLINK)
        assert r["stderr"] == ""


def test_exchange_directories_are_removed(lab):
    lab([1, 1, 2, 2], op="split", args=[1, 1, 2, -1])
    assert list(lab.shared.iterdir()) == []


def test_unique_ids_are_random(lab):
    r, = lab([1], op="uniqueid")
    a, b = r["ids"]
    other, = lab([1], op="uniqueid")
    assert len({a, b, *other["ids"]}) == 4


def test_nic_counters_reach_node_exporter_names(lab, tmp_path):
    """fakeib/ib-port-counters publishes the NIC bytes for node_exporter's
    textfile collector, named as its infiniband collector names them."""
    r = lab([1, 1, 2, 2])[0]
    out = tmp_path / "infiniband.prom"
    subprocess.run([str(REPO / "fakeib/ib-port-counters"), str(lab.tray(0) / "occ"), str(out), "4"], check=True)
    samples = dict(line.rsplit(" ", 1) for line in out.read_text().splitlines() if not line.startswith("#"))
    tx = samples['node_infiniband_port_data_transmitted_bytes_total{device="mlx5_0",port="1"}']
    rx = samples['node_infiniband_port_data_received_bytes_total{device="mlx5_0",port="1"}']
    assert int(tx) == r["ib_tx"] > 0 and int(rx) == r["ib_rx"] > 0
    assert samples['node_infiniband_port_data_transmitted_bytes_total{device="mlx5_1",port="1"}'] == "0"


def test_cost_follows_the_gpu_profile(lab):
    """fakegpu_nvlink_link_gbs and ib_link_rate set the rates the model uses."""
    for r in lab([1, 1, 1, 1], profile=dict(nvlink_link_gbs=25)):
        assert near(r["busy_ns"], 10_000 + B * allreduce_bus(4) / (18 * 25))
    for r in lab([1, 1, 2, 2], profile=dict(ib_gbps=200)):
        assert near(r["busy_ns"], 20_000 + B * allreduce_bus(2) / NVLINK + B * allreduce_bus(2) / 2 / 25)
