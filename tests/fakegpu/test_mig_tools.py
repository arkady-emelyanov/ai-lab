"""NVIDIA's own MIG tools, unchanged, on the emulated GPUs:

- mig-parted (what the GPU Operator's MIG manager runs) applies, asserts and
  exports each of its GB200 configurations (config-default.yaml, device
  filter 0x294110DE), and applying one again changes nothing;
- GPU Feature Discovery labels a node of MIG-partitioned GB200s as it does a
  real one, with the mixed and single strategies.

Built from NVIDIA's sources with Go into .cache/mig-tools (skipped without Go
or the network). mig-parted finds GPUs through sysfs and wants the nvidia
module loaded: each run gets a user namespace with a GB200's PCI devices
and /proc/modules bind-mounted, as on a GB200 node."""
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pytest
import yaml

from mig_tray import Tray

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / ".cache" / "mig-tools"
SOURCES = {
    "nvidia-mig-parted": ("mig-parted", "0.15.1", "./cmd/nvidia-mig-parted"),
    "gpu-feature-discovery": ("k8s-device-plugin", "0.20.1", "./cmd/gpu-feature-discovery"),
}
GB200 = "0x294110DE"
CONFIGS = ["all-disabled", "all-enabled", "all-1g.23gb", "all-1g.23gb.me", "all-1g.47gb", "all-2g.47gb",
           "all-3g.93gb", "all-4g.93gb", "all-7g.186gb", "all-balanced"]
GPUS = 2
ADDRESSES = ["0008:01:00.0", "0009:01:00.0"]


def source(repo, version):
    src = CACHE / f"{repo}-{version}"
    if not src.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tgz = CACHE / f"{repo}-{version}.tar.gz"
        urllib.request.urlretrieve(f"https://github.com/NVIDIA/{repo}/archive/refs/tags/v{version}.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(CACHE, filter="data")
    return src


@pytest.fixture(scope="session")
def tools():
    if not shutil.which("go") or not shutil.which("unshare"):
        pytest.skip("needs Go and unshare")
    paths = {}
    for binary, (repo, version, pkg) in SOURCES.items():
        out = CACHE / "bin" / f"{binary}-{version}"
        try:
            src = source(repo, version)
        except OSError as e:
            pytest.skip(f"cannot download {repo}: {e}")
        if not out.exists():
            subprocess.run(["go", "build", "-mod=vendor", "-o", str(out), pkg], cwd=src, check=True,
                           env=dict(os.environ, CGO_ENABLED="1", GOFLAGS="-buildvcs=false"))
        paths[binary] = out
    paths["config"] = source("mig-parted", "0.15.1") / "deployments" / "systemd" / "config-default.yaml"
    return paths


@pytest.fixture
def node(tmp_path):
    """A GB200 node: GPUS emulated GPUs, their PCI devices and the nvidia
    module, for commands run in a user namespace."""
    tray = Tray(tmp_path, count=GPUS)
    sys_dir = tmp_path / "sys"
    (sys_dir / "drivers" / "nvidia").mkdir(parents=True)
    for i, addr in enumerate(ADDRESSES):
        d = sys_dir / "devices" / addr
        d.mkdir(parents=True)
        for f, v in (("vendor", "0x10de"), ("device", "0x2941"), ("class", "0x030200"),
                     ("subsystem_vendor", "0x10de"), ("subsystem_device", "0x2046"), ("numa_node", str(i)),
                     ("resource", "0x0000000000000000 0x0000000003ffffff 0x0000000000040200\n"
                                  "0x0000000000000000 0x0000000000000000 0x0000000000000000")):
            (d / f).write_text(v + "\n")
        (d / "driver").symlink_to(sys_dir / "drivers" / "nvidia")
    modules = tmp_path / "modules"
    modules.write_text("nvidia_uvm 2056192 0 - Live 0x0000000000000000\nnvidia 14376960 1 nvidia_uvm, Live 0x0000000000000000\n")

    def run(*cmd, check=True):
        script = (f"mount --bind {modules} /proc/modules && mount --bind {sys_dir / 'devices'} /sys/bus/pci/devices"
                  ' && exec "$@"')
        smi_dir = tmp_path / "bin"
        smi_dir.mkdir(exist_ok=True)
        smi = smi_dir / "nvidia-smi"
        if not smi.exists():
            smi.write_text(f"#!/bin/sh\nexec python3 {REPO / 'fakegpu' / 'nvidia-smi'} \"$@\"\n")
            smi.chmod(0o755)
        env = dict(tray.env, PATH=f"{smi_dir}:{os.environ['PATH']}")
        p = subprocess.run(["unshare", "-Urm", "sh", "-c", script, "sh", *map(str, cmd)], env=env,
                           capture_output=True, text=True, timeout=300)
        if check:
            assert p.returncode == 0, p.stdout + p.stderr
        return p

    run.tray = tray
    return run


def gb200_entry(config_file, name):
    configs = yaml.safe_load(Path(config_file).read_text())["mig-configs"][name]
    for entry in configs:
        filt = entry.get("device-filter")
        if filt is None or GB200 in (filt if isinstance(filt, list) else [filt]):
            return entry
    raise AssertionError(f"{name}: no GB200 entry")


@pytest.mark.parametrize("config", CONFIGS)
def test_mig_parted_applies_its_gb200_configurations(tools, node, config):
    mp, cfg = tools["nvidia-mig-parted"], tools["config"]
    node(mp, "apply", "-f", cfg, "-c", config)
    node(mp, "assert", "-f", cfg, "-c", config)
    exported = yaml.safe_load(node(mp, "export").stdout)
    want = gb200_entry(cfg, config)
    got = {}
    for entry in exported["mig-configs"]["current"]:
        for dev in (range(GPUS) if entry["devices"] == "all" else entry["devices"]):
            got[dev] = (entry["mig-enabled"], entry.get("mig-devices") or {})
    assert got == {i: (want["mig-enabled"], want.get("mig-devices") or {}) for i in range(GPUS)}
    again = node(mp, "apply", "-f", cfg, "-c", config)  # already applied: nothing to do
    assert again.returncode == 0


def test_mig_parted_moves_between_configurations(tools, node):
    mp, cfg = tools["nvidia-mig-parted"], tools["config"]
    for config in ("all-1g.23gb", "all-balanced", "all-7g.186gb", "all-disabled"):
        node(mp, "apply", "-f", cfg, "-c", config)
        node(mp, "assert", "-f", cfg, "-c", config)
    assert node.tray.nvml("out = [m.mig_mode(m.dev(i)) for i in range(2)]") == [[0, 0, 0]] * 2


def gfd_labels(tools, node, strategy, tmp_path):
    out = tmp_path / f"gfd-{strategy}"
    machine = tmp_path / "machine-type"
    machine.write_text("GB200 NVL72\n")
    node(tools["gpu-feature-discovery"], "--oneshot", "--no-timestamp", "--use-node-feature-api=false",
         f"--output-file={out}",
         f"--mig-strategy={strategy}", f"--machine-type-file={machine}")
    return dict(line.split("=", 1) for line in out.read_text().split())


def test_gpu_feature_discovery_mixed(tools, node, tmp_path):
    node(tools["nvidia-mig-parted"], "apply", "-f", tools["config"], "-c", "all-1g.23gb")
    labels = gfd_labels(tools, node, "mixed", tmp_path)
    assert labels["nvidia.com/mig.capable"] == "true" and labels["nvidia.com/mig.strategy"] == "mixed"
    expected = {"count": str(7 * GPUS), "memory": "22528", "multiprocessors": "18", "slices.gi": "1",
                "slices.ci": "1", "engines.copy": "2", "engines.decoder": "1", "engines.encoder": "0",
                "engines.jpeg": "1", "engines.ofa": "0"}
    for key, value in expected.items():
        assert labels[f"nvidia.com/mig-1g.23gb.{key}"] == value, key


def test_gpu_feature_discovery_single(tools, node, tmp_path):
    node(tools["nvidia-mig-parted"], "apply", "-f", tools["config"], "-c", "all-3g.93gb")
    labels = gfd_labels(tools, node, "single", tmp_path)
    assert labels["nvidia.com/gpu.product"] == "NVIDIA-GB200-MIG-3g.93gb"
    assert labels["nvidia.com/gpu.count"] == str(2 * GPUS)
    assert labels["nvidia.com/gpu.memory"] == "93696"
    assert labels["nvidia.com/gpu.multiprocessors"] == "72"


def test_gpu_feature_discovery_without_mig(tools, node, tmp_path):
    labels = gfd_labels(tools, node, "mixed", tmp_path)
    assert labels["nvidia.com/mig.capable"] == "true"
    assert not any(k.startswith("nvidia.com/mig-") for k in labels)
    assert labels["nvidia.com/gpu.count"] == str(GPUS)
