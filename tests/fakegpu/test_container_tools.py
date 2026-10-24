"""The fake driver's container tools, on this machine, no lab needed:

- nvidia-container-cli lists the GPUs' device nodes and the driver from the
  tray's CDI specification, and `configure` (what enroot's NVIDIA hook and
  Apptainer's --nv run) brings the device nodes of the GPUs asked for and
  the driver's files into a container's root filesystem, in an unprivileged
  user namespace as enroot and Apptainer run it;
- nvidia-container-runtime adds the GPUs NVIDIA_VISIBLE_DEVICES or the Slurm
  step names to an OCI bundle's config.json, then runs the real runtime with
  the same arguments, and leaves bundles without GPUs alone.

The CDI specification here names stand-in device nodes (/dev/null, /dev/zero,
/dev/full) and a driver directory of the test's own."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "fakegpu" / "nvidia-container-cli"
RUNTIME = REPO / "fakegpu" / "nvidia-container-runtime"
UUIDS = ["GPU-11111111-2222-3333-4444-555555555555", "GPU-66666666-7777-8888-9999-000000000000"]
NODES = ["/dev/zero", "/dev/full"]  # stand-ins for /dev/nvidia0, /dev/nvidia1


@pytest.fixture
def spec(tmp_path):
    driver = tmp_path / "fakegpu"
    driver.mkdir()
    (driver / "libcuda.so.1").write_text("fake\n")
    (tmp_path / "fakegpu.conf").write_text("count = 2\n")
    devices = []
    for idx, (uuid, node) in enumerate(zip(UUIDS, NODES)):
        devices += [{"name": name, "containerEdits": {"deviceNodes": [{"path": node}]}} for name in (str(idx), uuid)]
    path = tmp_path / "fakegpu.json"
    path.write_text(json.dumps({
        "cdiVersion": "0.6.0", "kind": "nvidia.com/gpu", "devices": devices,
        "containerEdits": {
            "deviceNodes": [{"path": "/dev/null"}],
            "env": ["LD_PRELOAD=/opt/fake/preload.so", "LD_LIBRARY_PATH=/opt/fake"],
            "mounts": [
                {"hostPath": str(driver), "containerPath": "/opt/fake", "options": ["bind", "ro"]},
                {"hostPath": str(tmp_path / "fakegpu.conf"), "containerPath": "/etc/fake.conf", "options": ["bind", "ro"]},
            ],
        },
    }))
    return path


def run(tool, *args, spec, env=None, **kw):
    environ = dict(os.environ, FAKEGPU_CDI_SPEC=str(spec), **(env or {}))
    return subprocess.run([str(tool), *args], env=environ, capture_output=True, text=True, **kw)


# ---- nvidia-container-cli -----------------------------------------------------------

def test_cli_lists_device_nodes_and_driver(spec):
    out = run(CLI, "list", spec=spec, check=True).stdout.split()
    assert out[:3] == ["/dev/null", *NODES]
    assert any(p.endswith("/fakegpu/libcuda.so.1") for p in out)
    one = run(CLI, "list", f"--device={UUIDS[1]}", spec=spec, check=True).stdout.split()
    assert one == ["/dev/null", NODES[1]]


def test_cli_rejects_unknown_devices(spec):
    p = run(CLI, "list", "--device=7", spec=spec)
    assert p.returncode != 0 and "unknown device" in p.stderr


def test_cli_without_spec_reports_no_driver(tmp_path):
    p = run(CLI, "list", spec=tmp_path / "missing.json")
    assert p.returncode != 0 and "no CDI specification" in p.stderr


def test_cli_version():
    p = subprocess.run([str(CLI), "--version"], capture_output=True, text=True, check=True)
    assert p.stdout.startswith("cli-version: ")


@pytest.mark.skipif(not shutil.which("unshare"), reason="needs unshare")
def test_cli_configure_brings_gpus_into_a_rootfs(spec, tmp_path):
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    # As enroot's hook runs it: root in an unprivileged user namespace, in the
    # container's mount namespace; the check runs in the same namespaces.
    script = (f"{CLI} --user configure --no-cgroups --device=1 --compute --utility {rootfs} && "
              f"cat {rootfs}/opt/fake/libcuda.so.1 && ls {rootfs}/dev && "
              f"(echo x > {rootfs}/opt/fake/new 2>/dev/null && echo writable || echo read-only)")
    p = run("unshare", "--user", "--map-root-user", "--mount", "sh", "-c", script, spec=spec)
    if p.returncode != 0 and "unshare" in p.stderr:
        pytest.skip(f"no user namespaces here: {p.stderr.strip()}")
    assert p.returncode == 0, p.stderr
    lines = p.stdout.split()
    assert lines[0] == "fake"
    assert "full" in lines and "null" in lines and "zero" not in lines  # GPU 1 only, and nvidiactl
    assert lines[-1] == "read-only"


# ---- nvidia-container-runtime -------------------------------------------------------

@pytest.fixture
def bundle(tmp_path):
    def make(env):
        b = tmp_path / "bundle"
        b.mkdir(exist_ok=True)
        (b / "config.json").write_text(json.dumps({"process": {"env": env, "args": ["sh"]}, "mounts": []}))
        return b
    return make


def configured(bundle_dir):
    return json.loads((bundle_dir / "config.json").read_text())


def test_runtime_adds_the_step_gpus(spec, bundle):
    b = bundle(["PATH=/usr/bin", "SLURM_STEP_GPUS=1", "LD_PRELOAD=/user.so"])
    p = run(RUNTIME, "--rootless=true", "run", "--bundle", str(b), "c1", spec=spec,
            env={"FAKEGPU_OCI_RUNTIME": shutil.which("true")})
    assert p.returncode == 0, p.stderr
    config = configured(b)
    dests = [m["destination"] for m in config["mounts"]]
    assert "/dev/null" in dests and NODES[1] in dests and NODES[0] not in dests
    assert "/opt/fake" in dests and "/etc/fake.conf" in dests
    assert "LD_PRELOAD=/opt/fake/preload.so" in config["process"]["env"]
    assert "LD_PRELOAD=/user.so" not in config["process"]["env"]


def test_runtime_follows_nvidia_visible_devices(spec, bundle):
    b = bundle([f"NVIDIA_VISIBLE_DEVICES={UUIDS[0]}", "SLURM_STEP_GPUS=0,1"])
    run(RUNTIME, "create", "--bundle", str(b), "c1", spec=spec, env={"FAKEGPU_OCI_RUNTIME": shutil.which("true")})
    dests = [m["destination"] for m in configured(b)["mounts"]]
    assert NODES[0] in dests and NODES[1] not in dests


@pytest.mark.parametrize("env", [[], ["NVIDIA_VISIBLE_DEVICES=none"], ["NVIDIA_VISIBLE_DEVICES=void", "SLURM_STEP_GPUS=0"]])
def test_runtime_leaves_bundles_without_gpus_alone(spec, bundle, env):
    b = bundle(env)
    before = (b / "config.json").read_text()
    run(RUNTIME, "run", "--bundle", str(b), "c1", spec=spec, env={"FAKEGPU_OCI_RUNTIME": shutil.which("true")})
    assert (b / "config.json").read_text() == before


def test_runtime_passes_other_commands_through(spec, tmp_path):
    out = tmp_path / "args"
    fake = tmp_path / "crun"
    fake.write_text(f"#!/bin/sh\necho \"$@\" > {out}\n")
    fake.chmod(0o755)
    run(RUNTIME, "--root=/run/x", "state", "c1", spec=spec, env={"FAKEGPU_OCI_RUNTIME": str(fake)}, check=True)
    assert out.read_text().split() == ["--root=/run/x", "state", "c1"]
