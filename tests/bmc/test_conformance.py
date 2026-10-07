"""GB200 conformance: how NVIDIA's real GB200 BMCs and NVLink fabric behave,
from NVIDIA's own references, so tools written against the lab work on the
hardware. Part of make test-bmc; behaviour the lab does not model yet is
marked conformance and runs with --conformance (make test-bmc-conformance),
where a failure is a known gap, not a regression.

Sources, cited per test:
  [mock]   dsx-ai-factory/infra-controller crates/bmc-mock/src/hw/: NVIDIA's
           BMC models of real hardware (wiwynn_gb200_nvl.rs, nvidia_gb200.rs,
           nvidia_gbx00.rs, nvidia_switch_nd5200_ld.rs, nvidia_switch_n5700_ld.rs)
  [nico]   infra-controller docs/architecture/redfish/endpoints_reference.md:
           the Redfish fields NICo needs, with importance levels
  [sbmc]   NVIDIA Switch BMC User Manual 88.0060.2110, Redfish commands
           https://docs.nvidia.com/networking-ethernet-software/bmc-user-manual-88.0060.2110/User_Interface_Redfish_Commands.html
  [part]   NVIDIA GB200 NVL Partition User's Guide (DU-12143-001)
           https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf
  [mc]     NVIDIA Mission Control, NVLink Partition Management
           https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.2.0/nvlink-partition-management.html
  [smi]    nvidia-smi documentation, --gpu-reset
           https://docs.nvidia.com/deploy/nvidia-smi/
"""
import json
import subprocess
import time

import pytest

from conftest import CONTROLLER, REPO, ROOT, at, eventually, gpu_path, gpu_port_path, members, switch_port_path

# ---- compute tray BMC ----------------------------------------------------------------

def test_gpus_live_under_the_hgx_baseboard(lab, tray):
    """[mock] wiwynn_gb200_nvl.rs, [nico] "Systems/HGX_Baseboard_0/Processors":
    System_0 is the Grace host; the GPUs are behind the HMC, as processors of
    the HGX_Baseboard_0 system."""
    b = lab.tray_bmc(tray)
    systems = members(b.get(f"{ROOT}/Systems"))
    assert f"{ROOT}/Systems/HGX_Baseboard_0" in systems, systems
    gpus = members(b.get(f"{ROOT}/Systems/HGX_Baseboard_0/Processors"))
    assert {f"{ROOT}/Systems/HGX_Baseboard_0/Processors/GPU_{g}" for g in range(lab.gpus)} <= set(gpus)


def test_each_gpu_has_a_chassis_with_its_uuid(lab, tray):
    """[mock] nvidia_gb200.rs: "Hardware reports one UUID per GPU on the GPU's
    Processor, its enclosing Chassis and its PCIeDevice alike" (HGX_GPU_<n>)."""
    b = lab.tray_bmc(tray)
    chassis = members(b.get(f"{ROOT}/Chassis"))
    for g in range(lab.gpus):
        path = f"{ROOT}/Chassis/HGX_GPU_{g}"
        assert path in chassis, f"no {path}"
        assert b.get(path).get("UUID") == b.get(gpu_path(g))["UUID"]


def test_tray_has_host_bmc_and_hmc_managers(lab, tray):
    """[mock] wiwynn_gb200_nvl.rs manager_config: BMC_0 and HGX_BMC_0 (the HMC)."""
    mgrs = members(lab.tray_bmc(tray).get(f"{ROOT}/Managers"))
    assert {f"{ROOT}/Managers/BMC_0", f"{ROOT}/Managers/HGX_BMC_0"} <= set(mgrs), mgrs


def test_tray_identity_for_inventory_tools(lab, tray):
    """[nico]: System SerialNumber is Critical (machine id, pairing fails
    without it); Chassis_0/Assembly carries the GB200 serial."""
    b = lab.tray_bmc(tray)
    assert b.get(f"{ROOT}/Systems/System_0").get("SerialNumber"), "System_0 has no SerialNumber"
    assert at(b.get(f"{ROOT}/Chassis/Chassis_0"), "Assembly", "@odata.id"), "Chassis_0 has no Assembly"


def test_firmware_inventory(lab, bmc_name):
    """[nico] Required: UpdateService and FirmwareInventory with versioned components
    ([mock]: HGX_FW_BMC_0, FW_BMC_0, ... on trays; MGX_FW_* on the switch tray)."""
    b = lab.bmc(bmc_name)
    us = at(b.get(ROOT), "UpdateService", "@odata.id")
    assert us, "service root has no UpdateService"
    inv = members(b.get(at(b.get(us), "FirmwareInventory", "@odata.id")))
    assert inv, "empty FirmwareInventory"
    assert all(b.get(i).get("Version") for i in inv)


def test_compute_tray_reports_its_rack_slot(lab, tray):
    """[mock] nvidia_gbx00.rs cbc_chassis, [nico]: compute trays publish
    Oem.Nvidia ChassisPhysicalSlotNumber / ComputeTrayIndex / TopologyId on a
    CBC chassis; topology tools place the tray in the NVLink domain with them."""
    b = lab.tray_bmc(tray)
    oems = [at(b.get(c), "Oem", "Nvidia") or {} for c in members(b.get(f"{ROOT}/Chassis"))]
    assert any({"ChassisPhysicalSlotNumber", "ComputeTrayIndex", "TopologyId"} <= o.keys() for o in oems)


# ---- switch tray BMC -------------------------------------------------------------------

def test_switch_tray_identifies_as_nvswitch(lab):
    """[nico] site-explorer: a BMC is an NVLink switch tray when its chassis
    include MGX_NVSwitch_0; [sbmc], [mock]: MGX_NVSwitch_0/1, MGX_BMC_0, BMC_eeprom."""
    chassis = set(members(lab.bmc(lab.switch_bmc).get(f"{ROOT}/Chassis")))
    want = {f"{ROOT}/Chassis/{c}" for c in ("MGX_NVSwitch_0", "MGX_NVSwitch_1", "MGX_BMC_0", "BMC_eeprom")}
    assert want <= chassis, f"missing {sorted(want - chassis)}"


def test_switch_tray_cpu_is_a_system(lab):
    """[sbmc]: System_0 is the switch tray's CPU (NVOS host), with ComputerSystem.Reset."""
    sys = lab.bmc(lab.switch_bmc).get(f"{ROOT}/Systems/System_0")
    assert at(sys, "Actions", "#ComputerSystem.Reset", "target")


def test_switch_fabric_name(lab):
    """[sbmc]: the switch tray's fabric is MGX_NVLinkFabric_0
    (e.g. /redfish/v1/Fabrics/MGX_NVLinkFabric_0/Switches/NVSwitch_0)."""
    assert f"{ROOT}/Fabrics/MGX_NVLinkFabric_0" in members(lab.bmc(lab.switch_bmc).get(f"{ROOT}/Fabrics"))


# ---- NVLink behaviour across components ------------------------------------------------

def _find_switch_port(lab, tray, g, link):
    b = lab.bmc(lab.switch_bmc)
    for sw in range(lab.switches):
        for p in range(lab.switch_ports):
            r = at(b.get(switch_port_path(sw, p)), "Oem", "Nvidia", "RemoteEndpoint") or {}
            if (r.get("Host"), r.get("GPU"), r.get("Port")) == (tray, f"GPU_{g}", f"NVLink_{link}"):
                return sw, p
    pytest.fail(f"no switch port cabled to {tray} GPU_{g} NVLink_{link}")


@pytest.fixture
def downed_link(lab):
    """Take the switch end of trays[0] GPU_0 NVLink_0 down; restore afterwards."""
    tray, g, link = lab.trays[0], 0, 0
    sw, p = _find_switch_port(lab, tray, g, link)
    b = lab.bmc(lab.switch_bmc)
    yield lambda: b.write("PATCH", switch_port_path(sw, p), {"LinkState": "Disabled"}), tray, g, link
    b.write("PATCH", switch_port_path(sw, p), {"LinkState": "Enabled"})


def test_link_down_shows_at_both_ends(lab, downed_link):
    """A link has two ends: with the switch port down, the GPU's port on the
    tray BMC reports LinkDown too (NVLink training fails on both sides)."""
    down, tray, g, link = downed_link
    down()
    assert not lab.link_active(tray, g, link)
    assert lab.tray_bmc(tray).get(gpu_port_path(g, link))["LinkStatus"] == "LinkDown"


def _grpc(lab, method, **req):
    req.setdefault("gateway_id", "bmc-tests")
    p = subprocess.run([str(REPO / "bin/grpcurl"), "-plaintext", "-d", json.dumps(req),
                        f"{lab.ips[lab.nvswitch]}:{lab.vars['nmxc_port']}", f"nmxlab.v1.NMXController/{method}"],
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, f"{method}: {p.stderr.strip()}"
    return json.loads(p.stdout or "{}")


def gpu_reset(lab, tray, ids=None):
    """nvidia-smi --gpu-reset on the tray as root; its output."""
    return lab.sh(tray, "nvidia-smi --gpu-reset" + (f" -i {ids}" if ids is not None else ""))


def recovery_actions(lab, tray):
    out = lab.sh(tray, "nvidia-smi -q")
    return [line.split(":", 1)[1].strip() for line in out.splitlines() if "GPU Recovery Action" in line]


@pytest.fixture
def tray_partition(lab):
    """The last tray's GPUs in their own NVLink partition (id 7), not yet
    reset; back in the default partition, and reset, afterwards."""
    _grpc(lab, "Hello")
    parts = _grpc(lab, "GetPartitionInfoList").get("partitions", [])
    if [p["partitionId"] for p in parts] != [32766]:
        pytest.skip(f"partitions other than the default exist: {[p['partitionId'] for p in parts]}")
    tray = lab.trays[-1]
    if lab.sh(tray, "nvidia-smi --query-compute-apps=pid --format=csv,noheader").strip():
        pytest.skip(f"{tray} has GPU processes; its GPUs cannot be reset")
    uids = [g["gpuUid"] for g in _grpc(lab, "GetGpuInfoList")["gpus"] if g["hostname"] == tray]
    _grpc(lab, "RemoveGpusFromPartition", partition_id=32766, gpu_uids=uids)
    _grpc(lab, "CreatePartition", partition_name="bmc-test", partition_id=7, gpu_uids=uids)
    yield tray, 7
    _grpc(lab, "DeletePartition", partition_id=7)
    _grpc(lab, "AddGpusToPartition", partition_id=32766, gpu_uids=uids)
    gpu_reset(lab, tray)


def test_partition_change_waits_for_gpu_reset(lab, tray_partition):
    """[mc]: "We need to reset the GPUs (or reboot the nodes) in order for the
    Clique ID to update"; [part] 4.1: reset a GPU added to a partition (clears
    its registers and memory, for tenant handover); 10.4: GPU Recovery Action."""
    tray, pid = tray_partition
    assert all(g["clique"] != pid for g in lab.smi(tray)), "clique changed without a GPU reset"
    assert set(recovery_actions(lab, tray)) == {"GPU_RESET"}
    info = [g for g in _grpc(lab, "GetGpuInfoList")["gpus"] if g["hostname"] == tray]
    assert all(g.get("resetPending") for g in info), "the partition controller does not report the pending reset"

    assert "All done." in gpu_reset(lab, tray)
    assert all(g["clique"] == pid for g in lab.smi(tray))
    assert set(recovery_actions(lab, tray)) == {"None"}


def test_bmc_reports_the_partition_clique(lab, tray_partition):
    """[part] 10.3: the GPU's CliqueId is its NVLink partition. The BMC's view
    of the GPU follows it as NVML does: after the GPU reset."""
    tray, pid = tray_partition
    gpu_reset(lab, tray)
    assert all(g["clique"] == pid for g in lab.smi(tray))
    for g in range(lab.gpus):
        assert at(lab.tray_bmc(tray).get(gpu_path(g)), "Oem", "Nvidia", "FabricClique", "CliqueId") == pid


def test_gpu_reset_refuses_a_gpu_in_use(lab):
    """[smi]: --gpu-reset needs root and no process using the GPU."""
    tray, gpu = lab.trays[0], 3
    apps = f"nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader"
    if lab.sh(tray, apps).strip():
        pytest.skip(f"{tray} GPU {gpu} is in use")
    hold = ("import ctypes, time; cu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0); "
            "d, c, p = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64(); "
            f"cu.cuDeviceGet(ctypes.byref(d), {gpu}); cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), d); "
            "cu.cuCtxSetCurrent(c); cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(1 << 30)); time.sleep(60)")
    lab.sh(tray, f'systemd-run --unit=bmc-gpu-hold --collect python3 -c "{hold}"')
    try:
        eventually(lambda: not lab.sh(tray, apps).strip() and "holder not on the GPU yet")
        assert lab.sh(tray, f"nvidia-smi --gpu-reset -i {gpu}", check=False) is None, "reset a GPU in use"
    finally:
        lab.sh(tray, "systemctl stop bmc-gpu-hold", check=False)
    eventually(lambda: lab.sh(tray, f"nvidia-smi --gpu-reset -i {gpu}", check=False) is None
               and "reset still refused after the process ended", timeout=20)
    assert lab.sh(tray, f"su -s /bin/sh nobody -c 'nvidia-smi --gpu-reset -i {gpu}'", check=False) is None, \
        "a non-root user reset a GPU"


def _squeue_state(lab, job):
    out = lab.sh(CONTROLLER, f"sacct -n -X -P -j {job} -o State", check=False) or ""
    return out.strip().split("\n")[0].split()[0] if out.strip() else "UNKNOWN"


@pytest.mark.conformance
def test_access_link_down_fails_running_job(lab, downed_link):
    """[part] 6.2: "An access link failure causes the GPU in the partition to
    lose NVLink connectivity. This causes the workload in the partition to run
    into errors." A DDP job across the domain must fail when one of its GPUs
    loses an NVLink."""
    if lab.scheduler != "slurm":
        pytest.skip("submits through Slurm (sbatch/sacct)")
    if lab.sh("sched-login", "test -x /shared/venv/bin/python", check=False) is None:
        pytest.skip("needs the frameworks venv (make frameworks)")
    if lab.sh(CONTROLLER, "squeue -h").strip():
        pytest.skip("jobs are running; this test needs the whole domain")
    down, tray, g, link = downed_link
    job = lab.sh("sched-login", "su - joe -c 'cd examples/slurm && sbatch --parsable ddp-train.sbatch --steps 1000000'").strip()
    try:
        eventually(lambda: (s := _squeue_state(lab, job)) != "RUNNING" and f"job {job} is {s}", timeout=120, interval=3)
        time.sleep(30)  # past NCCL initialisation, into training
        down()
        eventually(lambda: _squeue_state(lab, job) == "RUNNING" and f"job {job} still running with {tray} "
                   f"GPU {g} NVLink {link} down", timeout=90, interval=5)
        assert _squeue_state(lab, job) == "FAILED"
    finally:
        lab.sh(CONTROLLER, f"scancel {job}", check=False)
