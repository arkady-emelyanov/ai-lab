"""Power actions on a GPU tray (--disruptive). Like an operator, each test
drains the tray in Slurm first and refuses to touch a tray running jobs; it
uses the last tray so the first keeps serving.

Real-hardware behaviour checked: NVLink Settings apply at the next reset; a
plain disable lasts one reset, a sticky one until cleared
(Oem.Nvidia.LinkDisableSticky, NvidiaPort schema); a powered-off tray shows its
NVLinks down; Slurm takes the tray back once it is powered on."""
import pytest

from conftest import CONTROLLER, ROOT, eventually, gpu_port_path

pytestmark = pytest.mark.disruptive


def node_state(lab, tray):
    out = lab.sh(CONTROLLER, f"sinfo -h -N -n {tray} -o %T", check=False)
    return (out or "unknown").strip()


@pytest.fixture
def power_tray(lab):
    tray = lab.trays[-1]
    jobs = lab.sh(CONTROLLER, f"squeue -h -w {tray}").strip()
    if jobs:
        pytest.fail(f"{tray} is running jobs; not power-cycling it:\n{jobs}")
    lab.sh(CONTROLLER, f"scontrol update nodename={tray} state=drain reason=bmc-power-test")
    yield tray

    def back():
        lab.sh(CONTROLLER, f"scontrol update nodename={tray} state=resume", check=False)
        st = node_state(lab, tray)
        return st != "idle" and f"{tray} is {st} in Slurm"
    eventually(back, timeout=180, interval=5)


def reset(lab, tray, reset_type):
    b = lab.tray_bmc(tray)
    b.write("POST", f"{ROOT}/Systems/System_0/Actions/ComputerSystem.Reset", {"ResetType": reset_type})
    if reset_type in ("ForceOff", "GracefulShutdown"):
        return

    def up():
        if (s := b.get(f"{ROOT}/Systems/System_0")["PowerState"]) != "On":
            return f"PowerState {s}"
        return lab.sh(tray, "nvidia-smi -L", check=False) is None and "tray OS not up"
    eventually(up, timeout=180, interval=2)


def set_link(lab, tray, g, link, state):
    lab.tray_bmc(tray).write("PATCH", gpu_port_path(g, link) + "/Settings", {"LinkState": state})


def set_sticky(lab, tray, g, link, sticky):
    lab.tray_bmc(tray).write("PATCH", gpu_port_path(g, link), {"Oem": {"Nvidia": {"LinkDisableSticky": sticky}}})


def assert_link(lab, tray, g, link, up):
    """The link state at every layer that shows it: BMC port, nvidia-smi, fabric telemetry."""
    assert lab.tray_bmc(tray).get(gpu_port_path(g, link))["LinkState"] == ("Enabled" if up else "Disabled")
    assert lab.link_active(tray, g, link) == up
    assert lab.metric("nvlink_gpu_active_links", host=tray, gpu=str(g)) == lab.nvlinks - (0 if up else 1)


def test_link_disable_applies_at_reset_and_lasts_one_reset(lab, power_tray):
    tray, g, link = power_tray, 2, 4
    try:
        set_link(lab, tray, g, link, "Disabled")
        reset(lab, tray, "ForceRestart")
        assert_link(lab, tray, g, link, up=False)
        assert lab.tray_bmc(tray).get(gpu_port_path(g, link) + "/Settings")["LinkState"] == "Disabled"

        reset(lab, tray, "ForceRestart")  # not requested again: the link comes back
        assert_link(lab, tray, g, link, up=True)
    finally:
        set_link(lab, tray, g, link, "Enabled")
        if lab.tray_bmc(tray).get(gpu_port_path(g, link))["LinkState"] != "Enabled":
            reset(lab, tray, "ForceRestart")


def test_sticky_link_disable_survives_resets(lab, power_tray):
    tray, g, link = power_tray, 3, 11
    try:
        set_sticky(lab, tray, g, link, True)
        set_link(lab, tray, g, link, "Disabled")
        reset(lab, tray, "ForceRestart")
        assert_link(lab, tray, g, link, up=False)
        reset(lab, tray, "PowerCycle")
        assert_link(lab, tray, g, link, up=False)
    finally:
        set_sticky(lab, tray, g, link, False)
        set_link(lab, tray, g, link, "Enabled")
        reset(lab, tray, "ForceRestart")
    assert_link(lab, tray, g, link, up=True)


def test_force_off_and_on(lab, power_tray):
    tray, b = power_tray, lab.tray_bmc(power_tray)
    try:
        reset(lab, tray, "ForceOff")
        assert b.get(f"{ROOT}/Systems/System_0")["PowerState"] == "Off"
        assert b.get(f"{ROOT}/Chassis/Chassis_0")["PowerState"] == "Off"
        assert b.get(gpu_port_path(0, 0))["LinkStatus"] == "LinkDown"
        assert lab.sh(tray, "true", check=False) is None, "tray OS still answers after ForceOff"
        assert b.request("GET", ROOT, auth=False).status_code == 200, "the BMC must stay up with the host off"
    finally:
        if b.get(f"{ROOT}/Systems/System_0")["PowerState"] != "On":
            reset(lab, tray, "On")
    # Drained before the test, so back as drained: slurmd re-registered.
    eventually(lambda: (st := node_state(lab, tray)) != "drained" and f"Slurm state {st}", timeout=120, interval=5)


def test_power_on_when_on_is_a_noop(lab, power_tray):
    uptime = lambda: float(lab.sh(power_tray, "cut -d' ' -f1 /proc/uptime"))  # noqa: E731
    before = uptime()
    lab.tray_bmc(power_tray).write("POST", f"{ROOT}/Systems/System_0/Actions/ComputerSystem.Reset", {"ResetType": "On"})
    assert uptime() >= before, "the tray restarted on ResetType On"
