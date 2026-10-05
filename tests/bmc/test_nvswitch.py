"""NVLink switch tray BMC: the fabric, its NVSwitch chips and ports, and the
cabling to the GPU trays. A switch port taken down is an access link down
(NVIDIA GB200 NVL Partition User's Guide, section 6.2): the GPU at the other
end loses that link at once, and the fabric telemetry shows it."""
import re
from collections import Counter

import pytest

from conftest import ROOT, at, eventually, gpu_path, members, redfish_message_id, switch_path, switch_port_path

FABRIC = f"{ROOT}/Fabrics/NVLinkFabric_0"


def test_fabric_shape(lab):
    b = lab.bmc(lab.switch_bmc)
    fabrics = members(b.get(f"{ROOT}/Fabrics"))
    assert len(fabrics) == 1
    f = b.get(fabrics[0])
    assert f["FabricType"] == "NVLink"
    assert len(members(b.get(f["Switches"]["@odata.id"]))) == lab.switches
    for sw in range(lab.switches):
        s = b.get(switch_path(sw))
        assert s["SwitchType"] == "NVLink"
        assert s["TotalSwitchWidth"] == lab.switch_ports
        assert len(members(b.get(switch_path(sw) + "/Ports"))) == lab.switch_ports


def test_cabling_is_consistent(lab):
    """Every GPU link is cabled to exactly one switch port, each GPU spreads its
    links evenly over the switches, and the switch side names each GPU by the
    UUID its own tray BMC reports."""
    b = lab.bmc(lab.switch_bmc)
    uuids = {f"{t}/GPU_{g}": lab.tray_bmc(t).get(gpu_path(g))["UUID"] for t in lab.trays for g in range(lab.gpus)}
    ends, per_switch, problems = {}, Counter(), []
    for sw in range(lab.switches):
        for p in range(lab.switch_ports):
            remote = at(b.get(switch_port_path(sw, p)), "Oem", "Nvidia", "RemoteEndpoint")
            if not remote:
                continue  # not cabled
            gpu = f"{remote['Host']}/{remote['GPU']}"
            end = (gpu, remote["Port"])
            if end in ends:
                problems.append(f"{end} cabled to {ends[end]} and NVSwitch_{sw} port {p}")
            ends[end] = f"NVSwitch_{sw} port {p}"
            if uuids.get(gpu) != remote.get("GPUUUID"):
                problems.append(f"NVSwitch_{sw} port {p}: {gpu} UUID {remote.get('GPUUUID')}, tray BMC {uuids.get(gpu)}")
            per_switch[gpu, sw] += 1
    assert not problems, "\n".join(problems)
    assert len(ends) == len(lab.trays) * lab.gpus * lab.nvlinks
    uneven = {k: n for k, n in per_switch.items() if n != lab.nvlinks // lab.switches}
    assert not uneven, f"links per (GPU, switch): {uneven}"


@pytest.fixture
def switch_port(lab):
    """NVSwitch_1 port 20 and the GPU link behind it; enabled again afterwards."""
    b, sw, p = lab.bmc(lab.switch_bmc), 1, 20
    port = b.get(switch_port_path(sw, p))
    if port["LinkState"] != "Enabled":
        pytest.skip(f"NVSwitch_{sw} port {p} is {port['LinkState']}")
    remote = at(port, "Oem", "Nvidia", "RemoteEndpoint")
    g, link = int(remote["GPU"].removeprefix("GPU_")), int(remote["Port"].removeprefix("NVLink_"))
    yield b, sw, p, remote["Host"], g, link
    b.write("PATCH", switch_port_path(sw, p), {"LinkState": "Enabled"})
    eventually(lambda: not lab.link_active(remote["Host"], g, link) and "link still inactive after re-enabling")


def test_switch_port_down_reaches_gpu_and_telemetry(lab, switch_port):
    """Port down -> the GPU loses that link now, the switch reports it, the
    partition controller's telemetry marks the GPU unhealthy; re-enabling
    restores all of them."""
    b, sw, p, host, g, link = switch_port
    b.write("PATCH", switch_port_path(sw, p), {"LinkState": "Disabled"})
    port = b.get(switch_port_path(sw, p))
    assert (port["LinkState"], port["LinkStatus"]) == ("Disabled", "LinkDown")
    eventually(lambda: lab.link_active(host, g, link) and f"{host} GPU {g} link {link} still active in nvidia-smi")
    assert lab.metric("nvswitch_port_up", switch=f"NVSwitch_{sw}", port=str(p)) == 0
    assert lab.metric("nvlink_gpu_active_links", host=host, gpu=str(g)) == lab.nvlinks - 1
    assert lab.metric("nvlink_gpu_healthy", host=host, gpu=str(g)) == 0
    assert b.get(switch_path(sw))["Status"]["Health"] != "OK"

    b.write("PATCH", switch_port_path(sw, p), {"LinkState": "Enabled"})
    assert lab.metric("nvlink_gpu_healthy", host=host, gpu=str(g)) == 1


def test_switch_settings_round_trip(lab):
    b, path = lab.bmc(lab.switch_bmc), switch_path(0)
    mode = lambda: at(b.get(path), "Oem", "Nvidia", "SwitchIsolationMode")  # noqa: E731
    orig = mode()
    flip = {"SwitchCommunicationEnabled": "SwitchCommunicationDisabled",
            "SwitchCommunicationDisabled": "SwitchCommunicationEnabled"}[orig]
    try:
        b.write("PATCH", path, {"Oem": {"Nvidia": {"SwitchIsolationMode": flip}}})
        assert mode() == flip
        r = b.request("PATCH", path, {"Oem": {"Nvidia": {"SwitchIsolationMode": "Isolated"}}})
        assert r.status_code == 400 and redfish_message_id(r).endswith(".PropertyValueNotInList")
        assert b.request("PATCH", switch_port_path(0, 0), {"LinkState": "Off"}).status_code == 400
    finally:
        b.write("PATCH", path, {"Oem": {"Nvidia": {"SwitchIsolationMode": orig}}})


def test_switch_config_upload(lab):
    """Configuration push: multipart with one ImportFile part, to the fabric's
    Oem.Nvidia.SwitchConfigPushURI."""
    b = lab.bmc(lab.switch_bmc)
    f = b.get(FABRIC)
    if at(f, "Oem", "Nvidia", "SwitchConfig"):
        pytest.skip("a switch configuration is already uploaded; not replacing it")
    uri = at(f, "Oem", "Nvidia", "SwitchConfigPushURI")
    assert uri, "fabric has no Oem.Nvidia.SwitchConfigPushURI"

    assert b.request("POST", uri, files={"Other": ("switch.conf", b"x")}).status_code == 400
    data = b"# lab switch configuration\nnvl partition default\n"
    try:
        b.write("POST", uri, files={"ImportFile": ("switch.conf", data)})
        cfg = at(b.get(FABRIC), "Oem", "Nvidia", "SwitchConfig")
        assert cfg and cfg["FileName"] == "switch.conf" and cfg["SizeBytes"] == len(data)
    finally:
        b.request("DELETE", uri)
    assert not at(b.get(FABRIC), "Oem", "Nvidia", "SwitchConfig"), "still shown after DELETE"


def test_switch_ids_are_valid(lab):
    for path in (f"{ROOT}/Fabrics/NVLinkFabric_0/Switches/NVSwitch_{lab.switches}",
                 switch_port_path(0, lab.switch_ports), f"{ROOT}/Fabrics/Other"):
        assert lab.bmc(lab.switch_bmc).request("GET", path).status_code == 404, path
    assert re.fullmatch(r"NVSwitch_\d+", lab.bmc(lab.switch_bmc).get(switch_path(0))["Id"])
