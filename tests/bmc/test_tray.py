"""GPU compute tray BMCs: what they report must match what the tray's OS sees
(nvidia-smi), and NVLink port changes follow the Redfish settings model of
NVIDIA's bmcweb: LinkState written to the port's Settings resource takes effect
at the next tray reset, never at once (the reset itself is in test_power)."""
import pytest

from conftest import ROOT, at, gpu_path, gpu_port_path, members, redfish_message_id


def test_inventory_matches_the_os(lab, tray):
    b = lab.tray_bmc(tray)
    sys = b.get(f"{ROOT}/Systems/System_0")
    assert sys["PowerState"] == "On"
    assert sys["ProcessorSummary"]["Count"] == lab.gpus
    assert len(members(b.get(f"{ROOT}/Systems/System_0/Processors"))) == lab.gpus

    smi = lab.smi(tray)
    assert len(smi) == lab.gpus
    for g, os in enumerate(smi):
        p = b.get(gpu_path(g))
        assert p["ProcessorType"] == "GPU"
        assert ("GPU-" + p["UUID"]).lower() == os["uuid"].lower()
        assert p["SerialNumber"] == os["serial"]
        assert at(p, "Oem", "Nvidia", "PCIeBusId") == os["bus_id"]
        assert at(p, "Oem", "Nvidia", "FabricClique", "ClusterUUID") == os["cluster_uuid"]
        assert at(p, "Oem", "Nvidia", "FabricClique", "CliqueId") == os["clique"]
        assert at(p, "MemorySummary", "TotalMemoryGiB") == os["mem_mib"] // 1024


def test_nvlink_ports(lab, tray):
    """Blackwell GPUs have 18 NVLink5 links, all to NVSwitches."""
    b = lab.tray_bmc(tray)
    inactive = lab.inactive_links(tray)
    for g in range(lab.gpus):
        assert len(members(b.get(f"{gpu_path(g)}/Ports"))) == lab.nvlinks
        for link in range(lab.nvlinks):
            p = b.get(gpu_port_path(g, link))
            assert p["PortProtocol"] == "NVLink"
            assert at(p, "@Redfish.Settings", "SettingsObject", "@odata.id") == gpu_port_path(g, link) + "/Settings"
            if p["LinkState"] == "Disabled":
                assert link in inactive.get(g, set()), f"GPU_{g} NVLink_{link}: disabled by the BMC, active in nvidia-smi"


@pytest.fixture
def idle_port(lab):
    """GPU_1 NVLink_7 on the first tray, enabled, with no change left pending afterwards."""
    tray, g, link = lab.trays[0], 1, 7
    b = lab.tray_bmc(tray)
    if b.get(gpu_port_path(g, link))["LinkState"] != "Enabled":
        pytest.skip(f"{tray} GPU_{g} NVLink_{link} is not enabled")
    yield b, tray, g, link
    b.write("PATCH", gpu_port_path(g, link) + "/Settings", {"LinkState": "Enabled"})


def test_link_change_is_pending_until_reset(lab, idle_port):
    b, tray, g, link = idle_port
    settings = gpu_port_path(g, link) + "/Settings"
    b.write("PATCH", settings, {"LinkState": "Disabled"})
    assert b.get(settings)["LinkState"] == "Disabled"
    assert b.get(gpu_port_path(g, link))["LinkState"] == "Enabled", "applied before the tray reset"
    assert lab.link_active(tray, g, link), "link went down before the tray reset"

    # Requesting the current state again cancels the pending change.
    b.write("PATCH", settings, {"LinkState": "Enabled"})
    assert b.get(settings)["LinkState"] == "Enabled"


def test_sticky_flag_round_trips(lab):
    b = lab.tray_bmc(lab.trays[0])
    port = gpu_port_path(0, 3)
    sticky = lambda: at(b.get(port), "Oem", "Nvidia", "LinkDisableSticky")  # noqa: E731
    if sticky() is not False:
        pytest.skip("LinkDisableSticky already set")
    try:
        b.write("PATCH", port, {"Oem": {"Nvidia": {"LinkDisableSticky": True}}})
        assert sticky() is True
        assert b.get(port)["LinkState"] == "Enabled", "the flag alone took the link down"
    finally:
        b.write("PATCH", port, {"Oem": {"Nvidia": {"LinkDisableSticky": False}}})


@pytest.mark.parametrize("method,path,body,status,message", [
    ("GET", gpu_path(99), None, 404, "ResourceNotFound"),
    ("GET", gpu_port_path(0, 18), None, 404, "ResourceNotFound"),
    ("GET", f"{ROOT}/Systems/System_9", None, 404, "ResourceNotFound"),
    ("PATCH", gpu_port_path(0, 0) + "/Settings", {"LinkState": "Down"}, 400, "PropertyValueNotInList"),
    ("PATCH", gpu_port_path(0, 0), {"LinkState": "Disabled"}, 400, "PropertyNotWritable"),
    ("POST", f"{ROOT}/Systems/System_0/Actions/ComputerSystem.Reset", {"ResetType": "Nmi"}, 400,
     "ActionParameterValueNotInList"),
])
def test_rejects_invalid_requests(lab, method, path, body, status, message):
    r = lab.tray_bmc(lab.trays[0]).request(method, path, body)
    assert r.status_code == status
    assert redfish_message_id(r).endswith("." + message)


def test_advertised_reset_types(lab, tray):
    sys = lab.tray_bmc(tray).get(f"{ROOT}/Systems/System_0")
    allowed = set(at(sys, "Actions", "#ComputerSystem.Reset", "ResetType@Redfish.AllowableValues"))
    assert {"On", "ForceOff", "GracefulShutdown", "GracefulRestart", "ForceRestart", "PowerCycle"} <= allowed
