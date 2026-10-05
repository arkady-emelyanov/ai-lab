"""Integration tests for the lab's BMCs, run from the host against the live
cluster (make test-bmc). They talk Redfish to the BMCs the way a management
client does and check the effect where real hardware shows it: nvidia-smi on
the tray, the partition controller's fabric telemetry, Slurm.

Tiers:
  default         read-only checks and reversible NVLink changes; every change
                  is undone by its fixture
  --disruptive    power actions on a GPU tray (refuses a tray running jobs)
  --conformance   behaviour of NVIDIA's GB200 BMCs and NVLink fabric that the
                  emulation may not reproduce yet; failures list the gaps

The tests share one cluster: run them serially (no xdist).
"""
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import requests

REPO = Path(__file__).resolve().parents[2]
ROOT = "/redfish/v1"
CONTROLLER = "sched-control"


def pytest_addoption(parser):
    parser.addoption("--disruptive", action="store_true", help="run tests that power-cycle a GPU tray")
    parser.addoption("--conformance", action="store_true",
                     help="run GB200 conformance tests (real-hardware behaviour the emulation may lack)")


def pytest_configure(config):
    config.addinivalue_line("markers", "disruptive: power-cycles a GPU tray (--disruptive)")
    config.addinivalue_line("markers", "conformance: real GB200 behaviour the emulation may lack (--conformance)")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for tier in ("disruptive", "conformance"):
            if tier in item.keywords and not config.getoption(tier):
                item.add_marker(pytest.mark.skip(reason=f"needs --{tier} (make test-bmc-{tier})"))


# ---- Redfish client ---------------------------------------------------------------

class RedfishError(AssertionError):
    pass


class BMC:
    def __init__(self, name, address, user, password):
        self.name, self.base = name, f"https://{address}"
        self.user, self.password = user, password
        self.token = None  # X-Auth-Token instead of basic auth when set

    def request(self, method, path, body=None, auth=True, **kw):
        headers = kw.pop("headers", {})
        if auth and self.token:
            headers["X-Auth-Token"] = self.token
        return requests.request(method, self.base + path, json=body, headers=headers, verify=False, timeout=120,
                                auth=(self.user, self.password) if auth and not self.token else None, **kw)

    def get(self, path):
        """The resource; raises unless the BMC answers 200 with a JSON object."""
        r = self.request("GET", path)
        if r.status_code != 200:
            raise RedfishError(f"{self.name} GET {path}: {r.status_code} {r.text[:300]}")
        return r.json()

    def write(self, method, path, body=None, **kw):
        """PATCH/POST/DELETE; raises unless the status is 2xx."""
        r = self.request(method, path, body, **kw)
        if not 200 <= r.status_code < 300:
            raise RedfishError(f"{self.name} {method} {path}: {r.status_code} {r.text[:300]}")
        return r

    def login(self):
        r = self.request("POST", f"{ROOT}/SessionService/Sessions",
                         {"UserName": self.user, "Password": self.password}, auth=False)
        assert r.status_code == 201, f"login: {r.status_code} {r.text[:300]}"
        self.token = r.headers["X-Auth-Token"]
        return r


def at(obj, *keys):
    """at(o, "Oem", "Nvidia", "LinkDisableSticky"): nested lookup, None if absent."""
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


def members(collection):
    return [m["@odata.id"] for m in collection.get("Members", [])]


def redfish_message_id(r):
    """MessageId of the first @Message.ExtendedInfo entry of an error response."""
    try:
        return r.json()["error"]["@Message.ExtendedInfo"][0]["MessageId"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def gpu_path(g):
    return f"{ROOT}/Systems/System_0/Processors/GPU_{g}"


def gpu_port_path(g, link):
    return f"{gpu_path(g)}/Ports/NVLink_{link}"


def switch_path(sw):
    return f"{ROOT}/Fabrics/NVLinkFabric_0/Switches/NVSwitch_{sw}"


def switch_port_path(sw, port):
    return f"{switch_path(sw)}/Ports/NVLink_{port}"


def eventually(check, timeout=10, interval=1):
    """Retry check() until it returns a falsy value; it returns a message while
    the condition does not hold yet."""
    deadline = time.monotonic() + timeout
    while True:
        msg = check()
        if not msg:
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"after {timeout}s: {msg}")
        time.sleep(interval)


# ---- the lab ------------------------------------------------------------------------

@dataclass
class Lab:
    ips: dict
    vars: dict
    trays: list = field(default_factory=list)  # GPU trays in cabling order
    switch_bmc: str = "sched-nvswitch-bmc"
    nvswitch: str = "sched-nvswitch"  # partition controller host

    def __post_init__(self):
        v = self.vars
        self.user, self.password = v["bmc_username"], v["bmc_password"]
        self.gpus, self.nvlinks = int(v["fakegpu_count"]), int(v["fakegpu_nvlinks"])
        self.switches, self.switch_ports = int(v["nvswitch_count"]), int(v["nvswitch_ports"])
        self.scheduler = v.get("scheduler", "slurm")

    def bmc(self, name):
        return BMC(name, self.ips[name], self.user, self.password)

    def tray_bmc(self, tray):
        return self.bmc(f"{tray}-bmc")

    def all_bmcs(self):
        return [f"{t}-bmc" for t in self.trays] + [self.switch_bmc]

    # -- the trays as the OS sees them --

    def sh(self, host, cmd, check=True):
        """Run a command as root on an instance (bin/ssh); stdout."""
        p = subprocess.run([str(REPO / "bin/ssh"), "-o", "ConnectTimeout=5", f"root@{host}", cmd],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        if check and p.returncode != 0:
            raise AssertionError(f"{host}: {cmd}: exit {p.returncode}: {p.stderr.strip()[:300]}")
        return p.stdout if p.returncode == 0 else None

    def smi(self, tray):
        """Per GPU: uuid, serial, bus_id, mem_mib, cluster_uuid, clique (nvidia-smi)."""
        out = self.sh(tray, "nvidia-smi --query-gpu=uuid,serial,pci.bus_id,memory.total,"
                            "fabric.clusterUuid,fabric.cliqueId --format=csv,noheader,nounits")
        gpus = []
        for line in out.strip().splitlines():
            uuid, serial, bus, mem, cluster, clique = (f.strip() for f in line.split(","))
            gpus.append(dict(uuid=uuid, serial=serial, bus_id=bus, mem_mib=int(mem),
                             cluster_uuid=cluster, clique=int(clique)))
        return gpus

    def inactive_links(self, tray):
        """{gpu: {links nvidia-smi reports <inactive>}}"""
        res, gpu = {}, None
        for line in self.sh(tray, "nvidia-smi nvlink -s").splitlines():
            if m := re.match(r"GPU (\d+):", line):
                gpu = int(m[1])
            elif (m := re.match(r"\s*Link (\d+): <inactive>", line)) and gpu is not None:
                res.setdefault(gpu, set()).add(int(m[1]))
        return res

    def link_active(self, tray, g, link):
        return link not in self.inactive_links(tray).get(g, set())

    def metrics(self):
        """Fabric telemetry of the partition controller: [(name, labels, value)]."""
        url = f"http://{self.ips[self.nvswitch]}:{self.vars['nmxc_metrics_port']}/metrics"
        samples = []
        for line in requests.get(url, timeout=10).text.splitlines():
            if m := re.match(r"^(\w+)(?:\{(.*)\})? (\S+)$", line):
                labels = dict(re.findall(r'(\w+)="([^"]*)"', m[2] or ""))
                samples.append((m[1], labels, float(m[3])))
        return samples

    def metric(self, name, **labels):
        """Value of the one series of name whose labels include labels."""
        found = [v for n, l, v in self.metrics() if n == name and labels.items() <= l.items()]
        assert len(found) == 1, f"{name}{labels}: {len(found)} matching series"
        return found[0]


def _read_vars():
    """Top-level scalars of inventory/group_vars/all.yml (all the tests need)."""
    out = {}
    for line in (REPO / "inventory/group_vars/all.yml").read_text().splitlines():
        if m := re.match(r"^([a-z_0-9]+):\s*([^#{\[\s]\S*)", line):
            out[m[1]] = m[2].strip("\"'")
    return out


def _read_instances():
    cmd = ["incus", "list", "--all-projects", "--format", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if p.returncode != 0:
        p = subprocess.run(["sg", "incus-admin", "-c", " ".join(cmd)], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, check=True)
    ips = {}
    for inst in json.loads(p.stdout):
        if inst["status"] != "Running":
            continue
        for a in ((inst.get("state") or {}).get("network") or {}).get("eth0", {}).get("addresses", []):
            if a["family"] == "inet":
                ips[inst["name"]] = a["address"]
    return ips


_LAB = None


def _lab():
    global _LAB
    if _LAB is None:
        ips = _read_instances()
        _LAB = Lab(ips=ips, vars=_read_vars())
        # sched-worker1, sched-worker2: the order the switch tray is cabled in
        _LAB.trays = sorted(n.removesuffix("-bmc") for n in ips if n.endswith("-bmc") and n != _LAB.switch_bmc)
        if not _LAB.trays or _LAB.switch_bmc not in ips or _LAB.nvswitch not in ips:
            pytest.exit("the lab is not running (make up)", returncode=2)
    return _LAB


@pytest.fixture(scope="session")
def lab():
    return _lab()


def pytest_generate_tests(metafunc):
    # bmc_name: every BMC in the lab; tray: every GPU tray.
    if "bmc_name" in metafunc.fixturenames:
        metafunc.parametrize("bmc_name", _lab().all_bmcs())
    if "tray" in metafunc.fixturenames:
        metafunc.parametrize("tray", _lab().trays)
