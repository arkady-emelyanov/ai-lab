"""GPU sensors on the tray BMCs: temperature and power read out of band must
be what NVML reports in band (nvidia-smi and the GPU exporter read NVML). The
BMC computes them with the fake NVML's model from the same shared state, so
the only difference allowed is time: on the tray, each BMC reading is taken
between two NVML readings a few milliseconds apart and must lie between them."""
import contextlib
import json

import pytest

from conftest import ROOT, eventually, gpu_path, members

TRAY = f"{ROOT}/Chassis/Chassis_0"


def gpu_sensors(g):
    """A GPU's sensors, on its own chassis (HGX_GPU_<n>), as on a real GB200 tray."""
    return f"{ROOT}/Chassis/HGX_GPU_{g}/Sensors"
POWER_TOL, TEMP_TOL = 0.05, 0  # W, whole degrees

# Runs on the tray: NVML, BMC, NVML for each GPU.
COMPARE = """
import base64, ctypes, json, ssl, sys, urllib.request
nvml = ctypes.CDLL("libnvidia-ml.so.1"); nvml.nvmlInit_v2()
bmc, auth, gpus = sys.argv[1], base64.b64encode(sys.argv[2].encode()).decode(), int(sys.argv[3])
tls = ssl._create_unverified_context()
def nv(g):
    h, t, p = ctypes.c_void_p(), ctypes.c_uint(), ctypes.c_uint()
    nvml.nvmlDeviceGetHandleByIndex_v2(g, ctypes.byref(h))
    nvml.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t)); nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(p))
    return t.value, p.value / 1000
def env(g):
    req = urllib.request.Request(f"https://{bmc}/redfish/v1/Systems/HGX_Baseboard_0/Processors/GPU_{g}/EnvironmentMetrics",
                                 headers={"Authorization": "Basic " + auth})
    e = json.load(urllib.request.urlopen(req, context=tls, timeout=10))
    return e["TemperatureCelsius"]["Reading"], e["PowerWatts"]["Reading"]
print(json.dumps([(nv(g), env(g), nv(g)) for g in range(gpus)]))
"""


def bmc_readings(b, gpus):
    """[(temperature, power)] per GPU from the processors' EnvironmentMetrics."""
    out = []
    for g in range(gpus):
        env = b.get(f"{gpu_path(g)}/EnvironmentMetrics")
        out.append((env["TemperatureCelsius"]["Reading"], env["PowerWatts"]["Reading"]))
    return out


def assert_follows_nvml(lab, tray):
    """[(temperature, power)] per GPU, as the BMC reported them."""
    bmc = lab.ips[f"{tray}-bmc"]
    out = lab.sh(tray, f"python3 - {bmc} '{lab.user}:{lab.password}' {lab.gpus} <<'EOF'\n{COMPARE}EOF")
    readings = []
    for g, ((t0, p0), (t, p), (t1, p1)) in enumerate(json.loads(out)):
        assert min(t0, t1) - TEMP_TOL <= t <= max(t0, t1) + TEMP_TOL, f"GPU_{g}: NVML {t0}..{t1} °C, BMC {t} °C"
        assert min(p0, p1) - POWER_TOL <= p <= max(p0, p1) + POWER_TOL, f"GPU_{g}: NVML {p0}..{p1} W, BMC {p} W"
        readings.append((t, p))
    return readings


@contextlib.contextmanager
def gpu_load(lab, tray, gpu, seconds=60):
    """Busy GPU on the tray through the fake CUDA driver (no scheduler involved)."""
    script = (
        "import ctypes, time\n"
        "cu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0)\n"
        "dev, ctx = ctypes.c_int(), ctypes.c_void_p()\n"
        f"cu.cuDeviceGet(ctypes.byref(dev), {gpu})\n"
        "cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)\n"
        "cu.fakegpu_run.argtypes = [ctypes.c_void_p, ctypes.c_uint64]\n"
        f"end = time.time() + {seconds}\n"
        "while time.time() < end: cu.fakegpu_run(None, 100000000)\n"
    )
    lab.sh(tray, f"cat > /tmp/bmc-load.py <<'EOF'\n{script}EOF\n"
                 "systemd-run --unit=bmc-sensor-load --collect python3 /tmp/bmc-load.py")
    try:
        yield
    finally:
        lab.sh(tray, "systemctl stop bmc-sensor-load; rm -f /tmp/bmc-load.py", check=False)


def test_sensor_resources(lab, tray):
    b = lab.tray_bmc(tray)
    for g in range(lab.gpus):
        sensors = gpu_sensors(g)
        assert b.get(f"{ROOT}/Chassis/HGX_GPU_{g}")["Sensors"]["@odata.id"] == sensors
        assert {m.rsplit("/", 1)[1] for m in members(b.get(sensors))} == {f"HGX_GPU_{g}_TEMP_0", f"HGX_GPU_{g}_Power_0"}
        t = b.get(f"{sensors}/HGX_GPU_{g}_TEMP_0")
        assert (t["ReadingType"], t["ReadingUnits"], t["PhysicalContext"]) == ("Temperature", "Cel", "GPU")
        assert t["RelatedItem"] == [{"@odata.id": gpu_path(g)}]
        p = b.get(f"{sensors}/HGX_GPU_{g}_Power_0")
        assert (p["ReadingType"], p["ReadingUnits"]) == ("Power", "W")
        assert 100 < p["Reading"] < 1100 and 25 <= t["Reading"] <= 80
        env = b.get(f"{gpu_path(g)}/EnvironmentMetrics")
        assert env["TemperatureCelsius"]["DataSourceUri"] == f"{sensors}/HGX_GPU_{g}_TEMP_0"

    chassis = b.get(TRAY)
    assert {m.rsplit("/", 1)[1] for m in members(b.get(chassis["Sensors"]["@odata.id"]))} == {"Total_GPU_Power_0"}
    thermal = b.get(chassis["ThermalSubsystem"]["@odata.id"])
    temps = b.get(thermal["ThermalMetrics"]["@odata.id"])["TemperatureReadingsCelsius"]
    assert [t["DeviceName"] for t in temps] == [f"GPU_{g}" for g in range(lab.gpus)]
    assert [t["DataSourceUri"] for t in temps] == [f"{gpu_sensors(g)}/HGX_GPU_{g}_TEMP_0" for g in range(lab.gpus)]
    total = b.get(chassis["EnvironmentMetrics"]["@odata.id"])["PowerWatts"]
    assert total["DataSourceUri"] == f"{TRAY}/Sensors/Total_GPU_Power_0"
    assert lab.gpus * 100 < total["Reading"] < lab.gpus * 1100

def test_readings_follow_nvml(lab, tray):
    assert_follows_nvml(lab, tray)


def test_readings_follow_load(lab):
    """Under load the BMC sees the GPU heat up and draw more, as NVML does."""
    tray, gpu = lab.trays[0], 1
    if lab.sh(tray, f"nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader").strip():
        pytest.skip(f"{tray} GPU {gpu} is in use")
    idle_temp = bmc_readings(lab.tray_bmc(tray), lab.gpus)[gpu][0]
    with gpu_load(lab, tray, gpu):
        def hot():
            temp, power = assert_follows_nvml(lab, tray)[gpu]
            return (power < 900 or temp < idle_temp + 5) and f"GPU {gpu}: {temp} °C, {power} W"
        eventually(hot, timeout=45, interval=2)


def test_exporter_reports_gpu_temperatures(lab, tray):
    """Out of band, as Prometheus sees it: idrac_exporter reads ThermalMetrics."""
    b = lab.tray_bmc(tray)
    before = [t for t, _ in bmc_readings(b, lab.gpus)]
    exported = [lab.oob_metric(f"{tray}-bmc", "idrac_sensors_temperature", name=f"GPU_{g}") for g in range(lab.gpus)]
    after = [t for t, _ in bmc_readings(b, lab.gpus)]
    for g, (t0, t, t1) in enumerate(zip(before, exported, after)):
        assert min(t0, t1) - 1 <= t <= max(t0, t1) + 1, f"GPU_{g}: BMC {t0}..{t1}, exporter {t}"  # slower bracket


def test_switch_bmc_is_discoverable(lab):
    """Generic Redfish clients start from a system: the switch tray has one."""
    assert lab.oob_metric(lab.switch_bmc, "idrac_system_power_on") == 1
