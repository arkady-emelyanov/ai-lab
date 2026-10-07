[← README](../README.md) · **BMCs (Redfish)**

# BMCs (Redfish, `fakebmc/`)

## Overview

The lab runs three BMCs, one per tray, each in its own container, serving Redfish over HTTPS with a self-signed certificate. `fakebmc` is a small Go service (standard library only) modelled on NVIDIA's OpenBMC fork, [NVIDIA/bmcweb](https://github.com/NVIDIA/bmcweb), with the resource layout of NVIDIA's GB200 BMCs (NVIDIA's BMC models in [infra-controller](https://github.com/dsx-ai-factory/infra-controller)'s `bmc-mock`, the Switch BMC manual), in one of two roles:

| BMC | Address | Role | Manages |
|---|---|---|---|
| `sched-worker1-bmc`, `sched-worker2-bmc` | .31, .32 | `tray` | the GPU tray: power, its 4 GPUs, their 18 NVLink ports each and their temperature and power sensors |
| `sched-nvswitch-bmc` | .33 | `nvswitch` | the NVLink switch tray: fabric, 2 NVSwitch chips × 72 ports |

Credentials: `root` / `0penBmc` (OpenBMC's default; `bmc_username`, `bmc_password`). Basic auth and Redfish sessions (`X-Auth-Token`) are supported; the service root is unauthenticated.

## Usage

`bin/redfish <tray> <path> [curl args]` sends a request to `<tray>-bmc` (e.g. `bin/redfish sched-worker1 …`, `bin/redfish sched-nvswitch …`) with the configured credentials. Errors use Redfish's `@Message.ExtendedInfo` format.

**GPU tray BMC.** As on a GB200 compute tray, `System_0` is the host (Grace) with power and reset, and the GPUs are processors of `HGX_Baseboard_0`, behind the HGX Management Controller (manager `HGX_BMC_0`), each with its own `HGX_GPU_<n>` chassis:

| Resource | Methods |
|---|---|
| `/redfish/v1/Systems/System_0` | `GET` (PowerState, SerialNumber); `POST Actions/ComputerSystem.Reset` with `ResetType` `On`, `ForceOff`, `GracefulShutdown`, `GracefulRestart`, `ForceRestart`, `PowerCycle` |
| `/redfish/v1/Systems/HGX_Baseboard_0` | `GET` (PowerState, ProcessorSummary: the GPUs) |
| `…/HGX_Baseboard_0/Processors/GPU_<n>` | `GET`: model, UUID (matches `nvidia-smi`), serial, PCIe address, NVLink fabric clique (the GPU's, which changes with its partition at the GPU's next reset) |
| `…/GPU_<n>/EnvironmentMetrics` | `GET`: GPU temperature and power (`TemperatureCelsius`, `PowerWatts`, linked to the GPU chassis' sensors), power limit |
| `…/GPU_<n>/Ports/NVLink_<k>` | `GET` (`LinkDown` when either end is disabled); `PATCH {"Oem": {"Nvidia": {"LinkDisableSticky": true}}}` |
| `…/GPU_<n>/Ports/NVLink_<k>/Settings` | `GET`; `PATCH {"LinkState": "Disabled" \| "Enabled"}`, applied at the next tray reset |
| `/redfish/v1/Chassis/Chassis_0` | `GET`: the tray (PowerState, serial); `Assembly` (serial for inventory tools); `Sensors` (`Total_GPU_Power_0`), `EnvironmentMetrics` (the tray's GPU power), `ThermalSubsystem/ThermalMetrics` (all GPU temperatures; no fans, the tray is liquid-cooled) |
| `/redfish/v1/Chassis/HGX_GPU_<n>` | `GET`: the GPU's UUID and serial; `Sensors` (`HGX_GPU_<n>_TEMP_0` °C, `HGX_GPU_<n>_Power_0` W; no reading, `UnavailableOffline`, with the tray off), `EnvironmentMetrics` |
| `/redfish/v1/Chassis/CBC_<n>` | `GET`: NVLink cable cartridges; `Oem.Nvidia` `ChassisPhysicalSlotNumber`, `ComputeTrayIndex`, `TopologyId` place the tray in the rack |
| `/redfish/v1/Chassis/BMC_0`, `HGX_Chassis_0` | `GET` |
| `/redfish/v1/Managers/BMC_0`, `HGX_BMC_0` | `GET`; `POST BMC_0/Actions/Manager.Reset` (drops sessions) |
| `/redfish/v1/UpdateService/FirmwareInventory` | `GET`: `FW_BMC_0`, `HGX_FW_BMC_0`, `HGX_FW_GPU_<n>` (the VBIOS `nvidia-smi` reports) |
| `/redfish/v1/SessionService/Sessions` | `POST` login, `GET`, `DELETE` |

**Switch tray BMC:**

| Resource | Methods |
|---|---|
| `/redfish/v1/Systems/System_0` | `GET`: the switch tray's CPU (NVOS host), always on; `POST Actions/ComputerSystem.Reset` is accepted without effect |
| `/redfish/v1/Chassis` | `BMC_eeprom`, `CPLD_0`, `MGX_BMC_0`, `MGX_NVSwitch_<n>` (one per switch chip) |
| `/redfish/v1/Fabrics/MGX_NVLinkFabric_0` | `GET` (shows an uploaded switch configuration) |
| `…/Switches/NVSwitch_<n>` | `GET`; `PATCH {"Oem": {"Nvidia": {"SwitchIsolationMode": "SwitchCommunicationEnabled" \| "SwitchCommunicationDisabled", "PPCIeModeEnabled": bool}}}` |
| `…/Switches/NVSwitch_<n>/Ports/NVLink_<k>` | `GET` (`Oem.Nvidia.RemoteEndpoint`: tray, GPU, UUID, link); `PATCH {"LinkState": …}`, effective immediately |
| `…/MGX_NVLinkFabric_0/upload-switch-config` | `POST` multipart with one part `ImportFile` (stored, not interpreted); `DELETE` |
| `/redfish/v1/UpdateService/FirmwareInventory` | `GET`: `MGX_FW_BMC_0`, `MGX_FW_CPLD_0`, `MGX_FW_NVSwitch_<n>` |

Paths differ between vendors and firmware releases; tools should start from a collection (`/redfish/v1/Systems`, `/Chassis`, `/Fabrics`) and follow its `Members`, which works on the lab and on hardware alike.

Cabling: each GPU's 18 links are split over the two switch chips, even links on `NVSwitch_0` and odd ones on `NVSwitch_1`, 9 on each; on each chip every GPU has a block of 9 consecutive ports (GPU 0, counted across both trays, on ports 0–8, GPU 1 on 9–17, ... GPU 7 on 63–71).

**Example**, disabling two NVLinks of GPU 2 and resetting the tray:

```
for l in 3 4; do
  bin/redfish sched-worker1 /redfish/v1/Systems/HGX_Baseboard_0/Processors/GPU_2/Ports/NVLink_$l/Settings -X PATCH -d '{"LinkState": "Disabled"}'
done
bin/redfish sched-worker1 /redfish/v1/Systems/System_0/Actions/ComputerSystem.Reset -X POST -d '{"ResetType": "ForceRestart"}'
bin/ssh sched-worker1 nvidia-smi topo -m           # GPU2 pairs now NV16
```

## How it works

- **Power** goes through the Incus API: the tray BMCs hold a client certificate restricted to the `trays` project, so a BMC can start, stop and restart the GPU trays but cannot reach any other instance. Resets apply pending NVLink settings first.
- **Link state** goes through the tray's sideband volume: the tray BMC writes `nvlink-disabled`, the switch BMC `nvlink-disabled-switch` (one file per tray), and the tray's NVML reports those links inactive. The partition controller and the fabric metrics see the same state ([NVLink partitions](nvlink-partitions.md)).
- State (pending settings, sticky flags, switch settings, uploaded config) is persisted under `/var/lib/fakebmc`.
- **Sensors** read the tray's GPU state (the fake GPU stack's occupancy file, on a volume mounted read-only in the BMC: the emulated I2C/SMBus path to the GPUs) and apply the fake NVML's own model to it, the same formulas and clock, so temperature and power match `nvidia-smi` and the GPU exporter at the same moment ([Emulated GPUs](fake-gpu.md#how-it-works)).
- **Polled out of band**: Prometheus scrapes the tray BMCs through a generic Redfish exporter on the controller, so a tray powered off through Redfish shows as `idrac_system_power_on 0` while its BMC stays up, and GPU temperatures arrive both in band and out of band ([Monitoring](monitoring.md#overview)).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `bmc_username`, `bmc_password` | `root`, `0penBmc` | credentials of every BMC |
| `bmc_firmware_version` | `fakebmc-1.0.0` | host BMC firmware: `FW_BMC_0` (switch tray: `MGX_FW_BMC_0`) and manager `BMC_0`'s `FirmwareVersion` |
| `hmc_firmware_version` | `fakebmc-hmc-1.0.0` | HGX Management Controller firmware: `HGX_FW_BMC_0` and manager `HGX_BMC_0` |
| `fakegpu_vbios_version` | `97.00.82.00.0F` | GPU VBIOS: `HGX_FW_GPU_<n>`, the same as `nvidia-smi` on the tray ([Emulated GPUs](fake-gpu.md#configuration)) |
| `nvswitch_firmware_version`, `nvswitch_cpld_firmware_version` | `fakebmc-nvswitch-1`, `fakebmc-cpld-1` | switch tray: `MGX_FW_NVSwitch_<n>`, `MGX_FW_CPLD_0` |

Port speeds (`CurrentSpeedGbps` of 2 lanes) follow `fakegpu_nvlink_link_gbs`, the GPU sensors and `PowerLimitWatts` the GPU profile's power figures, so they match `nvidia-smi` whatever the profile ([Emulated GPUs](fake-gpu.md#configuration)).

**Per tray.** A tray BMC takes the VBIOS and its firmware versions from the tray's own variables, so one tray can drift from the others (for inventory or compliance tests). Set them under the tray in `inventory/hosts.yml`, then `make configure`; the tray's `nvidia-smi` reports the same VBIOS:

```
            sched-worker2:
              tray: 2
              ip_host: 22
              fakegpu_vbios_version: 97.00.82.00.10
              bmc_firmware_version: fakebmc-1.0.1
```

The switch tray BMC takes its versions from `sched-nvswitch-bmc`'s variables.

## Verification

```
bin/redfish sched-worker1 /redfish/v1/Systems | jq -r '.Members[]."@odata.id"'          # System_0, HGX_Baseboard_0
bin/redfish sched-worker1 /redfish/v1/Systems/HGX_Baseboard_0/Processors/GPU_0 | jq .UUID   # == nvidia-smi UUID on the tray
bin/redfish sched-nvswitch /redfish/v1/Fabrics/MGX_NVLinkFabric_0/Switches/NVSwitch_0/Ports | jq '."Members@odata.count"'   # 72
bin/redfish sched-worker1 /redfish/v1/Systems/HGX_Baseboard_0/Processors/GPU_0/EnvironmentMetrics | jq '.TemperatureCelsius.Reading, .PowerWatts.Reading'
bin/ssh sched-worker1 nvidia-smi -i 0 --query-gpu=temperature.gpu,power.draw --format=csv   # same values
```

`make test` checks that both tray BMCs report their tray powered on with 4 GPUs, also as seen through the Redfish exporter, and that the switch BMC exposes 72 ports per switch. `make test-bmc` runs the BMC integration tests, including the GB200 layout and behaviour checks from NVIDIA's references, with a disruptive tier and a conformance tier for behaviour the lab does not model yet ([Testing](testing.md#bmc-integration-tests)).

## Limitations

- Only the resources listed above; no firmware update (the inventory is read-only), logs or event subscriptions; no Grace CPU, memory, PCIe or network adapter resources. Sensors cover the GPUs only (no CPU, memory, inlet or switch ASIC sensors, no thresholds) and there is no `PowerSubsystem`: GB200 compute trays draw from the rack's power shelves.
- The Redfish exporter skips processors that are not CPUs, so it reports no per-GPU health, and it reads power only from power supplies: from the BMCs it exports GPU temperatures but not GPU power (use the sensors or the GPU exporter).
- `SwitchIsolationMode` governs switch-to-switch trunks, which a single switch tray does not have: it is stored and reported, without effect on GPU links.
- After `ForceOff`, Slurm marks the tray down only after `SlurmdTimeout`; Kubernetes marks the node `NotReady` after its node-monitor grace period.
- `Chassis_0` is listed first in the chassis collection (real trays list `BMC_0` first), so generic clients that take the first chassis, such as the Redfish exporter, find the GPU temperatures.
- The switch tray's `ComputerSystem.Reset` has no effect: the lab has no power control over the switch tray host.
- A job keeps running when one of its GPUs loses an NVLink (fake NCCL never fails); on hardware it runs into errors. `make test-bmc-conformance` checks this remaining gap.

## References

- [DMTF Redfish](https://www.dmtf.org/standards/redfish) and the [Redfish schema index](https://redfish.dmtf.org/redfish/schema_index)
- [NVIDIA/bmcweb](https://github.com/NVIDIA/bmcweb): NVIDIA OEM schemas (`NvidiaPort`, `NvidiaSwitch`, `NvidiaFabric`) and NVLink routes
- [NVIDIA Switch BMC User Manual](https://docs.nvidia.com/networking/display/nvidia-switch-bmc-user-manual-v88-0002-0931.0931.pdf), [GB200 NVL72 firmware release notes](https://docs.nvidia.com/pdf/gb200-fw-relnote-1-3-6.pdf)
- [dsx-ai-factory/nv-redfish](https://github.com/dsx-ai-factory/nv-redfish): NVIDIA's Rust Redfish client with NVIDIA OEM types
- [infra-controller](https://github.com/dsx-ai-factory/infra-controller) `crates/bmc-mock/src/hw/` (`wiwynn_gb200_nvl.rs`, `nvidia_gb200.rs`, `nvidia_gbx00.rs`, `nvidia_switch_nd5200_ld.rs`): NVIDIA's models of real GB200 BMCs
- Sources: `fakebmc/` (`layout.go` systems, chassis, managers, firmware; `redfish.go` tray role; `nvswitch.go` switch role; `sensors.go`; `power.go`; `state.go`); role `roles/fakebmc`
