[← README](../README.md) · **BMCs (Redfish)**

# BMCs (Redfish, `fakebmc/`)

## Overview

The lab runs three BMCs, one per tray, each in its own container, serving Redfish over HTTPS with a self-signed certificate. `fakebmc` is a small Go service (standard library only) modelled on NVIDIA's OpenBMC fork, [NVIDIA/bmcweb](https://github.com/NVIDIA/bmcweb), in one of two roles:

| BMC | Address | Role | Manages |
|---|---|---|---|
| `sched-worker1-bmc`, `sched-worker2-bmc` | .31, .32 | `tray` | the GPU tray: power, its 4 GPUs and their 18 NVLink ports each |
| `sched-nvswitch-bmc` | .33 | `nvswitch` | the NVLink switch tray: fabric, 2 NVSwitch chips × 72 ports |

Credentials: `root` / `0penBmc` (OpenBMC's default; `bmc_username`, `bmc_password`). Basic auth and Redfish sessions (`X-Auth-Token`) are supported; the service root is unauthenticated.

## Usage

`bin/redfish <tray> <path> [curl args]` sends a request to `<tray>-bmc` (e.g. `bin/redfish sched-worker1 …`, `bin/redfish sched-nvswitch …`) with the configured credentials. Errors use Redfish's `@Message.ExtendedInfo` format.

**GPU tray BMC:**

| Resource | Methods |
|---|---|
| `/redfish/v1/Systems/System_0` | `GET` (PowerState, ProcessorSummary); `POST Actions/ComputerSystem.Reset` with `ResetType` `On`, `ForceOff`, `GracefulShutdown`, `GracefulRestart`, `ForceRestart`, `PowerCycle` |
| `…/Processors/GPU_<n>` | `GET`: model, UUID (matches `nvidia-smi`), serial, PCIe address, NVLink fabric clique |
| `…/GPU_<n>/Ports/NVLink_<k>` | `GET`; `PATCH {"Oem": {"Nvidia": {"LinkDisableSticky": true}}}` |
| `…/GPU_<n>/Ports/NVLink_<k>/Settings` | `GET`; `PATCH {"LinkState": "Disabled" \| "Enabled"}`, applied at the next tray reset |
| `/redfish/v1/Chassis/Chassis_0`, `/redfish/v1/Managers/BMC_0` | `GET`; `POST Actions/Manager.Reset` (drops sessions) |
| `/redfish/v1/SessionService/Sessions` | `POST` login, `GET`, `DELETE` |

**Switch tray BMC:**

| Resource | Methods |
|---|---|
| `/redfish/v1/Fabrics/NVLinkFabric_0` | `GET` (shows an uploaded switch configuration) |
| `…/Switches/NVSwitch_<n>` | `GET`; `PATCH {"Oem": {"Nvidia": {"SwitchIsolationMode": "SwitchCommunicationEnabled" \| "SwitchCommunicationDisabled", "PPCIeModeEnabled": bool}}}` |
| `…/Switches/NVSwitch_<n>/Ports/NVLink_<k>` | `GET` (`Oem.Nvidia.RemoteEndpoint`: tray, GPU, UUID, link); `PATCH {"LinkState": …}`, effective immediately |
| `…/NVLinkFabric_0/upload-switch-config` | `POST` multipart with one part `ImportFile` (stored, not interpreted); `DELETE` |

Cabling: GPU *g* (numbered across trays) link *l* lands on switch *l mod 2*, port *9g + ⌊l/2⌋*.

**Example**, disabling two NVLinks of GPU 2 and resetting the tray:

```
for l in 3 4; do
  bin/redfish sched-worker1 /redfish/v1/Systems/System_0/Processors/GPU_2/Ports/NVLink_$l/Settings -X PATCH -d '{"LinkState": "Disabled"}'
done
bin/redfish sched-worker1 /redfish/v1/Systems/System_0/Actions/ComputerSystem.Reset -X POST -d '{"ResetType": "ForceRestart"}'
bin/ssh sched-worker1 nvidia-smi topo -m           # GPU2 pairs now NV16
```

## How it works

- **Power** goes through the Incus API: the tray BMCs hold a client certificate restricted to the `trays` project, so a BMC can start, stop and restart the GPU trays but cannot reach any other instance. Resets apply pending NVLink settings first.
- **Link state** goes through the tray's sideband volume: the tray BMC writes `nvlink-disabled`, the switch BMC `nvlink-disabled-switch` (one file per tray), and the tray's NVML reports those links inactive. The partition controller and the fabric metrics see the same state ([NVLink partitions](nvlink-partitions.md)).
- State (pending settings, sticky flags, switch settings, uploaded config) is persisted under `/var/lib/fakebmc`.

## Verification

```
bin/redfish sched-worker1 /redfish/v1/Systems/System_0 | jq '{PowerState, ProcessorSummary}'
bin/redfish sched-worker1 /redfish/v1/Systems/System_0/Processors/GPU_0 | jq .UUID       # == nvidia-smi UUID on the tray
bin/redfish sched-nvswitch /redfish/v1/Fabrics/NVLinkFabric_0/Switches/NVSwitch_0/Ports | jq '."Members@odata.count"'   # 72
```

`make test` checks that both tray BMCs report their tray powered on with 4 GPUs and that the switch BMC exposes 72 ports per switch. `make test-bmc` runs the BMC integration tests, with disruptive and GB200-conformance tiers ([Testing](testing.md#bmc-integration-tests)).

## Limitations

- Only the resources listed above; no firmware update, sensors, logs or event subscriptions.
- `SwitchIsolationMode` governs switch-to-switch trunks, which a single switch tray does not have: it is stored and reported, without effect on GPU links.
- After `ForceOff`, Slurm marks the tray down only after `SlurmdTimeout`; Kubernetes marks the node `NotReady` after its node-monitor grace period.
- The resource layout is simpler than NVIDIA's GB200 BMCs: GPUs are processors of `System_0` (not of the HMC's `HGX_Baseboard_0`), there are no `HGX_GPU_<n>` / `MGX_NVSwitch_<n>` chassis or `HGX_BMC_0` manager, and the switch tray's fabric is `NVLinkFabric_0` (NVIDIA: `MGX_NVLinkFabric_0`). `make test-bmc-conformance` lists these gaps.
- A switch port taken down is reported down by the switch BMC and by NVML, but the GPU's port on the tray BMC still reports `LinkUp`; the GPU's `FabricClique` on the tray BMC is the configured one and does not follow partition changes (NVML's does).

## References

- [DMTF Redfish](https://www.dmtf.org/standards/redfish) and the [Redfish schema index](https://redfish.dmtf.org/redfish/schema_index)
- [NVIDIA/bmcweb](https://github.com/NVIDIA/bmcweb): NVIDIA OEM schemas (`NvidiaPort`, `NvidiaSwitch`, `NvidiaFabric`) and NVLink routes
- [NVIDIA Switch BMC User Manual](https://docs.nvidia.com/networking/display/nvidia-switch-bmc-user-manual-v88-0002-0931.0931.pdf), [GB200 NVL72 firmware release notes](https://docs.nvidia.com/pdf/gb200-fw-relnote-1-3-6.pdf)
- [dsx-ai-factory/nv-redfish](https://github.com/dsx-ai-factory/nv-redfish): NVIDIA's Rust Redfish client with NVIDIA OEM types
- Sources: `fakebmc/` (`redfish.go` tray role, `nvswitch.go` switch role, `power.go`, `state.go`); role `roles/fakebmc`
