[← README](../README.md) · **Testing**

# Testing

## Overview

`make test` (`playbooks/test.yml`) checks the cluster end to end, from the user's point of view where possible: jobs run through the scheduler (Slurm or k3s) as the directory user `joe`, and every management and monitoring interface is queried the way a client would.

## Usage

```
make up              # cluster
make frameworks      # needed for the DDP and Ray jobs (skipped without it)
make test
```

A full rebuild from nothing: `make purge && make up && make frameworks && make test`.

## What is checked

| Area | Check |
|---|---|
| Slurm (Slurm mode) | both trays available (waits out jobs still completing) |
| Kubernetes (k3s mode) | both trays ready with 4 GPUs, the GFD clique label and topograph's domain label; the four examples as JobSets through Kueue (`examples/kubernetes/submit --wait`) with the same output checks; 8 single-GPU pods got 8 different GPUs; `joe` reads nodes and queues but not `kube-system` |
| Jobs as `joe` (Slurm mode) | `gpu-topology`, `nvl8-hello`, `ddp-train`, `ray-cluster` run with `sbatch --wait`; their output contains the fabric line, rank 7, all 8 DDP ranks, 8 Ray tasks |
| Accounting (Slurm mode) | every job above is `COMPLETED` in `sacct` |
| Shared filesystem | `/pfs` mounted on every cluster node; a file written on the login node reads back everywhere |
| Object storage | RustFS healthy; from a job (Slurm) or a pod (k3s), `joe` writes and lists his bucket and is denied the JuiceFS bucket |
| BMCs | tray BMCs report their tray powered on with 4 GPUs, also through the Redfish exporter; the switch BMC exposes 72 ports per switch |
| Partition controller | gRPC port open; every GPU reports the domain UUID and a clique |
| Topology | topograph generates a block containing both trays (Slurm) or labels both trays with the NVLink domain and switch tiers (k3s) |
| Monitoring | every Prometheus target up; power for all 8 GPUs; scheduler GPU (`sched_*`) and NVLink fabric series present; Grafana serves the dashboards |

## BMC integration tests

`make test-bmc` runs `tests/bmc` (pytest, from the host against the running lab). The tests talk Redfish to the BMCs the way a management client does and check the effect where real hardware shows it: `nvidia-smi` on the tray, the partition controller's fabric telemetry, the scheduler. Every change a test makes is undone afterwards.

| Target | Tier | Checks |
|---|---|---|
| `make test-bmc` | default (~15 s) | Redfish protocol on every BMC (unauthenticated service root, 401s, session lifecycle, `Manager.Reset` ends sessions, Redfish error bodies, 404/405, a crawl of every reachable resource); tray inventory equals `nvidia-smi` (UUID, serial, PCI bus, memory, fabric clique); 18 NVLink ports per GPU; NVLink `Settings` stay pending until reset; sticky flag; invalid requests; fabric shape; cabling (each GPU link on exactly one switch port, 9 per switch, UUIDs agree across BMCs); a switch port taken down reaches `nvidia-smi` and fabric telemetry at once and recovers; switch settings and configuration upload; GPU sensors (resources, temperature and power equal to `nvidia-smi` at idle and under a load the test generates, the Redfish exporter's GPU temperatures, the switch BMC discoverable by the exporter) |
| `make test-bmc-disruptive` | `--disruptive` (~1 min) | on the last tray, taken out of scheduling first (Slurm drain, Kubernetes cordon) and refused if running jobs: pending link disable applies at `ForceRestart` and lasts one reset; sticky disable survives `ForceRestart` and `PowerCycle`; `ForceOff`/`On` (power state, links down, BMC stays up, the exporter reports the tray off, sensors without readings, slurmd or kubelet registers again and, with k3s, the tray's GPUs are allocatable again); `On` on a running tray is a no-op |
| `make test-bmc-conformance` | `--conformance` (~2 min) | behaviour of NVIDIA's real GB200 BMCs and NVLink fabric, each test citing its source; failures list the gaps between the emulation and the hardware |

Extra pytest arguments go in `PYTEST_ARGS`, e.g. `make test-bmc PYTEST_ARGS='-k nvswitch -x'`.

Idempotency is checked separately: a second `make configure` must report `changed=0` on every host.

## References

- `playbooks/test.yml`, `tests/bmc/`
- Per-component verification steps: see the *Verification* section of each page linked from the [README](../README.md#layers-and-components)
