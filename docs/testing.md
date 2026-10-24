[← README](../README.md) · **Testing**

# Testing

## Overview

`make test` (`playbooks/test.yml`) checks the cluster end to end, from the user's point of view where possible: jobs run through the scheduler (Slurm or k3s) as the directory user `joe`, and every management and monitoring interface is queried the way a client would.

Three more test sets run from the host: the [fake GPU library tests](#fake-gpu-library-tests) (`make test-fakegpu`) and the [device plugin tests](#device-plugin-tests) (`make test-fakedp`) need no lab, the [BMC integration tests](#bmc-integration-tests) (`make test-bmc`) run against the running lab.

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
| Kubernetes (k3s mode) | both trays ready with 4 GPUs, the GFD clique label and topograph's domain label; the four examples as JobSets through Kueue (`examples/kubernetes/submit --wait`) with the same output checks, with KubeRay also `ray-job` as a RayJob; 8 single-GPU pods got 8 different GPUs; `joe` reads nodes and queues but not `kube-system` |
| Jobs as `joe` (Slurm mode) | `gpu-topology`, `nvl8-hello`, `ddp-train`, `ray-cluster` run with `sbatch --wait`; their output contains the fabric line, rank 7, all 8 DDP ranks, 8 Ray tasks |
| Accounting (Slurm mode) | every job above is `COMPLETED` in `sacct` |
| GPU handover | the Slurm epilog reset the jobs' GPUs (Slurm mode); no GPU needs a reset or other recovery |
| Shared filesystem | `/pfs` mounted on every cluster node; a file written on the login node reads back everywhere |
| Object storage | RustFS healthy; from a job (Slurm) or a pod (k3s), `joe` writes and lists his bucket and is denied the JuiceFS bucket |
| BMCs | tray BMCs report their tray powered on with 4 GPUs, also through the Redfish exporter; the switch BMC exposes 72 ports per switch |
| Partition controller | gRPC port open; every GPU reports the domain UUID and a clique |
| Topology | topograph generates a block containing both trays (Slurm) or labels both trays with the NVLink domain and switch tiers (k3s) |
| MIG in Kubernetes (k3s mode) | the last GPU of a tray split into a `3g.93gb` and two `1g.23gb`: the device plugin offers them as `nvidia.com/mig-1g.23gb` (2) and `nvidia.com/mig-3g.93gb` (1) and the GPU no longer as `nvidia.com/gpu`; a pod as `joe` on one `1g.23gb` sees that slice only (`nvidia-smi -L`, CUDA: 22528 MiB, 18 SMs, out of memory beyond); undone afterwards |
| ComputeDomains (k3s mode, `nvidia_dra: true`) | `imex-share` runs as `joe`: rank 1 imports and reads the GPU memory rank 0 exported as a fabric handle on the other tray, over its ComputeDomain's IMEX channels, and Kueue admitted it with its 2 channels as `nvidia.com/imex-channel`; a GPU pod without a channel gets `CUDA_ERROR_NOT_PERMITTED` for fabric memory |
| KubeRay (k3s mode, `kuberay: true`) | with `make frameworks` (the venv's Ray), the `ray-job` example runs as `joe` (8 Ray tasks) after Kueue admitted its cluster; from the host, the dashboard serves its pages at `http://kuberay.<controller address>.nip.io` and points the browser at the API server behind the same name; the finished `ray-job`'s driver output is in the dashboard's History (logs uploaded to RustFS by its collectors), and its Logs link leads to its cluster's Jobs page in Ray's dashboard for finished clusters; the API server lists Ray jobs and clusters, creates a Ray cluster whose Ray dashboard answers through it and whose metrics reach Prometheus, and deletes it (the `ray-job` and API server checks are skipped without the venv); Ray's default dashboard is in Grafana's Ray folder for anonymous viewers |
| NVIDIA Run:ai (`runai: true`) | the cluster `Connected` to the control plane with every required dependency; Run:ai's Prometheus has DCGM metrics for all 8 GPUs; a 2-GPU training workload submitted through the REST API (project `test`) runs on a tray, scheduled by `runai-scheduler`, and lists its 2 GPUs; two half-GPU workloads share one GPU, each with half its memory as limit and out of memory beyond it; with KubeRay and `make frameworks`, a Ray cluster in project `test` is scheduled by `runai-scheduler` as one pod group, listed as a running `RayCluster` workload with 2 GPUs, and scraped by Prometheus |
| Monitoring | every Prometheus target up (retried for up to a minute, while targets of Ray pods just started or deleted settle); power for all 8 GPUs; scheduler GPU (`sched_*`), NVLink fabric and InfiniBand port counter series present; Grafana serves the dashboards |

## Rebuild scenarios

`make up` must give a working lab in one attempt whatever an earlier attempt left behind, or stop at once with the fix. Checked by hand before a release (each scenario ends with `make test`; they delete the lab):

| Scenario | Expected |
|---|---|
| Interrupt `make up` during `configure`, run `make up` again | completes; `make test` passes |
| `make down`, `make up` | completes on the kept volumes; `make test` passes; `make frameworks` reinstalls `uv` and keeps the venv |
| Interrupt `make up`, delete the checkout (and its `.secrets/`), clone, `make init`, `make up` | stops in `provision` at "Stop if the data volumes were created with other secrets", before changing anything; after `make purge`, `make up` completes and `make test` passes |
| Same, with the original `.secrets/` copied into the new clone | completes on the kept volumes; `make test` passes |
| Canonical's mirror unreachable | `apt_mirror: auto` picks another mirror; a mirror that stops answering fails apt within seconds (10 s timeout, 3 retries) instead of hanging |

## Fake GPU library tests

`make test-fakegpu` builds the lab's GPU libraries on this machine and tests them without the lab (`tests/fakegpu`, pytest). Each NCCL rank runs as a process on an emulated tray of its own. The tests read each operation's cost from the GPUs' busy time and InfiniBand counters, so they are exact. They check that:

- one partition runs at NVLink speed; two partitions cross InfiniBand (more than 2× slower) and count its bytes
- GPUs in no partition use only InfiniBand
- `ncclCommSplit` by partition is back on NVLink, and `ncclSend` to another partition uses InfiniBand
- a rank whose peers never report warns and falls back to NVLink after the timeout
- each GEMM precision runs at its rate, and the NUMA layout and CPU affinity match GB200's
- the InfiniBand byte counters appear under node_exporter's names
- a GPU fraction's memory limit (`NVIDIA_GPU_MEMORY_LIMIT`) holds allocations, and `cuMemGetInfo` reports it
- fabric memory (and multicast) reaches another tray of its NVLink partition and stops at the partition, a GPU in no partition has none, and creating or importing it needs an IMEX channel (`CUDA_ERROR_NOT_PERMITTED` without one)
- no library needs a glibc newer than 2.33, so they load in images built on older distributions (Ubuntu 22.04)
- MIG works as on a GB200 (`test_mig_*.py`): the fake NVML's MIG API against GB200's profiles, GPU and compute instance placements (with and without a requested placement) and the error codes its consumers rely on: changes need root (`Insufficient Permissions` otherwise), instances a process uses are `In use`; nvidia-smi's MIG commands in a GB200's formats (`mig -lgip` against real output verbatim, the default table's `MIG devices:` section), a pending mode while a process uses the GPU applied by `nvidia-smi -r`; CUDA's view of MIG devices (any compute instance by `MIG-` UUID, a GPU instance's memory shared across processes); a container sees only its slices, from `NVIDIA_VISIBLE_DEVICES` or `/var/run/nvidia-container-devices`; a driver load keeps MIG mode and drops the instances; NVIDIA's `mig-parted` (built from source) applies, asserts and exports each of its GB200 configurations, and GPU Feature Discovery labels MIG devices (mixed and single strategies)

As changing MIG state needs root, the MIG tests run NVML and `nvidia-smi` as root of a user namespace (`unshare -Ur`, `tests/fakegpu/mig_tray.py`), so the host needs unprivileged user namespaces: without them those tests fail, they are not skipped. The `mig-parted` and GPU Feature Discovery tests (`test_mig_tools.py`) build both from NVIDIA's sources with Go into `.cache/mig-tools`, and are skipped without Go (or `unshare`) or when the sources cannot be downloaded.

## Device plugin tests

`make test-fakedp` runs `go vet` and the Go unit tests (`go test -race`) of the lab's GPU device plugin (`fakedp`) on this machine, without the lab; it needs Go. They check the MIG layout read from `nvidia-smi` (compute instances smaller than their GPU instance left out), a MIG profile's resource name, a MIG allocation (parent GPU as CDI device, a `/var/run/nvidia-container-devices` file per slice), whole-GPU allocation refusing a GPU with MIG on, and a profile's plugin started and stopped with the profile.

## BMC integration tests

`make test-bmc` runs `tests/bmc` (pytest, from the host against the running lab). The tests talk Redfish to the BMCs the way a management client does and check the effect where real hardware shows it: `nvidia-smi` on the tray, the partition controller's fabric telemetry, the scheduler. Every change a test makes is undone afterwards.

| Target | Tier | Checks |
|---|---|---|
| `make test-bmc` | default (~30 s) | GB200 layout and behaviour from NVIDIA's references, each test citing its source (GPUs under `HGX_Baseboard_0`, an `HGX_GPU_<n>` chassis with the GPU's UUID, `BMC_0` and `HGX_BMC_0` managers, serials and `Chassis_0/Assembly`, firmware inventory, rack slot on the `CBC` chassis, `MGX_NVSwitch_<n>` chassis and `MGX_NVLinkFabric_0` on the switch tray, a link down at the switch end shown at the GPU end, a partition change waiting for a GPU reset (old clique and `GPU_RESET` recovery action until `nvidia-smi --gpu-reset`, the BMC following the GPU), a reset refused for a GPU in use or a non-root user, two NCCL ranks in different partitions all-reducing over InfiniBand at the modelled cost, with the tray's NIC counters showing it); Redfish protocol on every BMC (unauthenticated service root, 401s, session lifecycle, `Manager.Reset` ends sessions, Redfish error bodies, 404/405, a crawl of every reachable resource); tray inventory equals `nvidia-smi` (UUID, serial, PCI bus, memory, fabric clique); 18 NVLink ports per GPU; NVLink `Settings` stay pending until reset; sticky flag; invalid requests; fabric shape; cabling (each GPU link on exactly one switch port, 9 per switch, UUIDs agree across BMCs); a switch port taken down reaches `nvidia-smi` and fabric telemetry at once and recovers; switch settings and configuration upload; GPU sensors (resources, temperature and power equal to `nvidia-smi` at idle and under a load the test generates, the Redfish exporter's GPU temperatures, the switch BMC discoverable by the exporter) |
| `make test-bmc-disruptive` | `--disruptive` (~1 min) | on the last tray, taken out of scheduling first (Slurm drain, Kubernetes cordon) and refused if running jobs: pending link disable applies at `ForceRestart` and lasts one reset; sticky disable survives `ForceRestart` and `PowerCycle`; `ForceOff`/`On` (power state, links down, BMC stays up, the exporter reports the tray off, sensors without readings, slurmd or kubelet registers again and, with k3s, the tray's GPUs are allocatable again); `On` on a running tray is a no-op |
| `make test-bmc-conformance` | `--conformance` (~2 min) | real GB200 behaviour the lab does not model yet, each test marked as an expected failure (`xfail`, listed with its reason): a DDP job keeps running after one of its GPUs loses an NVLink (fake NCCL never fails). The target passes while the gaps stay as documented; it fails when a known gap unexpectedly passes (remove its marker) or a conformance test without the marker fails (a new gap or a regression) |

Extra pytest arguments go in `PYTEST_ARGS`, e.g. `make test-bmc PYTEST_ARGS='-k nvswitch -x'`.

Idempotency is checked separately: a second `make configure` must report `changed=0` on every host.

## References

- `playbooks/test.yml`, `tests/bmc/`
- Per-component verification steps: see the *Verification* section of each page in the [documentation](README.md)
