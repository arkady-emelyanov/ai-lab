---
name: ai-lab
description: Operate a running ai-lab, the emulated NVIDIA GB200 GPU cluster (an 8-GPU slice of an NVL72-style NVLink domain the lab calls NVL8; Slurm or Kubernetes, emulated GPUs, Redfish BMCs, NVLink partitions, Prometheus and Grafana). Use when the user wants to run GPU jobs on the lab, check its GPUs, nodes or queues, change NVLink partitions, reset GPUs, power-cycle a tray through its BMC, query metrics, or test the lab.
---

# Operating ai-lab

The lab runs on the user's machine; commands run from the repository's root (the checkout with `AGENTS.md`). Deploying it is described in `AGENTS.md`. Before acting, find out what the lab runs: `cat inventory/group_vars/all.yml local.yml 2>/dev/null | grep '^scheduler:' | tail -1` (`slurm` or `k3s`; `local.yml` holds the user's settings and wins) and whether it is up (`bin/ssh sched-control true`).

## Rules

- The lab may be in use by the user or another session: before anything that changes state (partitions, GPU resets, power, jobs that fill the GPUs, `make configure`), check nothing is running (Slurm: `bin/ssh sched-control squeue`; k3s: `bin/kubectl get pods -A --field-selector status.phase=Running`) and say what you are about to change.
- Leave the lab as you found it: one default NVLink partition with all 8 GPUs, GPUs reset, no test jobs left behind.
- Power actions through the BMCs stop the tray's jobs: only when the user asks for them.
- `make down` / `make purge` destroy the instances (and, for purge, the data and monitoring history): only when the user asks. Apply configuration changes with `make configure`, which keeps them.
- Long commands (`make test`, `make configure`, big jobs) run in the background with their output in a log file; give the user the `tail -f` command.
- Never print or commit `.secrets/`.

## Access

| Tool (from the repository root) | What |
|---|---|
| `bin/ssh login` | login node as the directory user `joe` (key-based); `bin/ssh root@<instance>` as root (`sched-control`, `sched-worker1`, `sched-worker2`, `sched-storage`, `sched-nvswitch`, BMCs) |
| `bin/scp -r examples login:` | copy the example jobs to joe's home |
| `bin/kubectl` | k3s as cluster admin (k3s mode) |
| `bin/nvlink` | NVLink domain and partitions (partition controller) |
| `bin/redfish <tray> <path> [curl args]` | Redfish request to `<tray>-bmc` (`sched-worker1`, `sched-worker2`, `sched-nvswitch`) |
| Grafana `http://<bridge>.10:3000` (admin, `.secrets/grafana.pass`), Prometheus `http://<bridge>.10:9090` | dashboards "Lab overview", "Scheduler & NVLink fabric"; `<bridge>` from `incus network get incusbr0 ipv4.address` (10.107.111 in the docs) |

## Run jobs

Slurm (on `bin/ssh login`):

```
sinfo -N -o "%N %G %T"                              # 2 trays, gpu:gb200:4 each
srun -N2 --gpus-per-node=4 nvidia-smi -L            # all 8 GPUs
cd examples/slurm && sbatch --wait nvl8-hello.sbatch
sbatch ddp-train.sbatch --steps 20000               # PyTorch DDP on 8 GPUs (needs make frameworks)
squeue; sacct -X -o JobID,JobName,AllocTRES%40,State
```

Kubernetes (k3s mode, on `bin/ssh login`; jobs are JobSets queued in Kueue):

```
kubectl get nodes -L nvidia.com/gpu.clique,accelerator.topograph.run/domain
cd examples/kubernetes && ./submit --wait nvl8-hello.yaml     # prints the run name, writes <name>.out
kubectl get jobsets,workloads
```

Examples (`examples/slurm/*.sbatch`, `examples/kubernetes/*.yaml`): `gpu-topology`, `nvl8-hello`, `ddp-train`, `ray-cluster`. The GPUs are emulated: frameworks initialise and every call succeeds with modelled timing and telemetry, but nothing is computed.

## GPUs

```
bin/ssh sched-worker1 nvidia-smi                    # utilisation, memory, power, processes
bin/ssh sched-worker1 nvidia-smi -q | grep -E "Recovery|CliqueId"
bin/ssh sched-worker1 nvidia-smi --gpu-reset [-i N] # root; refused while a process uses the GPU
```

A GPU takes a new NVLink partition only at a reset or a tray reboot (`GPU Recovery Action : GPU_RESET` until then). In Slurm mode the epilog resets each job's GPUs when it ends (`journalctl -t slurm-epilog` on the tray).

## NVLink partitions

```
bin/nvlink partitions                               # partitions, their GPUs, GPUs in none
bin/nvlink gpus [TRAY]                              # partition, clique, pending reset, links, health
bin/nvlink remove default sched-worker2             # a GPU is in at most one partition
bin/nvlink create tray2 --id 7 sched-worker2        # prints the reset to run
bin/ssh sched-worker2 nvidia-smi --gpu-reset        # the GPUs take partition 7
bin/nvlink delete tray2; bin/nvlink add default sched-worker2; bin/ssh sched-worker2 nvidia-smi --gpu-reset   # restore
```

- Keep each tray's GPUs in one partition: topograph needs one partition per node and stops updating the scheduler's topology on a split tray (`bin/nvlink` warns).
- The scheduler sees a change within a minute of the reset (Slurm `scontrol show topology`, k3s node labels).
- A job spanning partitions runs slower: its NCCL traffic between partitions crosses InfiniBand (`node_infiniband_port_data_transmitted_bytes_total`). Kueue keeps multi-pod jobs inside one NVLink domain; Slurm 23.11 lets them span blocks.

## BMCs (Redfish)

```
bin/redfish sched-worker1 /redfish/v1/Systems/System_0 | jq .PowerState
bin/redfish sched-worker1 /redfish/v1/Systems/HGX_Baseboard_0/Processors/GPU_0/EnvironmentMetrics | jq '.TemperatureCelsius.Reading, .PowerWatts.Reading'
bin/redfish sched-worker1 /redfish/v1/UpdateService/FirmwareInventory | jq -r '.Members[]."@odata.id"'
bin/redfish sched-worker2 /redfish/v1/Systems/System_0/Actions/ComputerSystem.Reset -X POST -d '{"ResetType": "ForceRestart"}'   # stops the tray's jobs: ask first
```

Layout is NVIDIA's GB200 one (GPUs under `HGX_Baseboard_0`, `HGX_GPU_<n>` chassis, switch fabric `MGX_NVLinkFabric_0`); start from a collection and follow `Members` rather than hard-coding paths. Before powering a tray off, take it out of scheduling (Slurm `scontrol update nodename=<tray> state=drain reason=...`, k3s `kubectl cordon <tray>`) and put it back afterwards.

## Monitoring

Prometheus queries that answer most questions:

```
sum by (instance) (nvidia_smi_utilization_gpu_ratio)       # GPU load per tray
sched_gpus_alloc / sched_gpus                               # allocation (either scheduler)
sched_jobs                                                  # jobs by state
rate(nvlink_gpu_tx_bytes_total[1m])                         # NVLink traffic per GPU
sum by (instance) (rate(node_infiniband_port_data_transmitted_bytes_total[1m]))   # InfiniBand per tray
idrac_system_power_on                                       # tray power from the BMCs
```

`curl -s --data-urlencode 'query=<q>' http://<bridge>.10:9090/api/v1/query | jq`

## Test

- `make test`: end to end (jobs, storage, BMCs, partitions, metrics), ~10 minutes.
- `make test-bmc`: BMC and partition integration tests, ~30 s; `make test-bmc-disruptive` power-cycles a tray (ask first).
- `make test-fakegpu`: the fake GPU libraries on the host, no lab needed.

## More

`docs/` in the repository: `slurm.md`, `kubernetes.md`, `fake-gpu.md`, `nvlink-partitions.md`, `bmc-redfish.md`, `monitoring.md`, `topology.md`, `platform.md` (configuration, CPU placement, troubleshooting).

## Installing this skill

Claude Code, available in every project:

```
mkdir -p ~/.claude/skills/ai-lab
curl -fsSL https://raw.githubusercontent.com/arkady-emelyanov/ai-lab/main/skills/ai-lab/SKILL.md -o ~/.claude/skills/ai-lab/SKILL.md
```

Other agents with Agent Skills support: copy `skills/ai-lab/` into their skills directory.
