[← README](../README.md) · **Monitoring**

# Monitoring

## Overview

Prometheus on `sched-control` scrapes every layer; Grafana on the same host presents it.

| Job | Targets | Exporter | Covers |
|---|---|---|---|
| `node` | all cluster nodes `:9100` | node_exporter (+ textfile collector) | hosts; on the controller also Slurm GPU allocation (Slurm mode) |
| `gpu` | trays `:9835`, every 5 s | [nvidia_gpu_exporter](https://github.com/utkuozdemir/nvidia_gpu_exporter) v1.15.1, NVML backend | per-GPU utilisation, memory, power, temperature, clocks, processes |
| `slurm` (Slurm mode) | controller `:9092`, every 30 s | [prometheus-slurm-exporter](https://github.com/rivosinc/prometheus-slurm-exporter) v1.8.0 (CLI mode) | nodes, CPUs, memory, partitions, jobs per account/user and state |
| `nvlink` | `sched-nvswitch:9372` | `fakenmxc` | NVLink partitions, link state, traffic, switch ports ([NVLink partitions](nvlink-partitions.md#usage)) |
| `juicefs` | all cluster nodes `:9567` | JuiceFS client | `/pfs` operations and throughput |
| `kube-state-metrics` (k3s mode) | controller NodePort `:30808` | [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) | nodes and their allocatable GPUs, pods' GPU requests, jobs |

**Slurm GPU allocation.** The Slurm exporter has no GPU metrics; `slurm-gpu-metrics` (systemd timer, every 15 s) writes them for node_exporter's textfile collector on the controller:

| Metric | Labels |
|---|---|
| `slurm_node_gpus`, `slurm_node_gpus_alloc` | `node`, `type` |
| `slurm_partition_gpus`, `slurm_partition_gpus_alloc` | `partition` |
| `slurm_job_gpus` | `job_id`, `user`, `account`, `partition` |

**Scheduler-neutral series.** Recording rules (`/etc/prometheus/sched-rules.yml`) compute the same series from Slurm's or from kube-state-metrics' data, and the dashboards use only these, so they work with either scheduler:

| Series | Labels | Meaning |
|---|---|---|
| `sched_gpus`, `sched_gpus_alloc` | | GPUs schedulable, allocated to running jobs/pods |
| `sched_node_gpus_alloc` | `node` | allocated GPUs per tray |
| `sched_user_gpus` | `user` | allocated GPUs per user (k3s: per namespace) |
| `sched_jobs` | `state` | jobs by state (`RUNNING`, `PENDING`, ...; k3s: Jobs in user namespaces, pending = suspended by Kueue) |
| `sched_nodes` | `state` | trays by scheduler state (k3s: `READY`, `NOT_READY`) |

GPU metrics reflect the simulated load ([Fake GPUs](fake-gpu.md#how-it-works)): they move with real jobs.

## Usage

| UI | URL (from your machine) | Login |
|---|---|---|
| Grafana | `http://10.107.111.10:3000` | `admin`, password in `.secrets/grafana.pass` |
| Prometheus | `http://10.107.111.10:9090` | none |

Dashboards (folder *Lab*):

| Dashboard | Content |
|---|---|
| **Lab overview** | GPU utilisation, memory, power and temperature per tray and GPU; processes on GPUs; domain power; JuiceFS throughput |
| **Scheduler & NVLink fabric** | GPUs allocated vs total, GPUs per node and user, jobs and nodes by state; unhealthy GPUs, switch ports down, partitions, NVLink and NVSwitch throughput |
| **Nvidia GPU Metrics** | the GPU exporter's own dashboard (grafana.com 14574) |
| **Node Exporter Full** | grafana.com 1860 |

Useful queries:

```
sum by (instance) (nvidia_smi_utilization_gpu_ratio)          # GPU load per tray
sched_gpus_alloc / sched_gpus                                  # GPU allocation ratio (either scheduler)
sched_user_gpus                                                # GPUs per user
rate(nvlink_gpu_tx_bytes_total[1m])                            # NVLink traffic per GPU
```

## Verification

```
curl -s http://10.107.111.10:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job) \(.labels.instance) \(.health)"'
bin/ssh slurm 'sbatch -N1 --gpus-per-node=3 --wrap "sleep 60"'; sleep 20     # Slurm mode
curl -s 'http://10.107.111.10:9090/api/v1/query?query=sched_gpus_alloc' | jq '.data.result[].value[1]'   # "3"
```

`make test` checks that every target is up, that GPU power is reported for all 8 GPUs, that the scheduler (`sched_*`) and fabric series exist and that Grafana serves the dashboards.

## Configuration

| Variable | Default |
|---|---|
| `gpu_exporter_version`, `gpu_exporter_port` | 1.15.1, 9835 |
| `slurm_exporter_version`, `slurm_exporter_port` | 1.8.0, 9092 (Slurm mode) |
| `ksm_chart_version`, `ksm_node_port` | 8.6.0, 30808 (k3s mode) |
| `prometheus_retention` | 15d |
| `grafana_port`, `grafana_dashboards` | 3000, pinned grafana.com revisions |

## References

- [Prometheus](https://prometheus.io/docs/), [Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [nvidia_gpu_exporter](https://github.com/utkuozdemir/nvidia_gpu_exporter), [rivosinc/prometheus-slurm-exporter](https://github.com/rivosinc/prometheus-slurm-exporter), [JuiceFS monitoring](https://juicefs.com/docs/community/administration/monitoring/)
- Roles: `roles/monitoring` (exporters, `slurm-gpu-metrics`, Prometheus), `roles/grafana`
