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
| `redfish` | all BMCs, through the controller `:9348`, every 30 s | [idrac_exporter](https://github.com/mrlhansen/idrac_exporter) v2.6.3, a generic Redfish exporter (iDRAC, iLO, XClarity, Supermicro, OpenBMC) | out of band: tray power state, GPU temperatures, system and BMC health, machine info ([BMCs](bmc-redfish.md)) |
| `node` (InfiniBand) | trays' node_exporter textfile collector, every 15 s | `ib-port-counters` (`roles/fakeib`) | NIC port counters under node_exporter's infiniband names: `node_infiniband_port_data_transmitted_bytes_total`, `..._received_bytes_total` `{device="mlx5_<gpu>", port="1"}`; NCCL traffic between NVLink partitions |
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

**Out of band.** The `redfish` job polls the BMCs, not the trays, so it keeps reporting while a tray's OS is down. Its series carry a `tray` label:

- tray powered off through its BMC: `idrac_system_power_on 0`, with `up{job="redfish"} 1`
- tray's GPU exporter stopped: `up{job="gpu"} 0`, with power still on
- BMC unreachable: `up{job="redfish"} 0`

| Metric | Labels | Meaning |
|---|---|---|
| `idrac_system_power_on` | `tray` | 1 when the BMC reports the tray `PowerState: On` |
| `idrac_system_health`, `idrac_manager_health` | `status` | 0 OK, 1 Warning, 2 Critical |
| `idrac_system_machine_info` | `hostname`, `manufacturer`, `model` | tray identity from the BMC |
| `idrac_sensors_temperature` | `name` (`GPU_<n>`) | GPU temperature read by the BMC; equals `nvidia_smi_temperature_gpu` of the same GPU |

GPU metrics reflect the simulated load ([Emulated GPUs](fake-gpu.md#how-it-works)): they move with real jobs.

## Usage

| UI | URL (from your machine) | Login |
|---|---|---|
| Grafana | `http://10.107.111.10:3000` | `admin`, password in `.secrets/grafana.pass` |
| Prometheus | `http://10.107.111.10:9090` | none |

Dashboards (folder *Lab*):

| Dashboard | Content |
|---|---|
| **Lab overview** | GPU utilisation, memory, power and temperature per tray and GPU; processes on GPUs; domain power; JuiceFS throughput; tray power from the BMCs next to the GPU exporters' state; GPU temperature in band (NVML) vs out of band (BMC) |
| **Scheduler & NVLink fabric** | GPUs allocated vs total, GPUs per node and user, jobs and nodes by state; unhealthy GPUs, switch ports down, partitions, NVLink and NVSwitch throughput; InfiniBand traffic per tray and NVLink vs InfiniBand for the domain |
| **Nvidia GPU Metrics** | the GPU exporter's own dashboard (grafana.com 14574) |
| **Node Exporter Full** | grafana.com 1860 |

Useful queries:

```
sum by (instance) (nvidia_smi_utilization_gpu_ratio)          # GPU load per tray
sched_gpus_alloc / sched_gpus                                  # GPU allocation ratio (either scheduler)
sched_user_gpus                                                # GPUs per user
rate(nvlink_gpu_tx_bytes_total[1m])                            # NVLink traffic per GPU
idrac_system_power_on == 0                                     # trays powered off (BMC view)
```

## Verification

```
curl -s http://10.107.111.10:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job) \(.labels.instance) \(.health)"'
bin/ssh login 'sbatch -N1 --gpus-per-node=3 --wrap "sleep 60"'; sleep 20     # Slurm mode
curl -s 'http://10.107.111.10:9090/api/v1/query?query=sched_gpus_alloc' | jq '.data.result[].value[1]'   # "3"
curl -s 'http://10.107.111.10:9348/metrics?target=sched-worker1-bmc' | grep power_on          # the exporter, scraping a BMC now
```

`make test` checks that every target is up, that GPU power is reported for all 8 GPUs, that the scheduler (`sched_*`), fabric and BMC power series exist and that Grafana serves the dashboards.

## Configuration

| Variable | Default |
|---|---|
| `gpu_exporter_version`, `gpu_exporter_port` | 1.15.1, 9835 |
| `slurm_exporter_version`, `slurm_exporter_port` | 1.8.0, 9092 (Slurm mode) |
| `redfish_exporter_version`, `redfish_exporter_port` | 2.6.3, 9348 |
| `ksm_chart_version`, `ksm_node_port` | 8.6.0, 30808 (k3s mode) |
| `prometheus_retention` | 15d |
| `grafana_port`, `grafana_dashboards` | 3000, pinned grafana.com revisions |

## References

- [Prometheus](https://prometheus.io/docs/), [Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [nvidia_gpu_exporter](https://github.com/utkuozdemir/nvidia_gpu_exporter), [idrac_exporter](https://github.com/mrlhansen/idrac_exporter), [rivosinc/prometheus-slurm-exporter](https://github.com/rivosinc/prometheus-slurm-exporter), [JuiceFS monitoring](https://juicefs.com/docs/community/administration/monitoring/)
- Roles: `roles/monitoring` (exporters, Redfish exporter, `slurm-gpu-metrics`, Prometheus), `roles/grafana`
