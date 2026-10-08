[← README](../README.md) · **Lab endpoints**

# Lab endpoints

Reachable from the host machine. Addresses are on the Incus bridge (`10.107.111.0/24` here; yours may differ, see `incus network get incusbr0 ipv4.address`).

| Service | Endpoint | Credentials | Docs |
|---|---|---|---|
| Login node (SSH) | `bin/ssh login` (10.107.111.11:22) | `joe` / `joe`, or your key | [Identity and access](identity-and-access.md) |
| Any instance as root (SSH) | `bin/ssh root@<instance>` | `.secrets/ssh/id_ed25519` | [Identity and access](identity-and-access.md) |
| Grafana | http://10.107.111.10:3000 | `admin` / `.secrets/grafana.pass` | [Monitoring](monitoring.md) |
| Prometheus | http://10.107.111.10:9090 | none | [Monitoring](monitoring.md) |
| Slurm exporter (Slurm mode) | http://10.107.111.10:9092/metrics | none | [Monitoring](monitoring.md) |
| Kubernetes API (k3s mode) | https://10.107.111.10:6443 (`bin/kubectl`) | admin: `.secrets/kubeconfig`; users: `~/.kube/config` | [Kubernetes](kubernetes.md) |
| kube-state-metrics (k3s mode) | http://10.107.111.10:30808/metrics | none | [Monitoring](monitoring.md) |
| topograph API | http://10.107.111.10:49021 | none | [Topology discovery](topology.md) |
| RustFS S3 API | http://10.107.111.12:9000 | admin: `.secrets/rustfs.access` / `.secrets/rustfs.secret`; users: `<name>` / `.secrets/users/<name>.s3` | [Storage](storage.md) |
| RustFS console | http://10.107.111.12:9001/rustfs/console/ | as S3 API | [Storage](storage.md) |
| GPU tray BMCs (Redfish) | https://10.107.111.31, https://10.107.111.32 (`bin/redfish sched-worker1 …`) | `root` / `0penBmc` | [BMCs](bmc-redfish.md) |
| NVLink switch tray BMC (Redfish) | https://10.107.111.33 (`bin/redfish sched-nvswitch …`) | `root` / `0penBmc` | [BMCs](bmc-redfish.md) |
| NVLink partition controller (gRPC) | 10.107.111.34:9370 (plaintext, reflection) | none | [NVLink partitions](nvlink-partitions.md) |
| Fabric telemetry | http://10.107.111.34:9372/metrics | none | [NVLink partitions](nvlink-partitions.md) |
| GPU exporters | http://10.107.111.21:9835/metrics, http://10.107.111.22:9835/metrics | none | [Monitoring](monitoring.md) |
| node_exporter, JuiceFS metrics | `<cluster node>:9100/metrics`, `<cluster node>:9567/metrics` | none | [Monitoring](monitoring.md) |

The BMCs use self-signed certificates (`curl -k`).
