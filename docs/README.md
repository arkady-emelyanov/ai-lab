[← README](../README.md) · **Documentation**

# Documentation

## The lab

- [What is real and what is modelled](real-and-modelled.md): what behaves like real hardware and software, what doesn't, and how timing works
- [Architecture](architecture.md): the instances, the networks and fabrics, and how the pieces connect
- [Lab endpoints](endpoints.md): addresses and credentials of every service
- [Commands](commands.md): every `make` target and `bin/` tool
- [Platform](platform.md): make targets, configuration, CPU placement, secrets, troubleshooting

## Emulated hardware

- [Emulated GPUs](fake-gpu.md): CUDA, NVML, NCCL, `nvidia-smi` and `numactl`, the timing and load model, GPU settings
- [BMCs](bmc-redfish.md): Redfish BMCs for the GPU trays and the NVLink switch tray
- [NVLink partitions](nvlink-partitions.md): the partition controller, `bin/nvlink`, fabric metrics
- [Topology discovery](topology.md): the InfiniBand fabric and topograph's scheduler topology

## Services

- [Identity and access](identity-and-access.md): LDAP users, SSH, `bin/ssh`
- [Storage](storage.md): `/shared`, `/pfs`, `/scratch` and S3
- [Monitoring](monitoring.md): Prometheus, exporters and Grafana dashboards

## Scheduling and jobs

- [Slurm](slurm.md): GPU scheduling, accounting, block topology, GPU handover
- [Kubernetes (k3s)](kubernetes.md): GPU pods, Kueue, JobSets, users
- [ComputeDomains (IMEX)](compute-domains.md): NVIDIA's DRA driver on k3s, IMEX channels and daemons for memory shared over NVLink between trays
- [KubeRay](kuberay.md): Ray clusters and jobs on k3s, the KubeRay dashboard and API server
- [NVIDIA Run:ai](runai.md): the open-source Run:ai control plane and cluster on k3s, KAI Scheduler, the DCGM exporter on the emulated GPUs
- [Frameworks and examples](frameworks-and-examples.md): the PyTorch and Ray environment, the example jobs and their expected results
- [Inference servers](inference.md): vLLM and SGLang on the emulated GPUs, what works, tensor parallelism over NVLink and InfiniBand

## Testing

- [Testing](testing.md): `make test`, the GPU library and device plugin tests and the BMC integration tests
