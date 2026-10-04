[← README](../README.md) · **Testing**

# Testing

## Overview

`make test` (`playbooks/test.yml`) checks the cluster end to end, from the user's point of view where possible: jobs run through Slurm as the directory user `joe`, and every management and monitoring interface is queried the way a client would.

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
| Slurm | both trays available (waits out jobs still completing) |
| Jobs as `joe` | `gpu-topology`, `nvl8-hello`, `ddp-train`, `ray-cluster` run with `sbatch --wait`; their output contains the fabric line, rank 7, all 8 DDP ranks, 8 Ray tasks |
| Accounting | every job above is `COMPLETED` in `sacct` |
| Shared filesystem | `/pfs` mounted on every Slurm node; a file written on the login node reads back everywhere |
| Object storage | RustFS healthy; from a job, `joe` writes and lists his bucket and is denied the JuiceFS bucket |
| BMCs | tray BMCs report their tray powered on with 4 GPUs; the switch BMC exposes 72 ports per switch |
| Partition controller | gRPC port open; every GPU reports the domain UUID and a clique |
| Topology | topograph generates a block containing both trays |
| Monitoring | every Prometheus target up; power for all 8 GPUs; Slurm GPU and NVLink fabric series present; Grafana serves the dashboards |

Idempotency is checked separately: a second `make configure` must report `changed=0` on every host.

## References

- `playbooks/test.yml`
- Per-component verification steps: see the *Verification* section of each page linked from the [README](../README.md#layers-and-components)
