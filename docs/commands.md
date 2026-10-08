[← Docs](README.md) · **Commands**

# Commands

All from the repository's root.

## The lab

| Command | Purpose |
|---|---|
| `make init` | prepare the host: tools, secrets, `local.yml`; reports anything the host still needs |
| `make up` | build the lab |
| `make configure` | apply changed settings (`local.yml`) to the running lab |
| `make frameworks` | install PyTorch and Ray into the lab (several GB) |
| `make down` | delete the instances; volumes are kept |
| `make purge` | delete everything, including the lab's data |

## Access

| Command | Purpose |
|---|---|
| `bin/ssh login` | log in to the login node as `joe` |
| `bin/ssh root@<instance>` | log in to any instance as root |
| `bin/scp`, `bin/ssh-copy-id` | the same for copying files and installing your key |
| `bin/kubectl` | kubectl as cluster admin (k3s mode) |
| `make shell`, `make shell-<instance>` | a shell through `incus exec`, without SSH |

## Hardware

| Command | Purpose |
|---|---|
| `bin/nvlink` | NVLink domain and partitions: list, create, delete, add or remove GPUs ([NVLink partitions](nvlink-partitions.md)) |
| `bin/redfish <tray> <path>` | Redfish request to a tray's BMC ([BMCs](bmc-redfish.md)) |
| `bin/ssh sched-control update-topology` | regenerate the scheduler's topology now; a timer does it every minute ([Topology discovery](topology.md)) |
| `bin/grpcurl` | raw gRPC calls to the partition controller ([NVLink partitions](nvlink-partitions.md)) |

## Tests

| Command | Purpose |
|---|---|
| `make test` | end-to-end checks of the running lab |
| `make test-fakegpu` | the GPU library tests, on this machine, no lab needed |
| `make test-bmc` | BMC and partition controller integration tests |
| `make test-bmc-disruptive` | the same, plus power-cycling a tray |
| `make test-bmc-conformance` | GB200 behaviour the lab does not model yet |

More in [Testing](testing.md).
