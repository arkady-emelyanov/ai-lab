[← README](../README.md) · **Platform: host, Incus and Ansible**

# Platform: host, Incus and Ansible

## Overview

Everything runs on a single Linux host as Incus system containers, created and configured by Ansible playbooks behind a `Makefile`. Host-side preparation (Python venv, Ansible collections, secrets, Go builds of the lab's own services) is done by `make init`; nothing is installed on the host outside the repository and Incus itself.

## Usage

| Command | What it does |
|---|---|
| `make init` | Prepares the host side: `.venv` with Ansible, collections in `.collections/`, secrets in `.secrets/`, builds `fakebmc`, `fakenmxc`, `fakedp` and topograph, checks host prerequisites. Every other target runs it first. |
| `make up` | `init` + `provision` + `configure`: a complete cluster. |
| `make provision` | `playbooks/provision.yml`: Incus project, containers, static addresses, volumes, device nodes, sideband and telemetry wiring, BMC certificates. |
| `make configure` | `playbooks/site.yml`: configures every instance (idempotent; re-run after changing variables). |
| `make frameworks` | `playbooks/frameworks.yml`: PyTorch and Ray venv on `/shared` (several GB). |
| `make test` | `playbooks/test.yml`: end-to-end checks ([Testing](testing.md)). |
| `make test-fakegpu` | Fake GPU library tests on this machine, without the lab (pytest, `tests/fakegpu`; [Testing](testing.md#fake-gpu-library-tests)). |
| `make test-bmc`, `make test-bmc-disruptive`, `make test-bmc-conformance` | BMC integration tests from the host (pytest, `tests/bmc`; [Testing](testing.md#bmc-integration-tests)). |
| `make down` | Deletes all instances; volumes (homes and venv, object data and its JuiceFS metadata, scratch), secrets and caches stay. |
| `make purge` | Also deletes volumes, the scratch pool, the `trays` project and BMC certificates. |
| `make shell` / `make shell-root` / `make shell-<instance>` | Shells via `incus exec` ([Identity & access](identity-and-access.md)). |
| `make proto` | Regenerates the partition controller's gRPC code (needs the tools in `.cache/tools`). |

Commands that need the Incus socket run under `sg incus-admin`, so a group membership added in the current session works without logging in again.

## Configuration

- `inventory/group_vars/all.yml`: every setting and its default.
- `local.yml`: your own values (created by `make init`, git-ignored); it overrides `all.yml`.
- `inventory/hosts.yml`: the instances, groups, static addresses (`ip_host`) and tray numbers.

After a change, run `make configure`, or `make provision` for instance settings such as limits and devices. The exception is `scheduler` (`slurm` or `k3s`): it decides what is installed, so changing it takes `make down` and `make up` (volumes are kept).

| Group | Instances |
|---|---|
| `cluster` (`controller`, `login`, `workers`) | scheduler nodes (Slurm or k3s) |
| `storage` | `sched-storage` |
| `nvswitch` | `sched-nvswitch` |
| `bmc` (`tray_bmc`, `nvswitch_bmc`) | the three BMCs |

**Package mirror.** Instances install Ubuntu packages from `apt_mirror`, with short apt timeouts so an unresponsive mirror fails in seconds.

- `auto` (the default) picks the fastest mirror from this machine: among `apt_mirror_candidates` (well-connected mirrors on every continent) and the mirrors Ubuntu suggests for your location. The choice is cached for a day in `.cache/apt-mirror`; delete it to choose again.
- A URL fixes the mirror, e.g. `apt_mirror: http://mirrors.edge.kernel.org/ubuntu` in `local.yml` ([list of Ubuntu mirrors](https://launchpad.net/ubuntu/+archivemirrors)).
- To let `auto` consider a mirror near you, add it to `apt_mirror_candidates`.

**Hardware parameters.** The emulated hardware is described by settings that default to GB200 values:

- GPU profile, driver, CUDA and VBIOS versions, NVLink and InfiniBand rates: [Emulated GPUs](fake-gpu.md#configuration)
- BMC firmware, per tray if needed: [BMCs](bmc-redfish.md#configuration)
- switch chips and ports: [NVLink partitions](nvlink-partitions.md#configuration)
- the InfiniBand fabric: [Topology](topology.md#configuration)

`make configure` applies a change and rejects inconsistent values first. The lab's size (trays, GPUs per tray, switches) is fixed.

**CPU placement.** `instance_limits` gives each instance a memory limit and a CPU count. `make provision` then gives each GPU tray whole physical cores of its own and pins every other instance to the remaining cores. On a 6-core, 12-thread host, for example, the trays get cores 0–1 and 2–3, everything else 4–5.

Without pinning, Incus moves containers between cores, so a tray shares cores with the controller and its GPUs slow down (in k3s mode, pods also keep only the CPUs the tray had when the kubelet started). Two limits remain:

- Processes on the host itself are not confined, so heavy work on the host can still slow a tray.
- On a host with too few cores, the instances keep plain CPU counts.

**Secrets** (`.secrets/`, git-ignored, owner-only, created by `make init` or on first use):

| File | Used for |
|---|---|
| `munge.key` | Slurm authentication |
| `k3s.token` | k3s cluster join token; `kubeconfig` (written by `make up` in k3s mode) is the admin kubeconfig for `bin/kubectl` |
| `slurmdbd.pass`, `ldap-admin.pass`, `redis.pass`, `grafana.pass` | service passwords |
| `rustfs.access`, `rustfs.secret` | RustFS admin keys |
| `ssh/id_ed25519` | admin (root) SSH key; `ssh/controller_ed25519` lets the controller run pdsh on the trays |
| `bmc/*.crt`, `bmc/*.key`, `bmc/incus-server.crt` | tray BMC Incus client certificates, pinned Incus server certificate |
| `users/<name>.pass`, `users/<name>.s3` | generated user passwords (when none is set) and S3 secrets |

**Secrets and data volumes belong together.** The data volumes (`/shared`, the object store, the JuiceFS metadata) outlive `make down` and even the checkout, while `.secrets/` is regenerated whenever it is missing. The object store and JuiceFS only work with the credentials they were created with.

So `make provision` marks each data volume with a fingerprint of those secrets, and stops before changing anything if existing volumes carry another one. Then either:

- put back the `.secrets/` the volumes were created with, or
- delete the volumes with `make purge` (the lab's data goes with them) and run `make up`.

Keep `.secrets/` when you move or re-clone the checkout, and drive a lab from one checkout only.

**Playbook order** (`site.yml`):

1. base system, LDAP, SSSD and SSH, login tools
2. emulated GPUs
3. Slurm and accounting, or k3s
4. storage, per-user S3, JuiceFS and S3 clients
5. emulated InfiniBand, topograph
6. monitoring and Grafana
7. BMCs and the switch tray host

## Verification

```
make check                         # "host ready" or the exact fix for what is missing
make configure                     # a second run must report changed=0 on every host
sg incus-admin -c 'incus list --all-projects -c ns4'   # all instances RUNNING with their pinned addresses
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `make up` stops with "The lab's data volumes … were created with other secrets" | `.secrets/` is not the one the data volumes were created with (a new clone, or a second checkout driving the same lab). Put back the original `.secrets/`, or `make purge` (deletes the lab's data) and `make up`. Labs built before this check have no fingerprint and are adopted as they are: if one shows `Input/output error` on `/pfs` and `InvalidAccessKeyId` in the JuiceFS log, it has this mismatch; `make purge` and `make up` fix it |
| `make up` or `make configure` stalls for minutes at a package task (e.g. `ssh : Install the SSH server`) | the Ubuntu mirror stopped answering (with `apt_mirror: auto`, after it was chosen; Canonical's mirrors sometimes time out from some networks). Stop the run, delete `.cache/apt-mirror` (or set a fixed `apt_mirror` in `local.yml`, [Configuration](#configuration)) and start it again |
| `make check`: not in `incus-admin` | `sudo usermod -aG incus-admin $USER` |
| `make check`: containers get no DNS | Incus' dnsmasq is denied reading NetworkManager's `no-stub-resolv.conf` by AppArmor. `make check` prints the drop-in to add under `/etc/apparmor.d/abstractions/nameservice.d/`. |
| `make check`: `fs.inotify.max_user_instances` too low; containers boot without network ("Too many open files") | Incus ships `/etc/sysctl.d/10-incus-inotify.conf` (1024) but it only applies after a reboot: `sudo sysctl --system`. |
| Container runtimes fail with `disk quota exceeded` (k3s pods stuck in `ContainerCreating`) | All unprivileged containers map root to one host uid and share its kernel keyring quota (200 keys by default). `make check` asks for Incus' recommended `kernel.keys.maxkeys=2000`, `kernel.keys.maxbytes=2000000` and prints the commands. |
| A node shows `down*` in Slurm after the host was suspended | `slurmctld` saw no answer for longer than `SlurmdTimeout` and marked the trays down; `slurmd` re-registers only when it starts. Resume them: `bin/ssh sched-control scontrol update nodename=sched-worker[1-2] state=resume`. |
| A node shows `down*` in Slurm after a restart | Addresses are static, so this should not happen; if it does, `make configure` restarts the Slurm daemons when `/etc/hosts` changes. |
| Host bind mounts with `shift=true` fail | Ubuntu's Incus 6.0.0 cannot do idmapped mounts on kernels ≥ 6.9 ([lxc/incus#882](https://github.com/lxc/incus/issues/882)); the lab uses Incus volumes instead. |
| One tray's GPUs run at lower utilisation than the other's during the same job (k3s mode especially) | The tray competes for CPU. Check each container's cores, `bin/ssh sched-worker1 cat /sys/fs/cgroup/cpuset.cpus.effective`, and in k3s mode the pods', `cat /sys/fs/cgroup/kubepods/cpuset.cpus.effective`: the trays must have distinct cores shared with no other instance, and the pods the tray's cores. `/sys/fs/cgroup/cpu.pressure` in the tray shows the stall. `make provision` restores the pinning and restarts a re-pinned tray's kubelet (which copies the tray's cores into the pod cgroup only when it starts). Work on the host itself is not confined and can still slow a tray. |
| topograph build fails with `GOSUMDB=off` | The Makefile enables the checksum database for that one build; a newer Go toolchain is downloaded automatically. |

## References

- [Incus documentation](https://linuxcontainers.org/incus/docs/main/)
- [Ansible community.general.incus connection](https://docs.ansible.com/ansible/latest/collections/community/general/incus_connection.html)
- [Architecture](architecture.md)
