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

All tunables and their defaults live in `inventory/group_vars/all.yml`; your own values go in `local.yml` (created by `make init`, not in git), which overrides it (Ansible extra vars). The instance list, groups, static addresses (`ip_host`) and tray numbers in `inventory/hosts.yml`. Change a value and run `make configure` (or `make provision` for instance-level settings such as limits and devices). The exception is `scheduler` (`slurm` or `k3s`): it decides what is installed and how instances are created, so changing it on a built cluster takes `make down` and `make up` (volumes are kept).

| Group | Instances |
|---|---|
| `cluster` (`controller`, `login`, `workers`) | scheduler nodes (Slurm or k3s) |
| `storage` | `sched-storage` |
| `nvswitch` | `sched-nvswitch` |
| `bmc` (`tray_bmc`, `nvswitch_bmc`) | the three BMCs |

**Package mirror.** Every instance installs Ubuntu packages from `apt_mirror` (archive and security pockets), with short apt timeouts so an unresponsive mirror address fails in seconds. The default, `auto`, picks the fastest mirror from this machine: the host checks `apt_mirror_candidates` (well-connected mirrors on every continent) and the mirrors Ubuntu suggests for its location, keeps those that serve the release, times a download from the quickest few and uses the fastest. The choice is cached for a day in `.cache/apt-mirror` (delete it to choose again). To use a fixed mirror, set its URL in `local.yml`, e.g. `apt_mirror: http://mirrors.edge.kernel.org/ubuntu` (list: [Ubuntu archive mirrors](https://launchpad.net/ubuntu/+archivemirrors)); to let `auto` consider one near you, add it to `apt_mirror_candidates`.

**Hardware parameters.** The emulated hardware is described by variables, each defaulting to GB200 values: the GPU profile, driver, CUDA and VBIOS versions, NVLink domain and InfiniBand link rate in [Emulated GPUs](fake-gpu.md#configuration), BMC firmware (per tray if needed) in [BMCs](bmc-redfish.md#configuration), switch chips and ports in [NVLink partitions](nvlink-partitions.md#configuration), the InfiniBand fabric in [Topology](topology.md#configuration). `make configure` applies a change; `make up` and `make configure` reject inconsistent values first. The lab's size (trays, GPUs per tray, switch trays, InfiniBand switches) is fixed.

**CPU placement.** `instance_limits` gives each instance a memory limit and a CPU count; `make provision` turns the CPU count of the GPU trays into whole physical cores of their own, from the host's topology (`lscpu`), and pins every other instance to the remaining cores (`playbooks/files/cpu-placement`; on this lab's 6-core, 12-thread host: trays on cores 0–1 and 2–3, everything else on 4–5). With a plain count Incus places and rebalances containers itself and lets them overlap: a tray then shares cores with the controller, and in k3s mode its pods keep only the CPUs the tray had when the kubelet started (the kubelet copies them into the pod cgroup once), so a long-running tray's GPUs slowed down. A tray whose pinning changes gets its kubelet restarted. Processes on the host itself are not confined: heavy work on the host can still slow a tray. If the host has too few cores, the count-based limits stay.

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

**Secrets and data volumes belong together.** The data volumes (`/shared`, the object store and its JuiceFS metadata) outlive `make down` and the checkout, while `.secrets/` is regenerated whenever it is missing (a fresh clone, a deleted directory). The object store and JuiceFS only work with the credentials they were created with, so `make provision` records a fingerprint of `rustfs.access`, `rustfs.secret` and `redis.pass` on each data volume (`user.ai-lab.secrets`) and stops, before changing anything, when existing volumes carry another one. Then either put back the `.secrets/` they were created with, or delete them with `make purge` (the lab's data goes with them) and run `make up`. Keep `.secrets/` when you move or re-clone the checkout, and drive a lab from one checkout only: two checkouts have different secrets but the same instances and volumes.

**Playbook order** (`site.yml`): base system (packages, `/etc/hosts`, munge for Slurm) → LDAP → SSSD and SSH → login tools → emulated GPUs → Slurm and accounting, or k3s (server, agents with GPUs, add-ons and users, kubectl) → storage and per-user S3 → JuiceFS and S3 clients → emulated InfiniBand → topograph → monitoring → Grafana → BMCs → switch tray host.

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
