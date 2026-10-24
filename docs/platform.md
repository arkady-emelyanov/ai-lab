[← README](../README.md) · **Platform: host, Incus and Ansible**

# Platform: host, Incus and Ansible

## Overview

Everything runs on a single Linux host as Incus system containers, created and configured by Ansible playbooks behind a `Makefile`. Host-side preparation (Python venv, Ansible collections, secrets, Go builds of the lab's own services) is done by `make init`; nothing is installed on the host outside the repository and Incus itself.

## Usage

| Command | What it does |
|---|---|
| `make init` | Prepares the host side: `.venv` with Ansible, collections in `.collections/`, secrets in `.secrets/`, builds `fakebmc`, `fakenmxc`, `fakedp` and topograph, checks host prerequisites. Every target that works on the lab runs it first. |
| `make up` | `init` + `provision` + `configure`: a complete cluster. |
| `make provision` | `playbooks/provision.yml`: Incus project, containers, static addresses, volumes, device nodes, sideband and telemetry wiring, BMC certificates. |
| `make configure` | `playbooks/site.yml`: configures every instance (idempotent; re-run after changing variables). |
| `make frameworks` | `playbooks/frameworks.yml`: PyTorch and Ray venv on `/shared` (several GB). |
| `make test` | `playbooks/test.yml`: end-to-end checks ([Testing](testing.md)). |
| `make test-fakegpu` | Fake GPU library tests on this machine, without the lab (pytest, `tests/fakegpu`; [Testing](testing.md#fake-gpu-library-tests)). |
| `make test-fakedp` | Unit tests of the GPU device plugin on this machine, without the lab (`go vet`, `go test -race` in `fakedp`; [Testing](testing.md#device-plugin-tests)). |
| `make test-bmc`, `make test-bmc-disruptive`, `make test-bmc-conformance` | BMC integration tests from the host (pytest, `tests/bmc`; [Testing](testing.md#bmc-integration-tests)). |
| `make stop`, `make start` | `playbooks/power.yml`: stops every instance, GPU trays first and storage last, or starts them in the reverse order. Instances, data and monitoring history are kept; the services take a minute or two to come back after `make start`. |
| `make down` | Deletes all instances; volumes (homes and venv, object data and its JuiceFS metadata, scratch), secrets and caches stay. |
| `make purge` | Also deletes volumes, the scratch pool, the `trays` project and BMC certificates. |
| `make shell` / `make shell-root` / `make shell-<instance>` | Shells via `incus exec` ([Identity & access](identity-and-access.md)). |
| `make proto` | Regenerates the partition controller's gRPC code (needs the tools in `.cache/tools`). |

Commands that need the Incus socket run under `sg incus-admin`, so a group membership added in the current session works without logging in again.

## Configuration

- `inventory/group_vars/all.yml`: every setting and its default.
- `local.yml`: your own values (created by `make init`, git-ignored); it overrides `all.yml`.
- `inventory/hosts.yml`: the instances, groups, static addresses (`ip_host`) and tray numbers.

After a change, run `make configure`, or `make provision` for instance settings such as limits and devices (it applies changed CPU and memory limits to running instances and restarts k3s there, so its nodes report their new capacity). The exceptions are `scheduler` (`slurm` or `k3s`), which decides what is installed, and turning `nvidia_dra` off: `make configure` does not remove what it installed, so both take `make down` and `make up` (volumes are kept). Turning `kuberay` off, `make configure` deletes KubeRay with its Ray clusters and jobs (Ray's CRDs stay).

| Group | Instances |
|---|---|
| `cluster` (`controller`, `login`, `workers`) | scheduler nodes (Slurm or k3s) |
| `storage` | `sched-storage` |
| `nvswitch` | `sched-nvswitch` |
| `bmc` (`tray_bmc`, `nvswitch_bmc`) | the three BMCs |

### Components

The controller, the login node, the GPU trays, the scheduler and the directory are always there; the components below can be switched off (and on again) in `local.yml`. All are on by default.

| Setting | Component | Instances | Without it |
|---|---|---|---|
| `bmc` | Redfish BMCs of the GPU trays and the NVLink switch tray ([BMCs](bmc-redfish.md)) | `sched-worker1-bmc`, `sched-worker2-bmc`, `sched-nvswitch-bmc` | no out-of-band power, sensors or NVLink port control; no Redfish exporter, no BMC panels in Grafana; `bin/redfish` and `make test-bmc` have nothing to talk to |
| `nvswitch` | NVLink switch tray host: partition controller and fabric telemetry ([NVLink partitions](nvlink-partitions.md)) | `sched-nvswitch` (and `sched-nvswitch-bmc`) | the GPUs keep one NVLink domain and their partitions (the default one on a new lab); no `bin/nvlink`, no fabric metrics or NVLink fabric panels |
| `storage` | storage node: RustFS (S3) and Redis, JuiceFS at `/pfs` and per-user S3 on every node ([Storage](storage.md)) | `sched-storage` | no `/pfs`, no S3 (`rc`, `~/.aws`); `/shared` and `/scratch` stay. KubeRay needs it (`kuberay: false` without it) |
| `monitoring` | Prometheus, Grafana and the exporters: node, GPU, Slurm, Redfish, kube-state-metrics ([Monitoring](monitoring.md)) | none | no metrics or dashboards; Ray's dashboard shows no metrics. Run:ai keeps its own Prometheus |
| `topology` | topograph: the scheduler's topology follows NVLink partitions and the InfiniBand fabric ([Topology](topology.md)) | none | the topology comes from the settings once: one NVLink block in Slurm's `topology.conf`, the same node labels in Kubernetes (Kueue's topology-aware scheduling still works), not following partition changes |
| `slurm_accounting` | Slurm accounting: slurmdbd and MariaDB on the controller ([Slurm](slurm.md)) | none | no `sacct`, `sreport`, accounts or limits (Slurm mode only) |

A change takes `make provision` and `make configure`, or `make up`, which runs both:

- `make provision` creates the instances of a component switched on, and deletes those of one switched off, as `make down` would (their volumes stay). Before that it clears what the instance set on the trays: a tray BMC's disabled NVLink ports, the switch tray BMC's disabled switch ports, the partition controller's partitions (the GPUs return to the default partition at their next reset). Without BMCs it also revokes their Incus certificates and closes the Incus API listener they used.
- `make configure` installs the component, or removes it from the instances that stay, and reconfigures what uses it: `/etc/hosts`, Prometheus' scrape jobs, Grafana's dashboards, Slurm's configuration, Ray's metrics settings, `endpoints.md`. A component removed has its services stopped and disabled, with their data kept (Prometheus' history, the accounting database, `/pfs` on its volumes), so switching it on again brings them back. The users' S3 settings in their homes go with `storage`.
- `make configure` refuses combinations that cannot work: `kuberay: true` (k3s) needs `storage: true`.

`make test` checks that the lab has exactly the instances its settings give it, and skips the checks of components that are off. `make stop`, `make start`, `make down` and `make purge` act on every instance that exists, whatever the settings.

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

Each optional component is installed when it is on and removed (its `absent` tasks) when it is off; `enabled.yml`, imported first, groups the instances the settings enable, so plays on optional instances run on `<group>:&enabled`.

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
| k3s pods stuck in `CreateContainerError` with "failed to reserve container name … is reserved for …" | k3s (with its containerd) restarted while it was creating the container: the container runs, but the kubelet never learned its ID and every retry collides with its name. Delete the pod (`bin/kubectl -n <ns> delete pod <pod>`); its controller creates a new one. |
| Pods with volumes on the controller stay `Pending`, local-path logs `helper pod … not found` | local-path's helper pod was evicted by the controller's `CriticalAddonsOnly` taint. `make configure` lets it tolerate the taint (and restarts local-path); after a k3s upgrade that rewrites local-path's configuration, run it again. |
| Pods fail with `ErrImagePull`: "pull QPS exceeded" | kubelet's own limit on image pulls (5/s), hit when a chart starts many pods at once. The lab sets `registry-qps=0`; pulls of `docker.io` images go through `k3s_registry_mirrors` (`mirror.gcr.io`). |
| Run:ai: the cluster stays `WaitingToConnect`, or its pods `Pending` with "Insufficient cpu" | Its services run on the controller: check `make provision` gave it `instance_limits_runai` (`bin/kubectl describe node sched-control`, Allocatable) and that the services' small requests are applied (`make configure`); `bin/kubectl -n runai logs deploy/runai-operator` names other causes. |
| Run:ai GPU fractions: pods fail with "no runtime for \"nvidia\" is configured", or stay `Pending` ("node is not ready for fractional GPU scheduling") | KAI decides between NVIDIA's runtime class and the NRI path from the `ClusterPolicy` when its operator starts: after the `ClusterPolicy` was added to a running Run:ai, `bin/kubectl -n runai rollout restart deploy/kai-operator`, then delete the failed pods. Not ready: `bin/kubectl get nodes -o custom-columns=N:.metadata.name,FRACTIONS:'.status.conditions[?(@.type=="gpu-fractioning.nvidia.com/Ready")].message'` and the `gpu-fractioning-*` pods in `runai` ([NVIDIA Run:ai](runai.md#gpu-fractions)). |
| topograph build fails with `GOSUMDB=off` | The Makefile enables the checksum database for that one build; a newer Go toolchain is downloaded automatically. |

## References

- [Incus documentation](https://linuxcontainers.org/incus/docs/main/)
- [Ansible community.general.incus connection](https://docs.ansible.com/ansible/latest/collections/community/general/incus_connection.html)
- [Architecture](architecture.md)
