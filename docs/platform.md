[← README](../README.md) · **Platform: host, Incus and Ansible**

# Platform: host, Incus and Ansible

## Overview

Everything runs on one Linux host as Incus system containers, created and configured by Ansible playbooks behind a `Makefile`. Host-side preparation (Python venv, Ansible collections, secrets, Go builds of the lab's own services) is done by `make init`; nothing is installed on the host outside the repository and Incus itself.

## Usage

| Command | What it does |
|---|---|
| `make init` | Prepares the host side: `.venv` with Ansible, collections in `.collections/`, secrets in `.secrets/`, builds `fakebmc`, `fakenmxc` and topograph, checks host prerequisites. Every other target runs it first. |
| `make up` | `init` + `provision` + `configure`: a complete cluster. |
| `make provision` | `playbooks/provision.yml`: Incus project, containers, static addresses, volumes, device nodes, sideband and telemetry wiring, BMC certificates. |
| `make configure` | `playbooks/site.yml`: configures every instance (idempotent; re-run after changing variables). |
| `make frameworks` | `playbooks/frameworks.yml`: PyTorch and Ray venv on `/shared` (several GB). |
| `make test` | `playbooks/test.yml`: end-to-end checks ([Testing](testing.md)). |
| `make down` | Deletes all instances; volumes (homes and venv, object data and its JuiceFS metadata, scratch), secrets and caches stay. |
| `make purge` | Also deletes volumes, the scratch pool, the `trays` project and BMC certificates. |
| `make shell` / `make shell-root` / `make shell-<instance>` | Shells via `incus exec` ([Identity & access](identity-and-access.md)). |
| `make proto` | Regenerates the partition controller's gRPC code (needs the tools in `.cache/tools`). |

Commands that need the Incus socket run under `sg incus-admin`, so a group membership added in the current session works without logging in again.

## Configuration

All tunables live in `inventory/group_vars/all.yml`; the instance list, groups, static addresses (`ip_host`) and tray numbers in `inventory/hosts.yml`. Change a value and run `make configure` (or `make provision` for instance-level settings such as limits and devices).

| Group | Instances |
|---|---|
| `cluster` (`controller`, `login`, `workers`) | scheduler nodes (Slurm or k3s) |
| `storage` | `sched-storage` |
| `nvswitch` | `sched-nvswitch` |
| `bmc` (`tray_bmc`, `nvswitch_bmc`) | the three BMCs |

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

**Playbook order** (`site.yml`): base system (packages, `/etc/hosts`, munge for Slurm) → LDAP → SSSD and SSH → login tools → fake GPUs → Slurm and accounting, or k3s (server, agents with GPUs, add-ons and users, kubectl) → storage and per-user S3 → JuiceFS and S3 clients → fake InfiniBand → topograph → monitoring → Grafana → BMCs → switch tray host.

## Verification

```
make check                         # "host ready" or the exact fix for what is missing
make configure                     # a second run must report changed=0 on every host
sg incus-admin -c 'incus list --all-projects -c ns4'   # all instances RUNNING with their pinned addresses
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `make check`: not in `incus-admin` | `sudo usermod -aG incus-admin $USER` |
| `make check`: containers get no DNS | Incus' dnsmasq is denied reading NetworkManager's `no-stub-resolv.conf` by AppArmor. `make check` prints the drop-in to add under `/etc/apparmor.d/abstractions/nameservice.d/`. |
| `make check`: `fs.inotify.max_user_instances` too low; containers boot without network ("Too many open files") | Incus ships `/etc/sysctl.d/10-incus-inotify.conf` (1024) but it only applies after a reboot: `sudo sysctl --system`. |
| Container runtimes fail with `disk quota exceeded` (k3s pods stuck in `ContainerCreating`) | All unprivileged containers map root to one host uid and share its kernel keyring quota (200 keys by default). `make check` asks for Incus' recommended `kernel.keys.maxkeys=2000`, `kernel.keys.maxbytes=2000000` and prints the commands. |
| A node shows `down*` in Slurm after a restart | Addresses are static, so this should not happen; if it does, `make configure` restarts the Slurm daemons when `/etc/hosts` changes. |
| Host bind mounts with `shift=true` fail | Ubuntu's Incus 6.0.0 cannot do idmapped mounts on kernels ≥ 6.9 ([lxc/incus#882](https://github.com/lxc/incus/issues/882)); the lab uses Incus volumes instead. |
| topograph build fails with `GOSUMDB=off` | The Makefile enables the checksum database for that one build; a newer Go toolchain is downloaded automatically. |

## References

- [Incus documentation](https://linuxcontainers.org/incus/docs/main/)
- [Ansible community.general.incus connection](https://docs.ansible.com/ansible/latest/collections/community/general/incus_connection.html)
- [Architecture](architecture.md)
