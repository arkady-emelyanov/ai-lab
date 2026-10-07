# Instructions for coding agents

This file is for AI coding agents (Claude Code, Codex, Cursor, ...) working with this repository on a user's machine. People should start with the [README](README.md). Operating a running lab (jobs, GPUs, NVLink partitions, BMCs, metrics, tests) is covered by the agent skill [skills/ai-lab/SKILL.md](skills/ai-lab/SKILL.md) (Claude Code loads it in this repository through the `.claude/skills/ai-lab` link); keep it in step with the tools and docs.

The repository builds an emulated NVIDIA GB200 NVL8 GPU cluster on one Linux machine: Incus system containers configured by Ansible, driven by a `Makefile`. Nothing is installed on the host outside the repository and Incus itself.

## Deploying the lab for the user

When the user asks to set up, install, deploy or try the lab, offer to do it end to end. Do the work yourself; ask the user only for decisions and for the one-time steps that need root.

### 1. Check the host (no changes)

- Linux, x86-64. The lab is developed and tested on Ubuntu 24.04 (Incus and Go from its packages; the Makefile fetches a newer Go toolchain when a build needs one). On other distributions install Incus from their packages or [Incus' own instructions](https://linuxcontainers.org/incus/docs/main/installing/).
- Resources: at least 4 CPU cores, 16 GiB of free RAM (the running lab uses about 10 GiB; container limits add up to 27 GiB, 32 GiB in k3s mode), about 25 GB free under `/var/lib/incus`. Report what the machine has; if it falls short, say so and let the user decide.
- Tools: `incus`, `make`, `python3` with `venv`, `go`, `jq`, `git`. Note what is missing.
- Incus state: `incus info` (as the user, or through `sg incus-admin -c 'incus info'` when the group was just added). An Incus that was never initialised needs `incus admin init --minimal`.
- Run `make check` once the tools are there: it reports every host setting the lab still needs, with the exact command.

### 2. Ask for root once, with reasons

Collect everything that needs `sudo` from step 1 and `make check`, and ask the user once, listing each command with why it is needed. Never run `sudo` without the user's explicit consent, and run only what you listed. The possible items:

| Command | Why it needs root |
|---|---|
| `sudo apt install incus make python3-venv golang-go jq git` (or the distribution's equivalent) | installs Incus (the container manager the lab runs in) and the build tools; package installation is a system change |
| `sudo usermod -aG incus-admin $USER` | lets the user (and the Makefile) manage Incus containers without root; membership in a system group can only be granted by root. It works in the current session through `sg incus-admin`, so no re-login is needed |
| `sudo sysctl --system` | applies Incus' own `fs.inotify.max_user_instances` setting (shipped in `/etc/sysctl.d/`, normally applied at boot): without it the containers' systemd runs out of inotify instances and boots without network |
| `/etc/sysctl.d/60-incus-keys.conf` with `kernel.keys.maxkeys = 2000` and `kernel.keys.maxbytes = 2000000`, then `sudo sysctl --system` | every unprivileged container shares one kernel keyring quota; the default (200 keys) is exhausted by the lab's containers and their runtimes fail with "disk quota exceeded". These are Incus' recommended values. Writing to `/etc` and kernel settings need root |
| AppArmor drop-in `/etc/apparmor.d/abstractions/nameservice.d/networkmanager-no-stub`, then `sudo systemctl restart incus` | only on hosts where NetworkManager manages `/etc/resolv.conf`: AppArmor stops Incus' DNS server from reading NetworkManager's resolver file, so containers get no DNS. `make check` prints the exact lines |

Everything else runs as the user: `make init`, `make up`, the tests and the lab's tools.

### 3. Build

```
make init          # Ansible venv, secrets, Go builds of the lab's services; ends with make check
make up            # creates and configures the instances, ~15-25 minutes
make test          # end-to-end checks, ~10 minutes
```

- Run long steps in the background with their full output in a log file (`make up > /tmp/ai-lab-up.log 2>&1`), and tell the user the path and `tail -f /tmp/ai-lab-up.log` so they can follow it.
- `scheduler: slurm` (default) or `scheduler: k3s` in `inventory/group_vars/all.yml` selects Slurm or Kubernetes; ask the user which one if they did not say, before `make up`.
- `make frameworks` installs PyTorch and Ray into the lab (several GB, needed for the DDP and Ray examples); offer it, do not run it unasked.
- If a step fails, read the log, check [Platform: Troubleshooting](docs/platform.md#troubleshooting), fix the cause and rerun the step: every target is idempotent.

### 4. Hand over

Tell the user what they have and how to use it, from the README's [Quickstart](README.md#quickstart) and [Lab endpoints](README.md#lab-endpoints): `bin/ssh login` (the login node as user `joe`), the example jobs (`bin/scp -r examples login:`), Grafana at `http://<bridge address>.10:3000` with the password in `.secrets/grafana.pass`, `make test` to re-check.

### Rules

- `make down` deletes the instances (data volumes are kept) and `make purge` deletes everything, including volumes: run them only when the user asks.
- Do not commit or print the contents of `.secrets/`.
- The lab's addresses come from the Incus bridge; the docs show `10.107.111.x`, the user's bridge may differ (`incus network get incusbr0 ipv4.address`).

## Working on the repository

- Layout and components: [README](README.md#repository-layout), [Architecture](docs/architecture.md).
- Tests: `make test-fakegpu` (fake GPU libraries, on the host, no lab), `make test-bmc` (BMC integration tests against the running lab; `-disruptive` power-cycles a tray), `make test` (end to end). See [Testing](docs/testing.md).
- Changes to roles or playbooks apply to a running lab with `make configure`; it keeps data and monitoring history, unlike `make down` + `make up`.
- Markdown: never hard-wrap prose; one line per paragraph or list item.
