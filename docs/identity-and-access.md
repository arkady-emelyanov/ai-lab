[← README](../README.md) · **Identity and access**

# Identity and access

## Overview

Users exist once, in OpenLDAP on `sched-control`, and every cluster node resolves them through SSSD: no node has local user accounts. Homes are on the shared volume (`/shared/home/<user>`). Users log in to the login node with their lab SSH key (one per user, generated into `.secrets/ssh/users/` and used by `bin/ssh`), their own key or their LDAP password; root logs in with the lab's admin key only. The same user list also creates each user's S3 identity and, depending on the scheduler, their Slurm association or their Kubernetes namespace, queue and credentials ([Kubernetes](kubernetes.md#how-it-works)).

| Piece | Where | Notes |
|---|---|---|
| OpenLDAP (`slapd`) | `sched-control` | base `dc=example,dc=com`; `ou=people`, `ou=groups`; user private groups (GID = UID) |
| SSSD | every cluster node | `id_provider`, `auth_provider` and `chpass_provider` = ldap |
| sshd | every instance | root: key only; users: password or key |
| `bin/ssh`, `bin/scp`, `bin/ssh-copy-id` | host | wrappers that resolve current addresses from Incus |

## Usage

**Connect from your machine:**

| Command | Lands on | As |
|---|---|---|
| `bin/ssh login` | login node | `joe`, with joe's lab key (`.secrets/ssh/users/joe_ed25519`); `bin/ssh <user>@login` for another directory user |
| `bin/ssh root@login` | login node | root (admin key `.secrets/ssh/id_ed25519`) |
| `bin/ssh sched-control` (or any instance name) | that instance | root |
| `bin/ssh login sbatch < job.sh` | login node | runs a command, here submitting a job from stdin |
| `bin/scp file login:` | login node | copies files |
| `make shell`, `make shell-root`, `make shell-<instance>` | via `incus exec`, no SSH | joe / root / root |

The aliases exist only inside the wrappers; nothing is added to `~/.ssh/config`. `make configure` creates a lab key per directory user and authorises it in their home on `/shared`, so it works on every node; `bin/ssh` picks it by user name. Plain `ssh` from elsewhere uses your own key (`bin/ssh-copy-id -i ~/.ssh/id_ed25519.pub login`) or the LDAP password.

**Add a user:** append an entry to `cluster_users` (copy the list from `inventory/group_vars/all.yml` to `local.yml`) and run `make configure`:

```yaml
cluster_users:
  - name: joe
    uid: 2001
    comment: Joe <joe@example.com>
    account: lab        # Slurm account (must exist in slurm_accounts; Slurm mode)
    password: joe       # omit to generate one into .secrets/users/<name>.pass
```

This creates the LDAP entry and private group, the home with skeleton files, the Slurm association (k3s mode: a namespace with a Kueue queue and `~/.kube/config`), the S3 user and bucket with credentials in the home. Passwords are re-applied on every `make configure`, so a change made with `passwd` is reverted.

## Verification

```
bin/ssh root@sched-worker1 getent passwd joe     # resolved via SSSD: joe:*:2001:2001:Joe <joe@example.com>:...
bin/ssh root@sched-worker1 grep -c '^joe:' /etc/passwd    # 0: no local account
bin/ssh login id                                  # uid=2001(joe) gid=2001(joe) groups=2001(joe)
bin/ssh root@sched-control sacctmgr show assoc user=joe format=user,account   # Slurm mode
bin/ssh login kubectl auth whoami                 # k3s mode: joe, groups lab-users
```

## Limitations

- LDAP runs without TLS (`ldap_auth_disable_tls_never_use_in_production = true` in SSSD), so passwords cross the Incus bridge in clear.
- In k3s mode users authenticate to Kubernetes with a client certificate (10 years, in `~/.kube/config`), not their LDAP password; a certificate cannot be revoked short of rotating the cluster's client CA. Pods run as the user's uid only because `examples/k8s/submit` sets it; nothing enforces it.
- Users authenticate to the login node; jobs on the trays run as the user via Slurm (or as the uid their pods ask for). Users cannot SSH from the login node to the trays unless they use agent forwarding with their own key.

## References

- [SSSD LDAP provider](https://sssd.io/docs/users/ldap_with_sssd.html)
- [OpenLDAP admin guide](https://www.openldap.org/doc/admin26/)
- Roles: `roles/ldap`, `roles/sssd`, `roles/ssh`, `roles/login`
