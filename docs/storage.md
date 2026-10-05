[← README](../README.md) · **Storage**

# Storage

## Overview

| Path / endpoint | Backend | Scope | Use |
|---|---|---|---|
| `/shared` | Incus volume `cluster-shared` | all cluster nodes | homes (`/shared/home/<user>`), frameworks venv |
| `/pfs` | JuiceFS: data in RustFS bucket `pfs`, metadata in Redis | all cluster nodes | shared datasets and results, per-user `/pfs/<user>` |
| `/scratch` (`$SCRATCH`) | btrfs volume per tray, 50 GiB quota | each tray | node-local data, caches, spill files; JuiceFS read cache |
| `http://sched-storage:9000` | RustFS (S3) | cluster network and host | object storage with per-user buckets |

`sched-storage` (10.107.111.12) runs RustFS (S3 API on `:9000`, web console on `:9001`, data on the Incus volume `rustfs-data`) and Redis (JuiceFS metadata, password-protected, append-only, data on the Incus volume `juicefs-meta`). Both volumes survive `make down`, so `/pfs` is kept across rebuilds; if the metadata is lost while the bucket still holds data, `make up` restores it from the newest metadata backup JuiceFS keeps in the bucket (`pfs/meta/dump-*.json.gz`) instead of formatting. JuiceFS is mounted through FUSE (Incus provides `/dev/fuse`) on every cluster node. In k3s mode pods mount `/shared`, `/pfs` and `/scratch` from their tray (`hostPath`) and reach S3 by name (`sched-storage` resolves in pods).

## Usage

**Filesystems:**

```
df -h /shared /pfs /scratch                   # on a tray
srun -N2 --tmp=40G ./job.sh                    # request node-local scratch; $SCRATCH is set in the job
cp results.tar /pfs/$USER/                     # visible on every node
```

**Object storage per user:** every cluster user has an S3 identity (access key = user name, secret in `.secrets/users/<name>.s3`) allowed only on a bucket with their name. Credentials are ready in the home:

| File | Read by |
|---|---|
| `~/.aws/credentials`, `~/.aws/config` (endpoint, path-style addressing) | AWS CLI, boto3, most S3 libraries and data loaders |
| `~/.config/rc/config.toml` (alias `s3`) | RustFS client `rc` (installed on all cluster nodes) |

```
rc cp /pfs/joe/results.tar s3/joe/runs/       # from a login shell or a job
rc ls s3/joe/runs/
python -c "import boto3; print(boto3.client('s3').list_objects_v2(Bucket='joe').get('KeyCount'))"
```

Admin access (all buckets): keys in `.secrets/rustfs.access` and `.secrets/rustfs.secret`, e.g. the console at `http://10.107.111.12:9001/rustfs/console/`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `scratch_size_gib`, `scratch_pool_size` | 50, 200GiB | per-tray scratch quota, btrfs pool (sparse loop file) |
| `rustfs_version`, `rustfs_port`, `rustfs_console_port` | 1.0.1, 9000, 9001 | RustFS |
| `juicefs_version`, `juicefs_mount`, `juicefs_cache_mib` | 1.4.1, `/pfs`, 10240 | JuiceFS client |
| `rc_version` | 0.1.36 | RustFS client |
| `rustfs_volume`, `juicefs_meta_volume` | `rustfs-data`, `juicefs-meta` | Incus volumes for the object data and the JuiceFS metadata |

Downloads are verified against the projects' published checksums.

## Verification

```
bin/ssh root@sched-worker1 'df -h --output=target,size /pfs /scratch /shared'
bin/ssh login 'echo hi > /pfs/joe/t && srun -N1 -w sched-worker2 cat /pfs/joe/t'      # written on login, read on a tray (Slurm)
bin/ssh login 'rc ls s3/joe/; rc ls s3/pfs/'                                          # own bucket lists; the JuiceFS bucket is denied
```

`make test` checks that `/pfs` is mounted on every node with a file written on the login node readable everywhere, that RustFS is healthy, and that a job (Slurm) or a pod (k3s) can write and list the user's bucket but not another bucket.

## Limitations

- Everything is backed by the host's disk (Incus storage under `/var/lib/incus`): bandwidth and capacity are the host's, not those of a parallel filesystem.
- JuiceFS is not a parallel filesystem in the Lustre/GPFS sense; it gives the same shared POSIX namespace with object-store data.
- `df` inside the trays shows the whole scratch pool; the per-volume quota is enforced by btrfs.

## References

- [JuiceFS documentation](https://juicefs.com/docs/community/introduction/)
- [RustFS](https://github.com/rustfs/rustfs), [rustfs/cli (`rc`)](https://github.com/rustfs/cli)
- [Incus storage volumes](https://linuxcontainers.org/incus/docs/main/howto/storage_volumes/)
- Roles: `roles/storage`, `roles/s3users`, `roles/s3client`, `roles/juicefs`
