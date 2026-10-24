#!/bin/bash
# The same job step in each of the lab's container runtimes, with the job's
# 2 GPUs: Pyxis and enroot, Apptainer, and Slurm's own OCI support (an OCI
# bundle made with skopeo and umoci). Each container sees the job's GPUs and
# only those. podman with scrun runs from the login node instead:
#   SCRUN_GPUS_PER_NODE=2 podman run --rm mirror.gcr.io/library/ubuntu:24.04 nvidia-smi -L
#SBATCH --job-name=containers
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

# Docker Hub images through Google's public cache.
image=mirror.gcr.io/library/ubuntu:24.04

echo "== Pyxis and enroot (srun --container-image)"
srun --container-image="${image/\//#}" sh -c 'grep PRETTY_NAME /etc/os-release; nvidia-smi -L'

echo "== Apptainer (apptainer exec --nv)"
srun apptainer exec --nv "docker://$image" sh -c 'grep PRETTY_NAME /etc/os-release; nvidia-smi -L'

# srun resolves the command on the submitting node: run it through the
# container's shell.
echo "== Slurm OCI (srun --container)"
bundle=$HOME/.cache/containers-example/$(echo "$image" | tr /: __)/bundle
if [ ! -d "$bundle" ]; then
    mkdir -p "$(dirname "$bundle")"
    env -u XDG_RUNTIME_DIR skopeo copy -q "docker://$image" "oci:$(dirname "$bundle")/image:latest"
    umoci unpack --rootless --image "$(dirname "$bundle")/image:latest" "$bundle" >/dev/null
fi
srun --container="$bundle" sh -c 'grep PRETTY_NAME /etc/os-release; nvidia-smi -L'
