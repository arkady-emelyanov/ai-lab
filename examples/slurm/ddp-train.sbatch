#!/bin/bash
# PyTorch DDP across the whole NVL8 domain: 2 trays x 4 GPUs, one torchrun
# per node launching 4 ranks. Needs the shared venv (make frameworks).
#SBATCH --job-name=ddp-train
#SBATCH --output=%x-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4

source /shared/venv/bin/activate
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

srun torchrun \
    --nnodes="$SLURM_NNODES" --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$head:29500" --rdzv-id="$SLURM_JOB_ID" \
    "$SLURM_SUBMIT_DIR/../ddp_train.py" "$@"
