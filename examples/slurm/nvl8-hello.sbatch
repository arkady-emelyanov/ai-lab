#!/bin/bash
# One task per GPU across the whole NVL8 domain: 2 trays x 4 GPUs. Each task
# reports which GPU Slurm bound it to and the NVLink domain that GPU is in.
#SBATCH --job-name=nvl8-hello
#SBATCH --output=%x-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --tmp=10G

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: $SLURM_NTASKS tasks"

srun -l bash -c '
    gpu=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,fabric.clique_id --format=csv,noheader)
    echo "rank $SLURM_PROCID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: $gpu, scratch $(df -h --output=avail "$SCRATCH" | tail -1)"
' | sort -n
