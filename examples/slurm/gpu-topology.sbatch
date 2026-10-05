#!/bin/bash
# Prints the emulated GPU topology of every tray in the NVL8 domain.
#SBATCH --job-name=gpu-topology
#SBATCH --output=%x-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

echo "== Slurm view"
scontrol show topology
sinfo -N -o "%N %G %c %m %d"

# One task per tray; -l prefixes every line with the task number.
srun -l bash -c '
    echo "== $(hostname): GPUs"
    nvidia-smi -L
    echo "== $(hostname): NVLink matrix"
    nvidia-smi topo -m | head -5
    echo "== $(hostname): NVLink fabric (must match on all trays)"
    nvidia-smi --query-gpu=index,pci.bus_id,fabric.state,fabric.cluster_uuid,fabric.clique_id --format=csv,noheader
'
