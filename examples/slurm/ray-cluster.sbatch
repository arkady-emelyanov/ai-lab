#!/bin/bash
# A Ray cluster on a Slurm allocation: head on the first node, one worker per
# other node, then a driver script. Ray's temp files and object spilling go to
# node-local scratch. Needs the shared venv (make frameworks).
#SBATCH --job-name=ray-cluster
#SBATCH --output=%x-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --tmp=10G

source /shared/venv/bin/activate
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head=${nodes[0]}
port=6379
ray_opts=(--num-gpus=4 --num-cpus="$SLURM_CPUS_PER_TASK" --object-store-memory=1000000000 --block)

# --temp-dir is honoured on the head only; workers use the head's choice for
# sessions but spill to their own node-local scratch.
srun --nodes=1 --ntasks=1 -w "$head" \
    ray start --head --port=$port --temp-dir="$SCRATCH/ray-$SLURM_JOB_ID" --include-dashboard=false "${ray_opts[@]}" &
sleep 15
for node in "${nodes[@]:1}"; do
    srun --nodes=1 --ntasks=1 -w "$node" ray start --address="$head:$port" "${ray_opts[@]}" &
done
sleep 15

python "$SLURM_SUBMIT_DIR/../ray_demo.py"
rc=$?
ray stop --force >/dev/null 2>&1
exit $rc
