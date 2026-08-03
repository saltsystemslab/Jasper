#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=node_info
#SBATCH --output=node_info_%j.log
#SBATCH --error=node_info_%j.err
#SBATCH --time=00:05:00

set -euo pipefail

module load cuda
module load cmake

cd /projects/SaltSystemsLab/hunter/jasperpy

set +e
echo "host: $(hostname)  nproc(visible)=$(nproc)"
echo "=== this job's alloc ==="
scontrol show job "$SLURM_JOB_ID" | grep -iE "NumCPUs|NumNodes|CPUs/Task|mem=|TRES" | head
echo "=== h200 nodes cpu/mem/alloc ==="
for n in d4052 d4053 d4054 d4055 d4072; do
  scontrol show node "$n" 2>/dev/null | tr ' ' '\n' | grep -iE "^NodeName=|^CPUTot=|^CPUAlloc=|^RealMemory=|^CfgTRES=|^AllocTRES=" | paste -sd' ' -
done
echo "=== done ==="
