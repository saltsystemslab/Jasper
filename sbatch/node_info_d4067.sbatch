#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=ninfo67
#SBATCH --output=ninfo67_%j.log
#SBATCH --error=ninfo67_%j.err
#SBATCH --time=00:05:00

set -euo pipefail

module load cuda
module load cmake

cd /projects/SaltSystemsLab/hunter/jasperpy

set +e
echo "=== d4067 ==="
scontrol show node d4067 2>&1 | tr ' ' '\n' | grep -iE "^NodeName=|^Partitions=|^State=|^CPUTot=|^Gres=|^CfgTRES=|^AllocTRES=|^RealMemory=" | paste -sd' ' -
echo "=== partitions that include d4067 ==="
sinfo -N -n d4067 -o "%P %.10T %G %m %c" 2>&1
echo "=== done ==="
