#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=diag_gpu
#SBATCH --output=diag_gpu_%j.log
#SBATCH --error=diag_gpu_%j.err
#SBATCH --time=00:10:00

set -euo pipefail

module load cuda
module load cmake

cd /projects/SaltSystemsLab/hunter/jasperpy

set +e
echo "=== hostname ==="; hostname
echo ""
echo "=== all partitions (name, nodes, state, gres) ==="
sinfo -o "%P %.6D %.12T %G %N" | head -60
echo ""
echo "=== GPU partitions only ==="
sinfo -O "Partition:20,Gres:28,NodeList:24,StateLong" | grep -iE "gpu|h200|h100|a100" | head -40
echo ""
echo "=== idle GPU nodes right now ==="
sinfo -t idle -O "Partition:20,NodeList:24,Gres:28" | grep -iE "gpu|h200|h100|a100" | head
echo ""
echo "=== our allocated GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv
echo ""
echo "=== compute apps sharing this GPU (co-tenants?) ==="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
echo ""
echo "=== MIG / mode ==="
nvidia-smi -q | grep -iE "MIG Mode|Compute Mode|Product Name" | head
echo "=== done ==="
