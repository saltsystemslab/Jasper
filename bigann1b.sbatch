#!/bin/bash
#SBATCH --job-name=create_scale_index
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodelist=d4067
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --mem=950G
#SBATCH --time=24:00:00
#SBATCH --output=logs/create_scale_index_%j.out
#SBATCH --error=logs/create_scale_index_%j.err

# Make sure the log directory exists
mkdir -p logs

# Print job info for debugging
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Run the indexing command
/home/wang.zikun/jasper/build/bin/create_scale_index \
    -f /tmp2/ann_data/bigann1B \
    -t uint8 \
    -d l2 \
    -n 64 \
    --n_parts 16 \
    -w 15GB \
    -i /tmp2/ann_data/bigann1B-index

echo "End time: $(date)"