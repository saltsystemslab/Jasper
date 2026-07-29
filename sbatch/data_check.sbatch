#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h200
#SBATCH --job-name=data_check
#SBATCH --output=data_check_%j.log
#SBATCH --error=data_check_%j.err
#SBATCH --time=00:10:00

set -uo pipefail

DATA=/projects/SaltSystemsLab/ann_data
GT=/projects/SaltSystemsLab/ann_gt
BASE="$DATA/bigann/bigann10M"
Q="$DATA/bigann/bigann10kquery"
GTF="$GT/GT_10M/bigann-10M"

echo "=== file stats (size + mtime) ==="
for f in "$BASE" "$Q" "$GTF"; do
  stat -c '%n  size=%s  mtime=%y' "$f" 2>&1 || echo "MISSING: $f"
done

echo ""
echo "=== headers (first two little-endian uint32) ==="
printf 'base   (n, dim)          : '; od -An -N8 -tu4 "$BASE" 2>&1
printf 'query  (n, dim)          : '; od -An -N8 -tu4 "$Q"    2>&1
printf 'gt     (n_queries, gt_k) : '; od -An -N8 -tu4 "$GTF"  2>&1

echo ""
echo "=== checksums ==="
md5sum "$GTF" "$Q" "$BASE" 2>&1

echo ""
echo "=== sibling GT files in $GT (any alternates / recent mtimes?) ==="
ls -la "$GT" 2>&1 | head -40
ls -la "$GT/GT_10M" 2>&1 | head -40

echo "=== done ==="
