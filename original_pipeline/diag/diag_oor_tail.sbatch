#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --job-name=diag_oor_tail

set -u
source "$HOME/mugi_profiling/original_pipeline/env.sh"

OUT="$WORK/diag"
mkdir -p "$OUT"

python -u "$ORIG/diag/diag_oor.py" --repo "$HOME/mugi_profiling" --self_check || exit 1

run_unit diag oor_tail python -u "$ORIG/diag/diag_oor.py" \
    --repo "$HOME/mugi_profiling" \
    --n_samples 16 --exp_dims 10 \
    --modes as_is,exact_sm_above,shift_anchor3,edge_sm_above,const9_sm_above,const10_sm_above \
    --out "$OUT/oor_tail.csv" --stats_out "$OUT/oor_tail_stats.csv"
