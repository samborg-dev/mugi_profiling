#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --job-name=diag_fixed

set -u
source "$HOME/mugi_profiling/original_pipeline/env.sh"

OUT="$WORK/diag"
mkdir -p "$OUT"

run_unit diag fixed python -u "$ORIG/diag/diag_fixed.py" \
    --repo "$HOME/mugi_profiling" \
    --ae_kernel "$HOME/mugi_ae/custom_nonlinear/custom_nonlinear_functions/vlp_softmax_approx.py" \
    --model_config "$HOME/mugi_profiling/config/model_config/llama/llama_2_7b.yaml" \
    --model_name NousResearch/Llama-2-7b-hf \
    --batch_size 2 --out "$OUT/fixed.csv" --stats_out "$OUT/fixed_stats.csv"
