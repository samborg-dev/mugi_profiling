#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:2
#SBATCH --mem=64g
#SBATCH --job-name=diag_start_ppl

set -u
source "$HOME/mugi_profiling/original_pipeline/env.sh"

DIAG="$ORIG/diag"
OUT="$WORK/diag"
mkdir -p "$OUT"
MODEL_CONFIG="$HOME/mugi_profiling/config/model_config/llama/llama_2_7b.yaml"

run_unit diag ae python -u "$DIAG/diag_eval.py" \
    --code ae --repo "$HOME/mugi_ae" \
    --model_config "$MODEL_CONFIG" --model_name weights/llama-2-7b-hf \
    --batch_sizes 8,4,2 --out "$OUT/ae.csv"

run_unit diag main python -u "$DIAG/diag_eval.py" \
    --code main --repo "$HOME/mugi_profiling" \
    --model_config "$MODEL_CONFIG" --model_name NousResearch/Llama-2-7b-hf \
    --batch_sizes 2,4,8 --out "$OUT/main.csv"
