#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:2
#SBATCH --mem=64g

set -u
source "$HOME/mugi_profiling/original_pipeline/env.sh"
cd "$HOME/mugi_ae"

echo "chunk $CHUNK"
run_unit ae "$CHUNK" python -u model_script.py \
    --model_config config/model_config/llama/llama_2_7b.yaml \
    --nonlinear_config "config/chunks/nonlinear_config_${CHUNK}.yaml" \
    --parameter_config config/parameter_config/parameter_config.yaml
