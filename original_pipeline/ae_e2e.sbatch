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

python "$ORIG/ae_merge.py" || exit 1

run_unit ae e2e python -u model_script.py \
    --model_config config/model_config/llama/llama_2_7b.yaml \
    --nonlinear_config config/nonlinear_config/nonlinear_e2e_config.yaml \
    --parameter_config config/parameter_config/end_to_end_config.yaml
