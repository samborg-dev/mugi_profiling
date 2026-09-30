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
cd "$HOME/mugi_newton"
export PYTHONPATH="$HOME/mugi_newton:$PYTHONPATH"

LIMIT_S="${LIMIT_S:-1800}"
SAFETY_S="${SAFETY_S:-120}"
job_start=$(date +%s)
last=0

while true; do
    layer=$(python "$ORIG/newton_next.py" next) || exit 1
    if [ "$layer" = "done" ]; then
        python "$ORIG/newton_next.py" summary
        exit 0
    fi
    used=$(( $(date +%s) - job_start ))
    if [ "$last" -gt 0 ] && [ $(( used + last * 13 / 10 + SAFETY_S )) -gt "$LIMIT_S" ]; then
        echo "stopping before layer $layer: ${used}s used, last layer took ${last}s"
        exit 0
    fi
    t0=$(date +%s)
    echo "layer $layer"
    run_unit newton "L$(printf %02d "$layer")" python -u src/model_script.py \
        --model_config config/model_config/llama/llama_2_7b.yaml \
        --nonlinear_config config/nonlinear_config/vlp/vlp_softmax_layers_llama_2_7b.yaml \
        --parameter_config config/parameter_config/parameter_config.yaml || exit 1
    last=$(( $(date +%s) - t0 ))
    echo "layer $layer took ${last}s"
done
