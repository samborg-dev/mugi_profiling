#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:2
#SBATCH --mem=64g
#SBATCH --job-name=lut_window_search
#SBATCH --error=window_search_error.txt
#SBATCH --output=window_search_output.txt

# MODE=noise  (default) measures the perplexity noise floor and the model load cost.
# MODE=search runs the automated per-layer window search and needs far more wall time
#             than the 1:00:00 directive above -- override it at submit time:
#
#   MODE=noise  sbatch --export=ALL slurm_scripts/window_search.sh
#   MODE=search NOISE_SUMMARY=output/window_search/cost_and_noise_summary.yaml \
#     sbatch --export=ALL --time=8:00:00 slurm_scripts/window_search.sh

set -u

cd ~/mugi_profiling

mkdir -p output/window_search

unset PYTHONPATH

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q "^mugi_profiling "; then
    conda activate mugi_profiling
else
    echo "conda env 'mugi_profiling' not found -- create it with:"
    echo "  git show upstream/asplos_2026_ae:environment.yaml > /tmp/environment.yaml"
    echo "  conda env create -f /tmp/environment.yaml"
    exit 1
fi

python -c "import torch, transformers" || { echo "torch/transformers unavailable"; exit 1; }

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set. Export it before sbatch:"
    echo "  export HF_TOKEN=hf_xxx   (then: sbatch --export=ALL slurm_scripts/window_search.sh)"
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

MODE="${MODE:-noise}"
REPEATS="${REPEATS:-5}"
N_SAMPLES="${N_SAMPLES:-8}"

COMMON=(
    --model_config config/model_config/llama/llama_2_7b.yaml
    --nonlinear_config config/nonlinear_config/nonlinear_config.yaml
    --parameter_config config/parameter_config/parameter_config.yaml
    --n_samples "$N_SAMPLES"
    --out output/window_search/cost_and_noise.csv
)

echo "python: $(which python)"
echo "mode=$MODE repeats=$REPEATS n_samples=$N_SAMPLES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$MODE" = "search" ]; then
    SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
    PATIENCE="${PATIENCE:-3}"

    ARGS=(--mode search --search_repeats "$SEARCH_REPEATS" --patience "$PATIENCE")

    if [ -n "${NOISE_FLOOR:-}" ]; then
        ARGS+=(--noise_floor "$NOISE_FLOOR")
    elif [ -n "${NOISE_SUMMARY:-}" ]; then
        ARGS+=(--noise_summary "$NOISE_SUMMARY")
    else
        echo "MODE=search needs a measured noise floor. Run MODE=noise first, then"
        echo "  NOISE_SUMMARY=output/window_search/cost_and_noise_summary.yaml"
        echo "or set NOISE_FLOOR=<float> explicitly."
        exit 1
    fi

    [ -n "${PROFILE_ROOT:-}" ] && ARGS+=(--profile_root "$PROFILE_ROOT")
    [ -n "${NO_SEED:-}" ] && ARGS+=(--no_seed)
    [ -n "${MAX_EVALS:-}" ] && ARGS+=(--max_evals "$MAX_EVALS")
    [ -n "${MAX_EVALS_PER_LAYER:-}" ] && ARGS+=(--max_evals_per_layer "$MAX_EVALS_PER_LAYER")
    [ -n "${RADIUS:-}" ] && ARGS+=(--radius "$RADIUS")
    [ -n "${TIME_BUDGET_S:-}" ] && ARGS+=(--time_budget_s "$TIME_BUDGET_S")
    [ -n "${LAYERS:-}" ] && ARGS+=(--layers "$LAYERS")

    python run_window_search.py "${COMMON[@]}" "${ARGS[@]}"
else
    python run_window_search.py "${COMMON[@]}" --mode noise --repeats "$REPEATS"
fi

echo "exit code: $?"
