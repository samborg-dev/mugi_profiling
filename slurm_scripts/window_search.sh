#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:2
#SBATCH --mem=64g
#SBATCH --job-name=lut_window_cost_noise
#SBATCH --error=output/window_search/error.txt
#SBATCH --output=output/window_search/output.txt

set -u

mkdir -p output/window_search

source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate mugi_profiling

cd ~/mugi_profiling

unset PYTHONPATH

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set. Export it before sbatch:"
    echo "  export HF_TOKEN=hf_xxx   (then: sbatch --export=ALL slurm_scripts/window_search.sh)"
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

REPEATS="${REPEATS:-5}"
N_SAMPLES="${N_SAMPLES:-8}"

echo "python: $(which python)"
echo "repeats=$REPEATS n_samples=$N_SAMPLES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python run_window_search.py \
    --model_config config/model_config/llama/llama_2_7b.yaml \
    --nonlinear_config config/nonlinear_config/nonlinear_config.yaml \
    --parameter_config config/parameter_config/parameter_config.yaml \
    --repeats "$REPEATS" \
    --n_samples "$N_SAMPLES" \
    --out output/window_search/cost_and_noise.csv

echo "exit code: $?"
