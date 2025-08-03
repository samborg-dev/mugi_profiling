#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --job-name=whisper_profiling
#SBATCH --error=whisper_error.txt
#SBATCH --output=whisper_profiling.txt

module load python
module load anaconda
module load cuda

conda deactivate
conda activate mugi_profiling

cd ~/mugi_profiling

# Configuration files to process
# model_configs=("config/model_config/llama/llama_2_7b.yaml")
# model_configs=("config/model_config/llama/llama_3_8b.yaml")
# model_configs=("config/model_config/swin/swinv2_tiny.yaml")
model_configs=("config/model_config/whisper/whisper_tiny.yaml"
               "config/model_config/whisper/whisper_small.yaml"
               "config/model_config/whisper/whisper_base.yaml"
               "config/model_config/whisper/whisper_medium.yaml"
               "config/model_config/whisper/whisper_large.yaml")
# model_configs=("config/model_config/vivit/vivit-b-16x2.yaml")
nonlinear_config="config/nonlinear_config/nonlinear_test.yaml"
parameter_config="config/parameter_config/parameter_config.yaml"
hf_token="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"

huggingface-cli login --token "$hf_token"

# Loop through each configuration
for model_config in "${model_configs[@]}"; do
    echo ""
    echo "Running experiment with configuration: $model_config"
    echo "----------------------------------------"
    
    # Check if config file exists
    if [ ! -f "$model_config" ]; then
        echo "Warning: Configuration file '$model_config' not found. Skipping..."
        continue
    fi
    
    # Run the transformer script with the current config
    python model_script.py --model_config "$model_config" \
                                --nonlinear_config "$nonlinear_config" \
                                --parameter_config "$parameter_config"
    
    # Capture the exit code
    exit_code=$?
    
    # Check if the script ran successfully
    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed experiment with $model_config"
    else
        echo "✗ Error occurred while running experiment with $model_config (exit code: $exit_code)"
        echo "Check whisper_detailed_log.txt and whisper_error.txt for details"
        echo "Continuing with next configuration..."
    fi
    
    echo "----------------------------------------"

    # rm -rf ~/.cache/huggingface
done
