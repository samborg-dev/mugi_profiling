#!/bin/bash

#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH --constraint=h100
#SBATCH --job-name=trasformer_profiling_test
#SBATCH --error=transformer_profiling_test.txt
#SBATCH --output=transformer_profiling_test.txt

module load python
module load anaconda
module load cuda

conda deactivate
conda activate mugi_profiling

cd ~/mugi_profiling

# Configuration files to process
configs=("small_models_config.yaml")

# Loop through each configuration
for config in "${configs[@]}"; do
    echo ""
    echo "Running experiment with configuration: $config"
    echo "----------------------------------------"
    
    # Check if config file exists
    if [ ! -f "$config" ]; then
        echo "Warning: Configuration file '$config' not found. Skipping..."
        continue
    fi
    
    # Run the transformer script with the current config
    python transfomer_script.py --config "$config"
    
    # Check if the script ran successfully
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed experiment with $config"
    else
        echo "✗ Error occurred while running experiment with $config"
        echo "Continuing with next configuration..."
    fi
    
    echo "----------------------------------------"

    rm -rf ~/.cache/huggingface
done
