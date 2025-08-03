#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=64
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
nonlinear_config="config/nonlinear_config/nonlinear_test.yaml"
parameter_config="config/parameter_config/parameter_config.yaml"
hf_token="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"

huggingface-cli login --token "$hf_token"

# Create results file with timestamp
results_file="experiment_results.txt"
echo "Experiment Results" > "$results_file"
echo "=======================================" >> "$results_file"
echo "" >> "$results_file"

dir="config/model_config/"

# Loop through each configuration
for sub_dir in "$dir"/*; do
    for model_config in "$sub_dir"/*; do
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
                                    --parameter_config "$parameter_config" #\
                                    #--hf_token "$hf_token"
        
        # Check if the script ran successfully
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed experiment with $model_config"
            echo "PASSED: $model_config" >> "$results_file"
        else
            echo "✗ Error occurred while running experiment with $model_config"
            echo "FAILED: $model_config" >> "$results_file"
            echo "Continuing with next configuration..."
        fi
        
        echo "----------------------------------------"

        # rm -rf ~/.cache/huggingface
    done
done

# Add summary to results file
echo "" >> "$results_file"
echo "=======================================" >> "$results_file"
echo "Experiment completed at $(date)" >> "$results_file"
echo "Results saved to: $results_file"
