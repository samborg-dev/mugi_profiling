#!/bin/bash


# Configuration files to process
# model_configs=("config/model_config/llama/llama_2_7b.yaml")
model_configs=("config/model_config/swin/swinv2_tiny.yaml")
# model_configs=("config/model_config/whisper/whisper_tiny.yaml")
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
                                --parameter_config "$parameter_config" #\
                                #--hf_token "$hf_token"
    
    # Check if the script ran successfully
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed experiment with $model_config"
    else
        echo "✗ Error occurred while running experiment with $model_config"
        echo "Continuing with next configuration..."
    fi
    
    echo "----------------------------------------"

    # rm -rf ~/.cache/huggingface
done
