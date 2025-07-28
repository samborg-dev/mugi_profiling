#!/bin/bash


# Configuration files to process
configs=("llama_2_config.yaml" "llama_3_config.yaml" "small_models_config.yaml")

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
