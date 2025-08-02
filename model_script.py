# TODO:
#
# test each step on all model types (llama2, llama3.1, whisper, swinv2, vivit)
# Run only 1 model at a time (e.g. llama 2 7b)
#
# 1. Load model config, nonlinear config, and inference parameters
# 2. Load model, dataset, tokenizer, processor, etc.
# 3. Batch dataset and preprocess
# 4. Loop through nonlinear configuration
# 5. Run inference
# 6. Profile with custom attention and ffn (by layer and seq len)
# 7. save ppl / loss
# 8. exit

import argparse
import yaml
import torch
import os
import pandas as pd

from utils import huggingface_login, validate_config
from inference_classes.audio_inference import AudioModel
from inference_classes.npl_inference import NLPModel
from inference_classes.video_inference import VideoModel
from inference_classes.vision_inference import VisionModel

token = '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'

def evaluate_model(model_dict, nonlinear_dict, parameter_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Validate configurations
    modality = validate_config(model_dict, nonlinear_dict, parameter_dict)

    inference_model = None
    try:
        # Initialize model class
        if modality == 'nlp':
            inference_model = NLPModel(model_dict, nonlinear_dict, parameter_dict, device)
        elif modality == 'audio':
            inference_model = AudioModel(model_dict, nonlinear_dict, parameter_dict, device)
        elif modality == 'vision':
            inference_model = VisionModel(model_dict, nonlinear_dict, parameter_dict, device)
        elif modality == 'video':
            inference_model = VideoModel(model_dict, nonlinear_dict, parameter_dict, device)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        inference_model.csv_file = f'csv/{inference_model.model_name}/metric.csv'
        if os.path.exists(inference_model.csv_file):
            os.remove(inference_model.csv_file)
        else:
            os.makedirs(os.path.dirname(inference_model.csv_file), exist_ok=True)

        inference_model.df = None

        print('Loading model')
        inference_model.load_model()
        print('Loading dataset')
        inference_model.load_streaming_dataset()
        print('Processing dataset')
        inference_model.process_dataset()
        print('Batching dataset')
        inference_model.batch_dataset()
        inference_model.set_profiling_dims()
        print('Looping through configurations')
        inference_model.loop_configuration()
        print('Saving results')

        inference_model.df.to_csv(inference_model.csv_file, index=False)

    finally:
        if inference_model is not None:
            inference_model.cleanup()
            del inference_model

def main():
    parser = argparse.ArgumentParser(description="Run profiling on transformer model with custom nonlinear functions.")
    parser.add_argument('--model_config', type=str, default=None, 
                        help='Path to model config YAML file (default: None)')
    parser.add_argument('--nonlinear_config', type=str, default=None,
                        help='Path to nonlinear config YAML file (default: None)')
    parser.add_argument('--parameter_config', type=str, default=None,
                        help='Path to inference parameters YAML file (default: None)')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face token for authentication (default: None, assumes hf is already logged in)')
    args = parser.parse_args()

    # if args.hf_token:
    #     huggingface_login(args.hf_token)

    model_config = yaml.safe_load(open(args.model_config))
    nonlinear_config = yaml.safe_load(open(args.nonlinear_config))
    parameter_config = yaml.safe_load(open(args.parameter_config))

    evaluate_model(model_config, nonlinear_config, parameter_config)


if __name__ == '__main__':
    main()