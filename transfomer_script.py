from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoModelForImageClassification, AutoModelForVideoClassification
from datasets import load_dataset
from datasets import interleave_datasets
import torch
import math
import yaml
import pandas as pd
import itertools
import os
import argparse
from tqdm import tqdm
from huggingface_hub import login
from utils import process_nlp_dataset, process_audio_dataset, process_image_dataset, process_video_dataset
from custom_approx import CustomSoftmax, CustomSilu, CustomGelu
from custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax
from custom_nonlinear_functions.vlp_silu_approx import VLPSilu
from custom_nonlinear_functions.vlp_gelu_approx import VLPGelu
from custom_nonlinear_functions.pwl_softmax_approx import PWLSoftmax
from custom_nonlinear_functions.pwl_silu_approx import PWLSilu
from custom_nonlinear_functions.pwl_gelu_approx import PWLGelu
from custom_nonlinear_functions.pwl_mobilenet_approx import PWLMobilenet
from custom_nonlinear_functions.taylor_softmax_approx import TaylorSoftmax
from nonlinear_approx import default_softmax, default_silu, default_gelu, set_vlp_softmax, set_vlp_silu, set_vlp_gelu, set_pwl_softmax, set_pwl_silu, set_pwl_gelu, set_pwl_mobilenet, set_taylor_softmax

def load_model(model_type, model_name, device, parameter_dict):
    if model_type == 'nlp':
        tok_proc = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok_proc.pad_token is None:
            tok_proc.pad_token = tok_proc.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation='eager').to(device)
        parameter_dict['max_length'] = model.config.max_position_embeddings

    elif model_type == 'audio':
        tok_proc = AutoProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation='eager').to(device)
        parameter_dict['target_sample_rate'] = tok_proc.feature_extractor.sampling_rate
        parameter_dict['max_length'] = model.config.max_source_positions

    elif model_type == 'vision':
        tok_proc = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation='eager').to(device)
        parameter_dict['max_length'] = model.config.image_size

    elif model_type == 'video':
        tok_proc = AutoProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForVideoClassification.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation='eager').to(device)
        parameter_dict['num_frames'] = model.config.num_frames
        parameter_dict['max_length'] = model.config.image_size

    model.eval()
    return tok_proc, model, parameter_dict

def load_config_dataset(dataset_name, config, split):
    from datasets import interleave_datasets
    
    if isinstance(split, list):
        if isinstance(config, list):
            datasets = []
            for cfg in config:
                for spl in split:
                    if cfg:
                        dataset = load_dataset(dataset_name, cfg, split=spl, streaming=True, trust_remote_code=True)
                    else:
                        dataset = load_dataset(dataset_name, split=spl, streaming=True, trust_remote_code=True)
                    datasets.append(dataset)
        else:
            datasets = []
            for spl in split:
                if config:
                    dataset = load_dataset(dataset_name, config, split=spl, streaming=True, trust_remote_code=True)
                else:
                    dataset = load_dataset(dataset_name, split=spl, streaming=True, trust_remote_code=True)
                datasets.append(dataset)
        
        combined_dataset = interleave_datasets(datasets)
        return combined_dataset
    
    elif isinstance(config, list):
        datasets = []
        for cfg in config:
            if cfg:
                dataset = load_dataset(dataset_name, cfg, split=split, streaming=True, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
            datasets.append(dataset)

        combined_dataset = interleave_datasets(datasets)
        return combined_dataset
    else:
        if config:
            dataset = load_dataset(dataset_name, config, split=split, streaming=True, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        return dataset

def process_dataset(model_type, tok_proc, dataset, n_samples, max_length=None, target_sample_rate=None, num_frames=None):
    if model_type == 'nlp':
        assert max_length is not None, "max_length must be specified for NLP models"
        processed_dataset = process_nlp_dataset(dataset, tok_proc, n_samples, max_length)

    elif model_type == 'audio':
        assert target_sample_rate is not None, "target_sample_rate must be specified for audio models"
        processed_dataset = process_audio_dataset(dataset, tok_proc, n_samples, target_sample_rate)

    elif model_type == 'vision':
        processed_dataset = process_image_dataset(dataset, tok_proc, n_samples)

    elif model_type == 'video':
        processed_dataset = process_video_dataset(dataset, tok_proc, n_samples, num_frames)

    return processed_dataset

def batch_dataset(processed_dataset, num_samples, batch_size, model_type=None, tokenizer=None):
    assert num_samples % batch_size == 0, "num_samples must be divisible by batch_size"

    batched_data = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Batching dataset"):
        batch = processed_dataset[i:i+batch_size]
        
        # For NLP models, pad sequences to same length within batch
        if model_type == 'nlp' and tokenizer is not None:
            batch_max_len = max(ex["input_ids"].shape[1] for ex in batch)
            
            padded_batch = []
            for ex in batch:
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [ex["input_ids"].squeeze(0)[:batch_max_len]],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id
                ).squeeze(0)
                
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                
                padded_batch.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })
            batch = padded_batch
        
        batched_data.append(batch)
    return batched_data

def compute_loss(model_type, model, tok_proc, batch):
    if model_type == 'nlp':
        input_ids = torch.stack([ex["input_ids"] for ex in batch]).to(model.device)
        attention_mask = torch.stack([ex["attention_mask"] for ex in batch]).to(model.device).bool()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        # Delete intermediate tensors
        del input_ids, attention_mask
        
    elif model_type == 'audio':
        input_features = torch.cat([ex["input_features"] for ex in batch], dim=0).to(model.device).to(torch.float16)
        labels = torch.cat([ex["labels"] for ex in batch], dim=0).to(model.device)
        labels[labels == tok_proc.tokenizer.pad_token_id] = -100
        with torch.no_grad():
            outputs = model(input_features=input_features, labels=labels, use_cache=False)
        # Delete intermediate tensors
        del input_features, labels
        
    elif model_type == 'vision':
        pixel_values = torch.stack([ex["pixel_values"].squeeze(0) for ex in batch]).to(model.device).to(torch.float16)
        labels = torch.stack([ex["labels"] for ex in batch]).squeeze(-1).to(model.device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, labels=labels)
        # Delete intermediate tensors
        del pixel_values, labels
        
    elif model_type == 'video':
        pixel_values = torch.stack([ex["pixel_values"].squeeze(0) for ex in batch]).to(model.device).to(torch.float16)
        labels = torch.stack([ex["labels"] for ex in batch]).squeeze(-1).to(model.device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, labels=labels)
        # Delete intermediate tensors
        del pixel_values, labels

    loss = outputs.loss
    del outputs  # Delete the full outputs object
    return loss

def run_inference(batched_data, model_type, model, tok_proc):
    total_loss = 0.0
    num_batches = 0
    for batch in tqdm(batched_data, desc="Model Inference"):
        batched_loss = compute_loss(
            model_type=model_type,
            model=model,
            tok_proc=tok_proc,
            batch=batch
        )
        total_loss += batched_loss.item()
        num_batches += 1
        del batch, batched_loss
        torch.cuda.empty_cache()

    if model_type in ['vision', 'video']:
        # For classification models, report average loss instead of perplexity
        avg_loss = compute_average_loss(total_loss, num_batches)
        result = avg_loss
        metric_name = "Average Loss"
    else:
        # For generative models (NLP, audio), compute perplexity
        perplexity = compute_perplexity(total_loss, num_batches)
        result = perplexity
        metric_name = "Perplexity"
    return result, metric_name

def compute_perplexity(loss, num_batches):
    assert num_batches > 0, "num_batches must be greater than 0"
    return math.exp(loss / num_batches)

def compute_average_loss(loss, num_batches):
    assert num_batches > 0, "num_batches must be greater than 0"
    return loss / num_batches

def flatten_dict_ranges(d):
    for key, value in d.items():
        if isinstance(value, list):
            if len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], int):
                d[key] = [i for i in range(value[0], value[1] + 1)]
    return d

def clean_lists(d):
    for key, value in d.items():
        if isinstance(value, list):
            if len(value) == 1:
                d[key] = value[0]
    return d

def convert_dict_to_list_dict(d):
    d = {
        k: v if isinstance(v, list) else [v]
        for k, v in d.items()
    }

    return d

def set_nonlinear_operations(attention_default, ffn_default, attention_operation, ffn_operation, approx_method, attention_config, ffn_config, nonlinear_config, model, model_type, torch_softmax, torch_silu, torch_gelu, attention_object=None, ffn_object=None):
    # Select attetion and feed-forward nonlinear operation class
    if approx_method == 'torch' or (attention_default and ffn_default):
        if attention_operation == 'softmax':
            set_attention = default_softmax
        if ffn_operation == 'silu':
            set_ffn = default_silu
        elif ffn_operation == 'gelu':
            set_ffn = default_gelu

    elif approx_method == 'vlp':
        if attention_operation == 'softmax' and not attention_default:
            set_attention = set_vlp_softmax
        else:
            set_attention = default_softmax

        if ffn_operation == 'silu' and not ffn_default:
            set_ffn = set_vlp_silu
        elif ffn_operation == 'gelu' and not ffn_default:
            set_ffn = set_vlp_gelu
        else:
            set_ffn = default_silu

    elif approx_method == 'pwl':
        if attention_operation == 'softmax' and not attention_default:
            set_attention = set_pwl_softmax
        else:
            set_attention = default_softmax

        if ffn_operation == 'silu' and not ffn_default:
            set_ffn = set_pwl_silu
        elif ffn_operation == 'gelu' and not ffn_default:
            set_ffn = set_pwl_gelu
        else:
            set_ffn = default_silu

    elif approx_method == 'pwl_mobilenet':
        set_attention = default_softmax

        if ffn_operation == 'silu' and not ffn_default:
            set_ffn = set_pwl_mobilenet
        else:
            set_ffn = default_silu

    elif approx_method == 'taylor':
        if attention_operation == 'softmax' and not attention_default:
            set_attention = set_taylor_softmax
        else:
            set_attention = default_softmax

        set_ffn = default_silu

    torch_attn = torch_softmax
    torch_ffn = torch_silu if ffn_operation == 'silu' else torch_gelu

    if not attention_object:
        attention_object = set_attention(config=attention_config, base_config=nonlinear_config, torch_nonlinear=torch_attn)
    else:
        attention_object = set_attention(config=attention_config, base_config=nonlinear_config, torch_nonlinear=torch_attn, nonlinear_object=attention_object)

    if not ffn_object:
        ffn_object = set_ffn(config=ffn_config, base_config=nonlinear_config, torch_nonlinear=torch_ffn)
    else:
        ffn_object = set_ffn(config=ffn_config, base_config=nonlinear_config, torch_nonlinear=torch_ffn, nonlinear_object=ffn_object)

    torch.nn.functional.softmax = attention_object

    if model_type == 'nlp':
        for layer in model.model.layers:
            layer.mlp.act_fn = ffn_object
    if model_type == 'audio':
        for layer in model.model.encoder.layers:
            layer.act_fn = ffn_object
        for layer in model.model.decoder.layers:
            layer.act_fn = ffn_object
    if model_type == 'vision':
        for stage in model.swinv2.encoder.layers:
            for block in stage.blocks:
                block.intermediate.intermediate_act_fn = ffn_object
    if model_type == 'video':
        for layer in model.vivit.encoder.layer:
            layer.intermediate.intermediate_act_fn = ffn_object

    return attention_object, ffn_object

def loop_configurations(batched_data, model_type, model, tok_proc, attention_operation, ffn_operation, nonlinear_config, save_dims):

    torch_softmax = torch.nn.functional.softmax
    torch_silu = torch.nn.functional.silu
    torch_gelu = torch.nn.functional.gelu

    default_dict = {
        'torch': {'attention_default': True, 'ffn_default': True},
        'attention': {'attention_default': False, 'ffn_default': True},
        'ffn': {'attention_default': True, 'ffn_default': False},
        'all': {'attention_default': False, 'ffn_default': False}
        }

    metric_df = None

    for approx_method, approx_config in nonlinear_config.items():

        attention_config = None
        ffn_config = None
        attention_combinations = None
        ffn_combinations = None

        nonlinear_config = {
            'device': model.device,
            'save_dims': save_dims,
            'profile': True if approx_method == 'torch' else False,
        }

        attention_config = approx_config.get('attention')
        ffn_config = approx_config.get('ffn')

        attention_config = flatten_dict_ranges(attention_config) if attention_config else None
        ffn_config = flatten_dict_ranges(ffn_config) if ffn_config else None

        if attention_config is not None:
            if 'profile' not in attention_config['path']:
                attention_config['path'] = os.path.join(
                "profile",
                attention_config['path'],
                os.path(str(model.config._name_or_path))
            )
            for key, value in attention_config.items():
                if not isinstance(value, list):
                    attention_config[key] = [value]

            attention_combinations = list(itertools.product(*attention_config.values()))
            attention_combinations = [dict(zip(attention_config.keys(), comb)) for comb in attention_combinations]

        if ffn_config is not None:
            if 'profile' not in ffn_config['path']:
                ffn_config['path'] = os.path.join(
                    "profile",
                    ffn_config['path'],
                    f"{model.config._name_or_path}"
            )
            for key, value in ffn_config.items():
                if not isinstance(value, list):
                    ffn_config[key] = [value]

            ffn_combinations = list(itertools.product(*ffn_config.values()))
            ffn_combinations = [dict(zip(ffn_config.keys(), comb)) for comb in ffn_combinations]

        attention_config = clean_lists(attention_config) if attention_config else None
        ffn_config = clean_lists(ffn_config) if ffn_config else None

        for default_key, defualt_value in default_dict.items():

            if approx_method == 'torch' and default_key != 'torch':
                continue

            attention_default = defualt_value['attention_default']
            ffn_default = defualt_value['ffn_default']

            attention_object = None
            ffn_object = None

            if default_key == 'torch':
                if attention_config is None:
                    attention_config = {'path': 'profile/null/'}
                if ffn_config is None:
                    ffn_config = {'path': 'profile/null/'}
                attention_object, ffn_object = set_nonlinear_operations(attention_default=attention_default,
                                                                        ffn_default=ffn_default,
                                                                        attention_operation=attention_operation,
                                                                        ffn_operation=ffn_operation,
                                                                        approx_method=approx_method,
                                                                        attention_config=attention_config,
                                                                        ffn_config=ffn_config,
                                                                        nonlinear_config=nonlinear_config,
                                                                        model=model,
                                                                        model_type=model_type,
                                                                        torch_softmax=torch_softmax,
                                                                        torch_silu=torch_silu,
                                                                        torch_gelu=torch_gelu,
                                                                        attention_object=attention_object,
                                                                        ffn_object=ffn_object)
                
                result, metric = run_inference(batched_data=batched_data,
                                               model_type=model_type,
                                               model=model,
                                               tok_proc=tok_proc)

                torch_dict = {
                    'approx_method': approx_method,
                    'attention': attention_object.__class__.__name__,
                    'ffn': ffn_object.__class__.__name__,
                    'model': model.config._name_or_path,
                    'metric': metric,
                    'value': result,
                    'type': default_key
                }

                torch_attention_df = pd.DataFrame(convert_dict_to_list_dict(attention_config))
                torch_ffn_df = pd.DataFrame(convert_dict_to_list_dict(ffn_config))

                torch_attention_df = torch_attention_df.add_suffix('_attention')
                torch_ffn_df = torch_ffn_df.add_suffix('_ffn')

                torch_df = pd.DataFrame(convert_dict_to_list_dict(torch_dict))

                torch_df = pd.concat([torch_df, torch_attention_df, torch_ffn_df], axis=1)
            
                if metric_df is None:
                    metric_df = torch_df
                else:
                    metric_df = pd.concat([metric_df, torch_df], ignore_index=True)
                
                # Clear intermediate DataFrames and CUDA cache
                del torch_dict, torch_attention_df, torch_ffn_df, torch_df
                torch.cuda.empty_cache()

            elif default_key == 'attention' and attention_combinations is not None:
                if ffn_config is None:
                    ffn_config = {'path': 'profile/null/'}
                for combination in attention_combinations:
                    attention_object, ffn_object = set_nonlinear_operations(attention_default=attention_default,
                                                                            ffn_default=ffn_default,
                                                                            attention_operation=attention_operation,
                                                                            ffn_operation=ffn_operation,
                                                                            approx_method=approx_method,
                                                                            attention_config=combination,
                                                                            ffn_config=ffn_config,
                                                                            model=model,
                                                                            model_type=model_type,
                                                                            torch_softmax=torch_softmax,
                                                                            torch_silu=torch_silu,
                                                                            torch_gelu=torch_gelu,
                                                                            nonlinear_config=nonlinear_config,
                                                                            attention_object=attention_object,
                                                                            ffn_object=ffn_object)
                    result, metric = run_inference(batched_data=batched_data,
                                                   model_type=model_type,
                                                   model=model,
                                                   tok_proc=tok_proc)


                    attention_dict = {
                        'approx_method': approx_method,
                        'attention': attention_object.__class__.__name__,
                        'ffn': ffn_object.__class__.__name__,
                        'model': model.config._name_or_path,
                        'metric': metric,
                        'value': result,
                        'type': default_key
                    }

                    attention_attention_df = pd.DataFrame(convert_dict_to_list_dict(combination))
                    attention_ffn_df = pd.DataFrame(convert_dict_to_list_dict(ffn_config))

                    attention_attention_df = attention_attention_df.add_suffix('_attention')
                    attention_ffn_df = attention_ffn_df.add_suffix('_ffn')

                    attention_df = pd.DataFrame(convert_dict_to_list_dict(attention_dict))

                    attention_df = pd.concat([attention_df, attention_attention_df, attention_ffn_df], axis=1)
                    
                    if metric_df is None:
                        metric_df = attention_df
                    else:
                        metric_df = pd.concat([metric_df, attention_df], ignore_index=True)
                    
                    # Clear intermediate DataFrames and CUDA cache
                    del attention_dict, attention_attention_df, attention_ffn_df, attention_df
                    torch.cuda.empty_cache()

            elif default_key == 'ffn' and ffn_combinations is not None:
                if attention_config is None:
                    attention_config = {'path': 'profile/null/'}
                for combination in ffn_combinations:
                    attention_object, ffn_object = set_nonlinear_operations(attention_default=attention_default,
                                                                            ffn_default=ffn_default,
                                                                            attention_operation=attention_operation,
                                                                            ffn_operation=ffn_operation,
                                                                            approx_method=approx_method,
                                                                            attention_config=attention_config,
                                                                            ffn_config=combination,
                                                                            model=model,
                                                                            model_type=model_type,
                                                                            torch_softmax=torch_softmax,
                                                                            torch_silu=torch_silu,
                                                                            torch_gelu=torch_gelu,
                                                                            nonlinear_config=nonlinear_config,
                                                                            attention_object=attention_object,
                                                                            ffn_object=ffn_object)
                    result, metric = run_inference(batched_data=batched_data,
                                                   model_type=model_type,
                                                   model=model,
                                                   tok_proc=tok_proc)
                    
                    ffn_dict = {
                        'approx_method': approx_method,
                        'attention': attention_object.__class__.__name__,
                        'ffn': ffn_object.__class__.__name__,
                        'model': model.config._name_or_path,
                        'metric': metric,
                        'value': result,
                        'type': default_key
                    }

                    ffn_attention_df = pd.DataFrame(convert_dict_to_list_dict(attention_config))
                    ffn_ffn_df = pd.DataFrame(convert_dict_to_list_dict(combination))

                    ffn_attention_df = ffn_attention_df.add_suffix('_attention')
                    ffn_ffn_df = ffn_ffn_df.add_suffix('_ffn')

                    ffn_df = pd.DataFrame(convert_dict_to_list_dict(ffn_dict))

                    ffn_df = pd.concat([ffn_df, ffn_attention_df, ffn_ffn_df], axis=1)

                    if metric_df is None:
                        metric_df = ffn_df
                    else:
                        metric_df = pd.concat([metric_df, ffn_df], ignore_index=True)
                    
                    # Clear intermediate DataFrames and CUDA cache
                    del ffn_dict, ffn_attention_df, ffn_ffn_df, ffn_df
                    torch.cuda.empty_cache()

            elif default_key == 'all' and attention_combinations is not None and ffn_combinations is not None:
                for attention_combination in attention_combinations:
                    for ffn_combination in ffn_combinations:
                        attention_object, ffn_object = set_nonlinear_operations(attention_default=attention_default,
                                                                                ffn_default=ffn_default,
                                                                                attention_operation=attention_operation,
                                                                                ffn_operation=ffn_operation,
                                                                                approx_method=approx_method,
                                                                                attention_config=attention_combination,
                                                                                ffn_config=ffn_combination,
                                                                                model=model,
                                                                                model_type=model_type,
                                                                                torch_softmax=torch_softmax,
                                                                                torch_silu=torch_silu,
                                                                                torch_gelu=torch_gelu,
                                                                                nonlinear_config=nonlinear_config,
                                                                                attention_object=attention_object,
                                                                                ffn_object=ffn_object)
                        result, metric = run_inference(batched_data=batched_data,
                                                       model_type=model_type,
                                                       model=model,
                                                       tok_proc=tok_proc)
                        
                        all_dict = {
                            'approx_method': approx_method,
                            'attention': attention_object.__class__.__name__,
                            'ffn': ffn_object.__class__.__name__,
                            'model': model.config._name_or_path,
                            'metric': metric,
                            'value': result,
                            'type': default_key
                        }

                        all_attention_df = pd.DataFrame(convert_dict_to_list_dict(attention_combination))
                        all_ffn_df = pd.DataFrame(convert_dict_to_list_dict(ffn_combination))

                        all_attention_df = all_attention_df.add_suffix('_attention')
                        all_ffn_df = all_ffn_df.add_suffix('_ffn')

                        all_df = pd.DataFrame(convert_dict_to_list_dict(all_dict))

                        all_df = pd.concat([all_df, all_attention_df, all_ffn_df], axis=1)

                        if metric_df is None:
                            metric_df = all_df
                        else:
                            metric_df = pd.concat([metric_df, all_df], ignore_index=True)
                        
                        # Clear intermediate DataFrames and CUDA cache
                        del all_dict, all_attention_df, all_ffn_df, all_df
                        torch.cuda.empty_cache()
            
            # Clear configuration objects and force garbage collection
            if attention_object is not None:
                del attention_object
            if ffn_object is not None:
                del ffn_object
            torch.cuda.empty_cache()
        if attention_combinations is not None:
            del attention_combinations
        if ffn_combinations is not None:
            del ffn_combinations
            

    return metric_df

def evaluate_model(model_dict, dataset_dict, parameter_dict, nonlinear_operations, nonlinear_config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = model_dict.get('model_type')
    model_name = model_dict.get('model_name')

    dataset_name = dataset_dict.get('dataset')
    dataset_config = dataset_dict.get('config')
    dataset_split = dataset_dict.get('split')

    attention_operation = nonlinear_operations.get('attention')
    ffn_operation = nonlinear_operations.get('ffn')

    tok_proc, model, parameter_dict = load_model(
        model_type=model_type,
        model_name=model_name,
        device=device,
        parameter_dict=parameter_dict
    )

    n_samples = parameter_dict.get('n_samples')
    batch_size = parameter_dict.get('batch_size')
    max_length = parameter_dict.get('max_length')
    target_sample_rate = parameter_dict.get('target_sample_rate')
    num_frames = parameter_dict.get('num_frames')

    dataset = load_config_dataset(
        dataset_name,
        dataset_config,
        dataset_split
    )

    processed_dataset = process_dataset(
        model_type=model_type,
        tok_proc=tok_proc,
        dataset=dataset,
        n_samples=n_samples,
        max_length=max_length if model_type == 'nlp' else None,
        target_sample_rate=target_sample_rate if model_type == 'audio' else None,
        num_frames=num_frames if model_type == 'video' else None
    )

    batched_data = batch_dataset(
        processed_dataset=processed_dataset,
        num_samples=n_samples,
        batch_size=batch_size,
        model_type=model_type,
        tokenizer=tok_proc if model_type == 'nlp' else None
    )
    max_length = max_length - 1
    save_dims = [max_length // 4, max_length // 2, max_length]

    pd.DataFrame(columns=['Approx_method', 'Model', 'Dataset', 'Metric', 'Value'])

    metric_df = loop_configurations(
                                            batched_data=batched_data,
                                            model_type=model_type,
                                            model=model,
                                            tok_proc=tok_proc,
                                            attention_operation=attention_operation,
                                            ffn_operation=ffn_operation,
                                            nonlinear_config=nonlinear_config,
                                            save_dims=save_dims    
                                        )

    del model, tok_proc, processed_dataset, batched_data
    torch.cuda.empty_cache()
    
    save_path = f"results/{model_type}/{model_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metric_df.to_csv(f"{save_path}.csv", index=False)

def huggingface_login():
    """
    Login to Hugging Face.
    """
    token = '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'
    try:
        login(token=token)
        print("Successfully logged in to Hugging Face.")
    except:
        print("HF_TOKEN invalid or not set.")
        exit()

def main():
    parser = argparse.ArgumentParser(description='Run transformer model profiling')
    parser.add_argument('--config', type=str, default='model_config.yaml', 
                        help='Path to model config YAML file (default: model_config.yaml)')
    args = parser.parse_args()
    
    # for debugging
    # torch.set_printoptions(threshold=float('inf'))
    huggingface_login()

    config = yaml.safe_load(open(args.config))
    parameter_dict = yaml.safe_load(open('parameter_config.yaml'))
    nonlinear_config = yaml.safe_load(open('nonlinear_test.yaml'))

    for model_type, model_config in config.items():
        model_type = model_type.split('_')[0]
        dataset_config = model_config.get('datasets')
        models = model_config.get('models')
        nonlinear_operations = model_config.get('nonlinear')

        for dataset, dataset_dict in dataset_config.items():
            for model in models:
                model_dict = {
                    'model_type': model_type,
                    'model_name': model
                }

                evaluate_model(model_dict, dataset_dict, parameter_dict, nonlinear_operations, nonlinear_config)

if __name__ == '__main__':
    main()