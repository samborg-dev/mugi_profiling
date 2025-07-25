from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoModelForImageClassification, AutoModelForVideoClassification
from datasets import load_dataset
import torch
import math
import yaml
from tqdm import tqdm
from utils import process_nlp_dataset, process_audio_dataset, process_image_dataset, process_video_dataset

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
        # No max_length parameter needed for vision models

    elif model_type == 'video':
        tok_proc = AutoProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForVideoClassification.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation='eager').to(device)
        parameter_dict['num_frames'] = model.config.num_frames
        #parameter_dict['max_length'] = model.config.max_position_embeddings

    model.eval()
    return tok_proc, model, parameter_dict

def load_config_dataset(dataset_name, config, split):
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
        
    elif model_type == 'audio':
        input_features = torch.cat([ex["input_features"] for ex in batch], dim=0).to(model.device).to(torch.float16)
        labels = torch.cat([ex["labels"] for ex in batch], dim=0).to(model.device)
        labels[labels == tok_proc.tokenizer.pad_token_id] = -100
        with torch.no_grad():
            outputs = model(input_features=input_features, labels=labels, use_cache=False)
        
    elif model_type == 'vision':
        pixel_values = torch.stack([ex["pixel_values"].squeeze(0) for ex in batch]).to(model.device).to(torch.float16)
        labels = torch.stack([ex["labels"] for ex in batch]).squeeze(-1).to(model.device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, labels=labels)
        
    elif model_type == 'video':
        pixel_values = torch.stack([ex["pixel_values"].squeeze(0) for ex in batch]).to(model.device).to(torch.float16)
        labels = torch.stack([ex["labels"] for ex in batch]).squeeze(-1).to(model.device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, labels=labels)

    return outputs.loss

def compute_perplexity(loss, num_batches):
    assert num_batches > 0, "num_batches must be greater than 0"
    return math.exp(loss / num_batches)

def compute_average_loss(loss, num_batches):
    assert num_batches > 0, "num_batches must be greater than 0"
    return loss / num_batches

def evaluate_model(model_dict, dataset_dict, parameter_dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = model_dict.get('model_type')
    model_name = model_dict.get('model_name')

    dataset_name = dataset_dict.get('dataset')
    dataset_config = dataset_dict.get('config')
    dataset_split = dataset_dict.get('split')

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
        del batch
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

    del model, tok_proc, processed_dataset
    torch.cuda.empty_cache()
    
    return result, metric_name

def main():
    print('hi')
    exit()
    config = yaml.safe_load(open('model_config.yaml'))
    parameter_dict = yaml.safe_load(open('parameter_config.yaml'))

    for model_type, model_config in config.items():
        model_type = model_type.split('_')[0]
        dataset_config = model_config.get('datasets')
        models = model_config.get('models')
        for dataset, dataset_dict in dataset_config.items():
            for model in models:
                model_dict = {
                    'model_type': model_type,
                    'model_name': model
                }

                result, metric_name = evaluate_model(model_dict, dataset_dict, parameter_dict)
                print(f"Model: {model}, Dataset: {dataset}, {metric_name}: {result:.2f}")

if __name__ == '__main__':
    main()