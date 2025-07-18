from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
import torch
import os

model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "c4"
dataset_config = "en"

def huggingface_login():
    token = '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'
    try:
        print("Using HF_TOKEN from environment variable")
        login(token=token)
    except:
        print("HF_TOKEN invalid or not set.")
        exit()

def run_inference():
    print(f"Loading model: {model_name}")
    print(f"Loading dataset: {dataset_name}")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_seq_length = model.config.max_position_embeddings

    dataset = load_dataset(dataset_name, dataset_config, split="test")
    
    model.eval()
    
    print("\nRunning inference on sample texts...")
    
    for i in range(min(3, len(dataset))):
        full_text = dataset[i]["text"]
        tokens = tokenizer.encode(full_text, truncation=False)
        if len(tokens) > max_seq_length:
            truncated_tokens = tokens[:max_seq_length]
            sample_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        else:
            sample_text = full_text
        if not sample_text.strip():
            continue
        
        inputs = tokenizer.encode(sample_text, return_tensors="pt", truncation=True, max_length=100)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nFinished running inference on sample texts.")

if __name__ == "__main__":
    huggingface_login()
    run_inference()