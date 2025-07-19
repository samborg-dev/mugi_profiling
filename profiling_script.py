from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
import torch
import pyyaml
import os
import multiprocessing

model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "allenai/c4"
dataset_config = "en"

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
    
def run_inference(model_name, dataset_name, dataset_config = None, dataset_split="validation"):
    print(f"Loading model: {model_name}")
    print(f"Loading dataset: {dataset_name}")
    
    # CUDA device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_seq_length = model.config.max_position_embeddings
    print(f"Model max sequence length: {max_seq_length}")

    # Load Dataset
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
    
    # Temp Testing on first 100 samples
    dataset = dataset.take(100)
    dataset = list(dataset)
    print(f"Loaded {len(dataset)} samples from dataset")

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
        
        # Move inputs to the same device as model
        inputs = tokenizer.encode(sample_text, return_tensors="pt", truncation=True, max_length=max_seq_length)
        inputs = inputs.to(device)
        
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

    with open("model_config.yaml", "r") as f:
        config = pyyaml.safe_load(f)

    for key, value in config.items():
        datasets = value.get("datasets", [])
        models = value.get("models", [])

        for dataset, dataset_config in datasets.items():
            dataset_name = dataset_config.get("name", None)
            dataset_config = dataset_config.get("config", None)
            dataset_split = dataset_config.get("split", None)

            for model_name in models:
                print(f"\nRunning inference for model: {model_name} on dataset: {dataset_name}")
                exit()
                run_inference(model_name, dataset_name, dataset_config, dataset_split)
                exit()