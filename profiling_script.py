from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
import torch
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
    
def run_inference():
    print(f"Loading model: {model_name}")
    print(f"Loading dataset: {dataset_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
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

    # Load only a small subset of the dataset to avoid downloading the entire thing
    print("Loading small subset of dataset...")
    dataset = load_dataset(dataset_name, dataset_config, split="validation", streaming=True)
    
    # Take only the first 100 samples from the streaming dataset
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
        inputs = tokenizer.encode(sample_text, return_tensors="pt", truncation=True, max_length=100)
        inputs = inputs.to(device)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Input length: {inputs.shape[1]} tokens")
        print(f"Input preview: {sample_text[:200]}...")
        
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