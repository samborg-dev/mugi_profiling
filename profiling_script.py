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
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision for memory efficiency
        device_map="auto"  # Automatically distribute model across available GPUs
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_seq_length = model.config.max_position_embeddings

    dataset = load_dataset(dataset_name, dataset_config, split="test[:1%]")
    dataset = dataset.map(tokenizer, batched=True, num_proc=len(os.sched_getaffinity(0)))

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
                max_new_tokens=50,  # Added back max_new_tokens for controlled generation
                num_return_sequences=1,
                temperature=0.7,  # Added back temperature for better generation
                do_sample=True,  # Added back sampling
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("-" * 50)

    print("\nFinished running inference on sample texts.")

if __name__ == "__main__":
    huggingface_login()
    run_inference()