from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from huggingface_hub import login
import torch
import yaml
import librosa

torch.set_printoptions(threshold=float('inf'))

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

torch_softmax = torch.nn.functional.softmax

def custom_softmax(attn_weights, dim, dtype=torch.float32, use_approx=False):
    if use_approx:
        pass
    else:
        if (len(attn_weights.shape) > 2):
            print(attn_weights.shape)
        return torch_softmax(attn_weights, dim=dim, dtype=torch.float32)


torch.nn.functional.softmax = custom_softmax

def nlp_inference(tokenizer, model, dataset, device, max_seq_length):
    total_loss = 0.0
    total_tokens = 0
    target_seq_length = int(max_seq_length * 0.9)  # Use 90% of max to leave room for special tokens
    processed_chunks = 0
    processed_samples = 0
    
    print(f"Combining C4 samples to create chunks of ~{target_seq_length} tokens (90% of {max_seq_length})")
    
    # Collect and combine samples
    combined_text = ""
    combined_tokens = []
    sample_count_in_chunk = 0
    
    for i, sample in enumerate(dataset):
        if processed_chunks >= 2:  # Process only 2 combined chunks for testing
            break
            
        text = sample["text"].strip()
        if not text:
            continue
            
        # Add separator between documents (double newline)
        if combined_text:
            combined_text += "\n\n" + text
        else:
            combined_text = text
            
        # Check current length
        current_tokens = tokenizer.encode(combined_text, truncation=False)
        sample_count_in_chunk += 1
        
        # If we've reached target length or processed many samples, create a chunk
        if len(current_tokens) >= target_seq_length or sample_count_in_chunk >= 10:
            # Use the combined text directly and let tokenizer handle truncation properly
            final_text = combined_text
            actual_token_count = len(current_tokens)
            
            # If too long, we'll truncate during tokenization later
            if actual_token_count > target_seq_length:
                actual_token_count = target_seq_length
            
            print(f"\nProcessing combined chunk {processed_chunks}")
            print(f"Combined {sample_count_in_chunk} C4 samples")
            print(f"Total tokens in chunk: {actual_token_count}")
            print(f"Text preview: {final_text[:150]}...")
            
            # Move inputs to the same device as model
            inputs = tokenizer(final_text, return_tensors="pt", truncation=True, max_length=max_seq_length, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Calculate perplexity by having model predict next tokens (INFERENCE MODE)
            with torch.no_grad():
                # Forward pass to get predictions for each token position
                model_outputs = model(**inputs, labels=inputs["input_ids"])
                loss = model_outputs.loss
                logits = model_outputs.logits
                
                # Count actual tokens (excluding padding)
                num_tokens = inputs["attention_mask"].sum().item()
                
                # Accumulate loss and token count for overall perplexity
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Calculate perplexity for this sample
                sample_perplexity = torch.exp(loss).item()
                print(f"Chunk {processed_chunks} perplexity: {sample_perplexity:.2f}")
                print(f"Chunk {processed_chunks} tokens processed: {num_tokens}")
                
                # Optional: Show per-token information for profiling
                if processed_chunks == 0:  # Only for first chunk to avoid spam
                    print(f"Logits shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
                    print(f"Input IDs shape: {inputs['input_ids'].shape}")
                    print(f"Attention mask shape: {inputs['attention_mask'].shape}")
                    
                # DECODING PROFILING - profile as LLM decodes sequence step by step
                print(f"\n--- Starting decoding profiling ---")
                # Use first part as prefill prompt
                prefill_length = 128  # Use 1/3 as prefill
                max_new_tokens = max_seq_length  # Generate up to target length
                
                prefill_ids = inputs["input_ids"][:, :prefill_length]
                print(f"Prefill length: {prefill_length} tokens")
                print(f"Will generate up to {max_new_tokens} new tokens")
                
                # Manual decoding loop to capture attention at each step
                current_ids = prefill_ids.clone()
                generated_tokens = 0
                
                for step in range(max_new_tokens):
                    # Forward pass for current sequence
                    with torch.no_grad():
                        outputs = model(input_ids=current_ids)
                        logits = outputs.logits
                        
                        # Get next token
                        next_token_logits = logits[:, -1, :]  # Last position
                        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        # Append to sequence
                        current_ids = torch.cat([current_ids, next_token_id], dim=-1)
                        generated_tokens += 1
                        current_length = prefill_length + generated_tokens
                        
                        print(f"Step {step + 1}: Generated token, sequence length now {current_length}")
                        
                        # Stop if we hit EOS or reach target
                        if next_token_id.item() == tokenizer.eos_token_id:
                            print(f"Hit EOS token, stopping generation")
                            break
                
                # Calculate perplexity on the final generated sequence
                final_sequence = current_ids
                final_length = final_sequence.shape[1]
                
                # Calculate perplexity on generated portion
                if final_length > prefill_length:
                    with torch.no_grad():
                        # Use the generated sequence for perplexity calculation
                        perplexity_outputs = model(input_ids=final_sequence, labels=final_sequence)
                        generation_loss = perplexity_outputs.loss
                        generation_perplexity = torch.exp(generation_loss).item()
                        
                        print(f"Final sequence length: {final_length}")
                        print(f"Generated {generated_tokens} new tokens")
                        print(f"Generation perplexity: {generation_perplexity:.2f}")
                        
                        # Decode the generated text to see what was produced
                        generated_text = tokenizer.decode(final_sequence[0], skip_special_tokens=True)
                        print(f"Generated text preview: {generated_text[-(min(200, len(generated_text))):]}...")
                else:
                    print("No tokens were generated")
            
            processed_chunks += 1
            processed_samples += sample_count_in_chunk
            
            # Reset for next chunk
            combined_text = ""
            sample_count_in_chunk = 0
    
    # Calculate overall perplexity across all chunks
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        overall_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"\n=== SUMMARY ===")
        print(f"Processed chunks: {processed_chunks}")
        print(f"Total C4 samples combined: {processed_samples}")
        print(f"Overall perplexity: {overall_perplexity:.2f}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Average tokens per chunk: {total_tokens/processed_chunks:.1f}")
    else:
        print(f"\nNo valid chunks created from C4 data")

def audio_inference(processor, model, dataset, device):
    """
    Perform inference on audio datasets using Whisper models.
    """
    # Get the model's expected sampling rate (Whisper uses 16000 Hz)
    model_sampling_rate = 16000
    
    print(f"Using model sampling rate: {model_sampling_rate}")
    
    for i, audio_sample in enumerate(dataset):
        if i >= 1:  # Process only first sample for testing
            break
            
        # Get audio data - handle different dataset formats
        if "audio" in audio_sample:
            # Standard audio format (LibriSpeech, etc.)
            if isinstance(audio_sample["audio"], dict) and "array" in audio_sample["audio"]:
                audio_data = audio_sample["audio"]["array"]
                original_sampling_rate = audio_sample["audio"]["sampling_rate"]
            else:
                # Common Voice format where audio might be directly the array
                audio_data = audio_sample["audio"]
                original_sampling_rate = 48000  # Common Voice default
                
            # Resample if needed
            if original_sampling_rate != model_sampling_rate:
                audio_data = librosa.resample(audio_data, orig_sr=original_sampling_rate, target_sr=model_sampling_rate)
                
        elif "path" in audio_sample:
            # Common Voice might use 'path' instead of 'file'
            print(f"Found path field: {audio_sample['path']}")
            # Skip for now since we need actual audio data
            continue
            
        elif "file" in audio_sample:
            # For datasets that store file paths
            audio_data = audio_sample["file"]
            
        else:
            print(f"Skipping sample {i}: No audio data found")
            print(f"Available keys: {list(audio_sample.keys())}")
            continue
        
        # Process audio input for Whisper (pad/truncate to 30 seconds)
        inputs = processor(
            audio_data, 
            sampling_rate=model_sampling_rate, 
            return_tensors="pt",
            padding="max_length",
            max_length=30 * model_sampling_rate,  # 30 seconds in samples
            truncation=True
        )
        
        # Move inputs to device and convert to model's dtype
        inputs = {k: v.to(device).to(torch.float16) for k, v in inputs.items()}

        with torch.no_grad():
            # Whisper generation with proper configuration
            outputs = model.generate(
                input_features=inputs["input_features"],
                max_new_tokens=100,
                do_sample=False,
                language="en",  # Force English to avoid language detection
                task="transcribe"  # Explicit task instead of deprecated forced_decoder_ids
            )
            
            # Decode the generated text
            transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"Sample {i} transcription: {transcription}")

def vision_inference(processor, model, dataset, device):
    for i, sample in enumerate(dataset):
        if i >= 1:
            break
            
        if "img" in sample:
            image = sample["img"]
        elif "image" in sample:
            image = sample["image"]
        else:
            print(f"Skipping sample {i}: No image data found")
            continue
        
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device).to(torch.float16) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
            predicted_label = model.config.id2label[predicted_class_id]
            confidence = predictions.max().item()
            print(f"Sample {i} prediction: {predicted_label} (confidence: {confidence:.3f})")
                

def run_inference(inf_type, model_name, dataset_name, dataset_config = None, dataset_split="validation"):
    print(f"Loading model: {model_name}")
    print(f"Loading dataset: {dataset_name}")

    torch.nn.functional.softmax = custom_softmax
    
    # CUDA device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    if inf_type == "nlp":
        # Load NLP model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"  # Force eager attention
        )

        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1000000:
            max_seq_length = tokenizer.model_max_length
        else:
            max_seq_length = getattr(model.config, 'max_position_embeddings', 4096)
        
        print(f"Using max sequence length: {max_seq_length}")

        if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        print(dataset_name, dataset_config, dataset_split)

        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True, trust_remote_code=True)
        model.eval()
        nlp_inference(tokenizer, model, dataset, device, max_seq_length)
        
    elif inf_type == "audio":
        # Load Whisper model and processor
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(dataset_name, dataset_config, dataset_split)

        try:
            dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load with config {dataset_config}, trying without config...")
            try:
                dataset = load_dataset(dataset_name, split=dataset_split, streaming=True, trust_remote_code=True)
            except Exception as e2:
                print(f"Failed to load dataset: {e2}")
                return
                
        dataset = dataset.take(100)
        model.eval()

        audio_inference(processor, model, dataset, device)
        
    elif inf_type == "vision":
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(dataset_name, dataset_config, dataset_split)

        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True, trust_remote_code=True)
        dataset = dataset.take(100)
        model.eval()

        vision_inference(processor, model, dataset, device)

if __name__ == "__main__":
    huggingface_login()

    with open("model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if key != 'nlp':
            continue
        datasets = value.get("datasets", [])
        models = value.get("models", [])

        for dataset, dataset_values in datasets.items():
            dataset_name = dataset_values.get("name", None)
            dataset_config = dataset_values.get("config", None)
            dataset_split = dataset_values.get("split", None)

            for model_name in models:
                # try:
                    print(f"\nRunning inference for model: {model_name} on dataset: {dataset_name}")
                    run_inference(key, model_name, dataset_name, dataset_config, dataset_split)
                # except Exception as e:
                #     print(f"Error running inference for model: {model_name} on dataset: {dataset_name}")
                #     print(e)
                #     exit()
