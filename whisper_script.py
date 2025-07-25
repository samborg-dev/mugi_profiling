from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset
import torch
import math
import librosa
from tqdm import tqdm
from custom_nonlinear_functions import vlp_silu_approx, vlp_softmax_approx, pwl_softmax_approx, pwl_silu_approx, pwl_mobilenet_approx, taylor_softmax_approx

# ----- Config -----
torch.set_printoptions(linewidth=200)
model_name = "openai/whisper-base"
split = "validation"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
n_samples = 10
target_sample_rate = 16000
softmax_exp_dim = 16
softmax_max_exp = 3
softmax_min_exp = -4
vlp_build = 'max'
save_dims = [1024, 2048, 4096]
window_size = 32

softmax = vlp_softmax_approx.VLPSoftmax(exp_dim=softmax_exp_dim, max_exp=softmax_max_exp, min_exp=softmax_min_exp, window_size=window_size, lut_build=vlp_build, device=device, path='profile/vlp/softmax/', save_dims=save_dims, profile=False)

# ----- Patch Softmax Function -----
#torch.nn.functional.softmax = softmax.forward

# Load processor and model
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation='eager'
).to(device)

model.eval()

max_seq_length = getattr(processor.feature_extractor, 'nb_max_frames')

# ----- Load dataset -----
# Use LibriSpeech instead of Common Voice for better compatibility
dataset = load_dataset("openslr/librispeech_asr", split="test.clean", streaming=True)

# Take first n_samples directly from dataset
subset = list(dataset.take(n_samples))

def process_audio(audio_sample):
    """Process audio to match Whisper's expected format"""
    audio_array = audio_sample["array"]
    original_sr = audio_sample["sampling_rate"]
    
    # Resample to 16kHz if needed
    if original_sr != target_sample_rate:
        audio_array = librosa.resample(
            audio_array, 
            orig_sr=original_sr, 
            target_sr=target_sample_rate
        )
    
    return audio_array

def get_audio_duration(audio_sample):
    """Get duration of audio in seconds"""
    return len(audio_sample["array"]) / audio_sample["sampling_rate"]

# ----- Preprocessing function -----
def process_sample(example):
    audio_array = process_audio(example["audio"])
    
    # Process audio for Whisper
    inputs = processor(
        audio_array,
        sampling_rate=target_sample_rate,
        return_tensors="pt",
        truncation=True,

    )
    
    # Convert to float16 to match model dtype
    inputs["input_features"] = inputs["input_features"].to(torch.float16)
    
    # Get transcription tokens
    # For Whisper, we need to properly format the sequence with required special tokens
    text = example["text"]
    
    # Alternative approach: manually construct the proper sequence
    # This ensures we have the exact format Whisper expects during training
    start_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    lang_token = processor.tokenizer.convert_tokens_to_ids("<|en|>")  # English
    task_token = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_token = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    
    # Tokenize just the text without special tokens
    text_only_tokens = processor.tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].squeeze(0)
    
    # Construct the full sequence manually
    full_sequence = torch.cat([
        torch.tensor([start_token, lang_token, task_token, notimestamps_token]),
        text_only_tokens,
        torch.tensor([processor.tokenizer.eos_token_id])
    ]).unsqueeze(0)
    
    text_inputs = {"input_ids": full_sequence}
    
    return {
        "input_features": inputs["input_features"],
        "labels": text_inputs["input_ids"],
        "text": text
    }

processed_subset = [process_sample(example) for example in tqdm(subset, desc="Processing audio")]

# ----- Perplexity calculation -----
def compute_loss(input_features, labels):
    with torch.no_grad():
        # Prepare labels for Whisper (mask padding tokens)
        labels_masked = labels.clone()
        labels_masked[labels_masked == processor.tokenizer.pad_token_id] = -100
        
        # Whisper expects labels to include special tokens for proper loss calculation

        

        outputs = model(
            input_features=input_features,
            labels=labels_masked,
            use_cache=False
        )
        return outputs.loss

# ----- Batch processing -----
total_loss = 0.0
num_batches = 0
batch_loss = []
total_tokens = 0

for i in tqdm(range(0, len(subset), batch_size), desc="Evaluating"):
    raw_batch = subset[i:i+batch_size]
    
    # Process each example in the batch
    batch = [process_sample(example) for example in raw_batch]
    
    # Get maximum sequence length in batch
    batch_max_len = max(ex["labels"].shape[1] for ex in batch)
    
    # Stack input features
    input_features = torch.stack([ex["input_features"].squeeze(0) for ex in batch])
    
    # Pad labels to same length
    labels = torch.nn.utils.rnn.pad_sequence(
        [ex["labels"].squeeze(0)[:batch_max_len] for ex in batch],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    
    # Move to device and ensure correct dtype
    input_features = input_features.to(device).to(torch.float16)
    labels = labels.to(device)
    
    # Compute loss
    loss = compute_loss(input_features, labels)
    
    # Count actual tokens (excluding padding)
    num_tokens_batch = (labels != processor.tokenizer.pad_token_id).sum().item()
    
    total_loss += loss.item()
    num_batches += 1
    batch_loss.append(loss.item())
    total_tokens += num_tokens_batch
    
    del input_features, labels
    torch.cuda.empty_cache()

# ----- Calculate perplexity -----
if num_batches > 0:
    average_loss = total_loss / num_batches
    perplexity = math.exp(average_loss)
    
    print(f"\n=== Results ===")
    print(f"Processed samples: {len(subset)}")
    print(f"Total batches: {num_batches}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per sample: {total_tokens/len(subset):.1f}")
    print(f"Average loss: {average_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    # Show individual batch losses
    print(f"\nBatch losses: {batch_loss}")
else:
    print("No samples processed!")
