import torch
import librosa
from tqdm import tqdm
import io
from decord import VideoReader, cpu
from PIL import Image

# ----- NLP processing functions -----
def process_nlp_dataset(dataset, tokenizer, n_samples, max_length):
    subset = []
    for example in tqdm(dataset, desc="Preprocessing dataset"):
        if len(subset) >= n_samples:
            break
        if len(example["text"]) > max_length:
            tokenized_example = tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            if tokenized_example['input_ids'].shape[-1] >= max_length:
                subset.append(tokenized_example)
    return subset

# ----- Audio processing functions -----
def process_audio(audio_sample, target_sample_rate):
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

def process_audio_dataset(dataset, processor, n_samples, target_sample_rate):
        
    subset = list(dataset.take(n_samples))
    processed_examples = []

    for example in tqdm(subset, desc="Processing audio"):
        audio_array = process_audio(example["audio"], target_sample_rate)
        
        inputs = processor(
            audio_array,
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            truncation=True
        )
        
        inputs["input_features"] = inputs["input_features"].to(torch.float16)
        
        text = example["text"]
        
        # Manually construct special tokens for Whisper
        start_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        lang_token = processor.tokenizer.convert_tokens_to_ids("<|en|>")  # English
        task_token = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        
        # process text
        text_only_tokens = processor.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].squeeze(0)
        
        # add in special tokens
        full_sequence = torch.cat([
            torch.tensor([start_token, lang_token, task_token, notimestamps_token]),
            text_only_tokens,
            torch.tensor([processor.tokenizer.eos_token_id])
        ]).unsqueeze(0)
        
        text_inputs = {"input_ids": full_sequence}

        processed_example = {
            "input_features": inputs["input_features"],
            "labels": text_inputs["input_ids"],
            "text": text
        }

        processed_examples.append(processed_example)
    
    return processed_examples

def process_image_dataset(dataset, processor, n_samples):
    subset = list(dataset.take(n_samples))
    processed_examples = []

    for example in tqdm(subset, desc="Processing images"):
        img = example["image"] if 'image' in example else example['jpg']

        inputs = processor(
            images=img,
            return_tensors="pt"
        )

        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        label = example["label"] if 'label' in example else example['cls']

        processed_example = {
            "pixel_values": inputs["pixel_values"],
            "labels": torch.tensor([label], dtype=torch.long),
            "label": label
        }

        processed_examples.append(processed_example)

    return processed_examples

def process_video_dataset(dataset, processor, n_samples, num_frames):
    subset = list(dataset.take(n_samples))
    processed_examples = []

    for example in tqdm(subset, desc="Processing videos"):
        video_reader = VideoReader(io.BytesIO(example['video']), ctx=cpu(0))
        frames_array = video_reader.get_batch(range(min(num_frames, len(video_reader)))).asnumpy()
        
        frames_pil = [Image.fromarray(frame) for frame in frames_array]
        inputs = processor(frames_pil, return_tensors='pt')

        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        label = example["label"]

        processed_example = {
            "pixel_values": inputs["pixel_values"],
            "labels": torch.tensor([label], dtype=torch.long),
            "label": label
        }

        processed_examples.append(processed_example)

    return processed_examples