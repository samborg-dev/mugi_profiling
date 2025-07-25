import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVideoClassification
from PIL import Image
from itertools import islice
from tqdm import tqdm
from decord import VideoReader, cpu
import io
import math

def compute_loss(inputs, labels, model):
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        return outputs.loss

# Load the UCF101 dataset using streaming mode

def main():

    batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'google/vivit-b-16x2'
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVideoClassification.from_pretrained(model_name).to(device)

    num_frames = model.config.num_frames

    dataset = load_dataset('nateraw/kinetics', split='validation', streaming=True)

    videos = []
    labels = []
    
    for i, sample in enumerate(islice(dataset, 10)):
        videos.append(sample['video'])
        labels.append(sample['label'])

    print(f"Loaded {len(videos)} videos")
    print(f"First video type: {type(videos[0])}")
    print(f"Labels: {labels}")

    total_loss = 0.0
    num_batches = 0
    batch_loss = []
    total_samples = 0

    for i in tqdm(range(0, len(videos), batch_size), desc="Evaluating"):
        batch_videos = videos[i:i+batch_size]
        video_readers = [VideoReader(io.BytesIO(video), ctx=cpu(0)) for video in batch_videos]
        
        # Extract frames and convert to list of PIL Images for each video
        batch_decoded_videos = []
        for reader in video_readers:
            # Get frames as numpy array: (num_frames, height, width, channels)
            frames_array = reader.get_batch(range(min(num_frames, len(reader)))).asnumpy()
            print(f"Raw frames shape: {frames_array.shape}")
            
            # Convert each frame to PIL Image
            frames_pil = [Image.fromarray(frame) for frame in frames_array]
            batch_decoded_videos.append(frames_pil)

        print(f"Number of videos in batch: {len(batch_decoded_videos)}")
        print(f"Frames per video: {len(batch_decoded_videos[0])}")
        print(f"First frame type: {type(batch_decoded_videos[0][0])}")
        print(f"First frame size: {batch_decoded_videos[0][0].size}")

        batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.long).to(device)

        batch_inputs = processor(batch_decoded_videos, return_tensors='pt').to(device)
        
        # Compute loss
        loss = compute_loss(batch_inputs, batch_labels, model)
        
        total_loss += loss.item()
        num_batches += 1
        batch_loss.append(loss.item())
        total_samples += len(batch_videos)
        
        del batch_videos, batch_labels
        torch.cuda.empty_cache()

    print(batch_loss)
    print(f"{math.exp(total_loss / num_batches):.2f}")

if __name__ == '__main__':
    main()