from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
import math
from tqdm import tqdm
from PIL import Image

# ----- Config -----
torch.set_printoptions(linewidth=200)
model_name = "microsoft/swinv2-tiny-patch4-window8-256"
split = "test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
n_samples = 10

# Load processor and model
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation='eager'
).to(device)

model.eval()

# ----- Load dataset -----
# Use CIFAR-100 which has 100 fine-grained image classes
dataset = load_dataset("cifar100", split=split, streaming=True)

# Take first n_samples directly from dataset
subset = list(dataset.take(n_samples))

def process_image(image):
    """Process image to match SwinV2's expected format"""
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

# ----- Preprocessing function -----
def process_sample(example):
    image = process_image(example["img"])  # CIFAR-100 uses 'img' field
    
    # Process image for SwinV2
    inputs = processor(
        images=image,
        return_tensors="pt"
    )
    
    # Convert to float16 to match model dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    
    # Get labels - CIFAR-100 uses 'fine_label' for the 100 classes
    label = example["fine_label"]
    
    processed_example = {
        "pixel_values": inputs["pixel_values"],
        "labels": torch.tensor([label], dtype=torch.long),
        "label": label
    }

    return processed_example

# ----- Loss calculation -----
def compute_loss(pixel_values, labels):
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, labels=labels)
        return outputs.loss

# ----- Batch processing -----
total_loss = 0.0
num_batches = 0
batch_loss = []
total_samples = 0

for i in tqdm(range(0, len(subset), batch_size), desc="Evaluating"):
    raw_batch = subset[i:i+batch_size]
    
    # Process each example in the batch
    batch = [process_sample(example) for example in raw_batch]
    
    # Stack pixel values and labels
    pixel_values = torch.stack([ex["pixel_values"].squeeze(0) for ex in batch])
    labels = torch.stack([ex["labels"] for ex in batch]).squeeze(-1)
    
    # Move to device and ensure correct dtype
    pixel_values = pixel_values.to(device).to(torch.float16)
    labels = labels.to(device)
    
    # Compute loss
    loss = compute_loss(pixel_values, labels)
    
    total_loss += loss.item()
    num_batches += 1
    batch_loss.append(loss.item())
    total_samples += len(batch)
    
    del pixel_values, labels
    torch.cuda.empty_cache()

# ----- Calculate average loss (perplexity equivalent for classification) -----
if num_batches > 0:
    average_loss = total_loss / num_batches
    
    # For classification, we can compute accuracy-like metrics
    # But we'll keep the loss-based approach for consistency
    
    print(f"\n=== Results ===")
    print(f"Processed samples: {len(subset)}")
    print(f"Total batches: {num_batches}")
    print(f"Total samples: {total_samples}")
    print(f"Average loss: {average_loss:.4f}")
    print(f"Exponential of average loss (perplexity-like): {math.exp(average_loss):.2f}")
    
    # Show individual batch losses
    print(f"\nBatch losses: {batch_loss}")
else:
    print("No samples processed!")
