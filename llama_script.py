from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from itertools import islice
import torch
import math
import inspect
from tqdm import tqdm

from custom_nonlinear_functions import vlp_silu_approx, vlp_softmax_approx, pwl_softmax_approx, pwl_silu_approx, pwl_mobilenet_approx, taylor_softmax_approx

# ----- Config -----
torch.set_printoptions(linewidth=200)
model_name = "meta-llama/Llama-2-7b-hf"
split = "validation"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 1024
batch_size = 1
softmax_exp_dim = 16
softmax_max_exp = 3
softmax_min_exp = -4
silu_exp_dim = 12
silu_max_exp = 1
window_size = 32
n_samples = 1
segments = 20
segment_0 = 4
degree_center = -6
degrees = 13
vlp_build = 'max'
save_dims = [1024, 2048, 4096]

# ----- Load custom nonlinear functions -----
# softmax = vlp_softmax_approx.VLPSoftmax(exp_dim=softmax_exp_dim, max_exp=softmax_max_exp, min_exp=softmax_min_exp, window_size=window_size, lut_build=vlp_build, device=device, path='profile/vlp/softmax/', save_dims=save_dims, profile=True)
# silu = vlp_silu_approx.VLPSilu(exp_dim=silu_exp_dim, max_pos_exp=silu_max_exp, max_neg_exp=silu_max_exp, window_size=window_size, device=device)
softmax = pwl_softmax_approx.PWLSoftmax(segments=segments, segment_0=segment_0, device=device, path='profile/pwl/softmax/', save_dims=save_dims, profile=True)
# silu = pwl_silu_approx.PWLSilu(segments=segments, segment_0=segment_0, device=device, path='profile/pwl/silu/', save_dims=save_dims, profile=False)
# silu = pwl_mobilenet_approx.PWLMobilenet(device=device, path='profile/pwl/mobilenet/', save_dims=save_dims, profile=False)
# softmax = taylor_softmax_approx.TaylorSoftmax(degree_center=degree_center, degrees=degrees, device=device, path='profile/taylor/softmax/', save_dims=save_dims, profile=False)

# ----- Patch Softmax Function -----
torch.nn.functional.softmax = softmax.forward

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float16,
                                             attn_implementation='eager'
                                             ).to(device)

# for layer in model.model.layers:
#     layer.mlp.act_fn = silu

model.eval()

# ----- Load dataset -----
dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)

def get_token_length(example):
    tokens = tokenizer(example["text"], truncation=False, return_tensors=None)
    return len(tokens["input_ids"])

subset = []
for example in tqdm(dataset, desc="Preprocessing dataset"):
    if len(subset) >= n_samples:
        break
    if len(example["text"]) > max_length:
        if get_token_length(example) >= max_length:
            subset.append(example)

# ----- Preprocessing function -----
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def get_token_length(example):
    tokens = tokenizer(example["text"], truncation=False, return_tensors=None)
    return len(tokens["input_ids"])

tokenized_subset = [tokenize(example) for example in subset]

# ----- Perplexity calculation -----
def compute_loss(input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        return outputs.loss

# ----- Batch processing -----
total_loss = 0.0
num_batches = 0
batch_loss = []

for i in tqdm(range(0, len(tokenized_subset), batch_size), desc="Evaluating"):
    batch = tokenized_subset[i:i+batch_size]

    batch_max_len = max(ex["input_ids"].shape[1] for ex in batch)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [ex["input_ids"].squeeze(0)[:batch_max_len] for ex in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    loss = compute_loss(input_ids, attention_mask)

    total_loss += loss.item()
    num_batches += 1
    batch_loss.append(loss.item())

    del input_ids, attention_mask
    torch.cuda.empty_cache()

# ----- Calculate perplexity -----
average_loss = total_loss / num_batches
perplexity = math.exp(average_loss)
print(f"\nPerplexity: {perplexity:.2f}")
