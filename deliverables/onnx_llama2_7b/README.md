# Llama-2-7B ONNX graph (prefill, fp16, opset 17)

The model operation graph for Llama-2-7B as ONNX, as input for the LLVM → RISC-V / array compilation work.
It's the standard HuggingFace model exported as-is: it does **not** contain the VLP or LUT approximations.
Softmax, SiLU and RMSNorm appear as ordinary ONNX ops, which are the ops the hardware path is meant to replace.

## What's in this folder

| File | What it is |
|---|---|
| `model_graph_only.onnx` | The full 32-layer graph: every node, shape and attribute, without the weights. Opens in [Netron](https://netron.app). It points at a `model.onnx_data` weights file that isn't in the repo, so it can't be *run* on its own. |
| `op_summary.md` | Readable summary: inputs and outputs, ops per decoder layer grouped by submodule, every matrix multiply with its weight shape, ops outside the layers, op totals, and the PyTorch-vs-ONNX validation result. |
| `op_summary.json` | The same, machine-readable. |
| `op_summary_1layer.md` | The same summary for the 1-layer model (see below). |
| `config.json` | The HuggingFace model config the graph was exported from. |

## The model

- **Source:** `NousResearch/Llama-2-7b-hf` (an ungated copy of `meta-llama/Llama-2-7b-hf`).
- **Form:** prefill only. One forward pass over the whole input; no KV cache inputs or outputs.
- **Inputs:** `input_ids`, `attention_mask` and `position_ids`, each `int64 [batch_size, sequence_length]` (both dims dynamic).
- **Output:** `logits`, `fp16 [batch_size, sequence_length, 32000]`.
- **Weights:** fp16, about 13.5 GB.
- **ONNX:** opset 17. Optimum recommends 18 for Llama; the export validates at 17, and you can regenerate at 18 with `--opset 18`.
- **Attention:** exported from the *eager* attention path, so every step is an explicit op, not a fused kernel.
- **Dimensions:** hidden 4096, 32 heads (head dim 128, no grouped-query attention), MLP width 11008, 32 decoder layers, vocab 32000.

### One decoder layer, in order

1. **RMSNorm** (`input_layernorm`), computed in fp32: Cast(fp32) → Pow → ReduceMean → Add(ε) → Sqrt → Div (1/√·) → Mul(x) → Cast(fp16) → Mul(weight).
2. **Q/K/V projections:** 3 MatMuls against 4096×4096 weights.
3. **RoPE** on Q and K: Mul, Slice, Neg, Concat, Add. The cos/sin tables are computed once, outside the layers (`rotary_emb`: MatMul, Cos, Sin).
4. **Scores:** MatMul(Q, Kᵀ), scaled, then Add(causal mask).
5. **Softmax** over the key dimension. In the exported fp16 graph it runs directly on fp16 inputs. PyTorch's `softmax(..., dtype=float32)` only survives as a Cast to fp32 and straight back to fp16 *after* the Softmax node.
6. **Context:** MatMul(probabilities, V), then the **output projection** (MatMul, 4096×4096) and the residual Add.
7. **RMSNorm** (`post_attention_layernorm`).
8. **MLP:** gate and up MatMuls (4096×11008), **SiLU** = Sigmoid then Mul, an elementwise Mul of gate and up, the down MatMul (11008×4096), and the residual Add.

Outside the layers:
- the embedding lookup (Gather);
- causal-mask construction from `attention_mask` (Range, LessOrEqual, Where, Expand, …);
- the final RMSNorm;
- `lm_head` (MatMul, 4096×32000).

The exact counts are in `op_summary.md`. Layer 0 carries 2 extra Constant and 2 extra Unsqueeze nodes, which are shape bookkeeping for the shared RoPE tables; the summary lists this explicitly.

## Getting the full model (with weights)

The weights are too large for the repo (GitHub's limit is 100 MB per file). There are two ways to get them.

**1. Copy from Delta**, if you're on the `bebv` allocation:

```
/work/hdd/bebv/sborges/onnx/llama-2-7b-prefill-fp16/        full model: model.onnx + model.onnx_data (~13.5 GB)
/work/hdd/bebv/sborges/onnx/llama-2-7b-prefill-fp16-1layer/ embedding + 1 decoder layer + lm_head, real weights (~1 GB)
```

The 1-layer model has the same per-layer graph as the full one and is the easier thing to start compiling.

**2. Regenerate it yourself** from this repo. You need a GPU with at least 24 GB, or a CPU machine with at least 64 GB of RAM, plus about 30 GB of free disk (13 GB to download the model, 13.5 GB for the export).

```bash
pip install "optimum-onnx>=0.1.0" "transformers==4.56.1" onnxruntime
python export_onnx.py --model NousResearch/Llama-2-7b-hf --dtype fp16 --device cuda --validate --out llama-2-7b-prefill-fp16
python export_onnx.py --model NousResearch/Llama-2-7b-hf --dtype fp16 --device cuda --validate --layers 1 --out llama-2-7b-prefill-fp16-1layer
```

On a CPU-only machine, use `--device cpu`. `--validate` runs 16 random tokens through both PyTorch and ONNX Runtime and stops with an error if the logits disagree. Each export writes its own `op_summary.md`.

On Delta, `sbatch slurm_scripts/export_onnx.sh` does all of the above. It also creates its own virtual environment, so the shared `pytorch-conda/2.8` module isn't modified.
