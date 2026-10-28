# ONNX op summary: NousResearch/Llama-2-7b-hf

- Task `text-generation`, dtype `fp16`, attention `eager`, opset 17, IR 8
- 275 nodes, 12 initializers, 0.93 GB of weights (embedded)
- 1 decoder layers, all with the identical op pattern below
- Validated against PyTorch on 16 random tokens: max |logit diff| 7.81e-03 (tolerance 2.07e-01), next-token argmax agreement 100%

## Inputs

| Name | Dims | Dtype |
|---|---|---|
| `input_ids` | ['batch_size', 'sequence_length'] | int64 |
| `attention_mask` | ['batch_size', 'sequence_length'] | int64 |
| `position_ids` | ['batch_size', 'sequence_length'] | int64 |

## Outputs

| Name | Dims | Dtype |
|---|---|---|
| `logits` | ['batch_size', 'sequence_length', 32000] | float16 |

## Ops in one decoder layer, by submodule

| Submodule | Ops |
|---|---|
| `input_layernorm` | Constant 3, Cast 2, Mul 2, Pow 1, ReduceMean 1, Add 1, Sqrt 1, Div 1 |
| `self_attn` | Constant 37, Unsqueeze 15, MatMul 6, Concat 6, Cast 6, Shape 5, Gather 5, Transpose 5, Mul 5, Slice 5, Reshape 4, Add 3, Div 2, Neg 2, Softmax 1 |
| `residual` | Add 2 |
| `post_attention_layernorm` | Constant 3, Cast 2, Mul 2, Pow 1, ReduceMean 1, Add 1, Sqrt 1, Div 1 |
| `mlp` | MatMul 3, Mul 2, Sigmoid 1 |

## Matrix multiplies in one decoder layer

| Node | Kind | Weight shape (K×N) |
|---|---|---|
| `/model/layers.0/self_attn/q_proj/MatMul` | weight | 4096×4096 |
| `/model/layers.0/self_attn/k_proj/MatMul` | weight | 4096×4096 |
| `/model/layers.0/self_attn/v_proj/MatMul` | weight | 4096×4096 |
| `/model/layers.0/self_attn/MatMul` | activation x activation | – |
| `/model/layers.0/self_attn/MatMul_1` | activation x activation | – |
| `/model/layers.0/self_attn/o_proj/MatMul` | weight | 4096×4096 |
| `/model/layers.0/mlp/gate_proj/MatMul` | weight | 4096×11008 |
| `/model/layers.0/mlp/up_proj/MatMul` | weight | 4096×11008 |
| `/model/layers.0/mlp/down_proj/MatMul` | weight | 11008×4096 |

## Ops outside the decoder layers

Constant 50, Cast 14, Unsqueeze 14, Shape 8, Gather 7, Mul 7, Reshape 4, Concat 4, Range 3, ConstantOfShape 3, Where 3, And 2, Add 2, Equal 2, Expand 2, MatMul 2, LessOrEqual 1, Flatten 1, Transpose 1, Cos 1, Sin 1, Pow 1, ReduceMean 1, Sqrt 1, Div 1

Matrix multiplies outside the layers:

| Node | Kind | Weight shape (K×N) |
|---|---|---|
| `/model/rotary_emb/MatMul` | activation x activation | – |
| `/lm_head/MatMul` | weight | 4096×32000 |

## Op totals, whole graph

Constant 93, Unsqueeze 29, Cast 24, Mul 18, Shape 13, Gather 12, MatMul 11, Concat 10, Add 9, Reshape 8, Transpose 6, Div 5, Slice 5, Range 3, ConstantOfShape 3, Where 3, Pow 3, ReduceMean 3, Sqrt 3, And 2, Equal 2, Expand 2, Neg 2, LessOrEqual 1, Flatten 1, Cos 1, Sin 1, Softmax 1, Sigmoid 1
