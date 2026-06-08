import argparse
from dataclasses import dataclass


# Known Llama-2-7B settings, used as a fallback if we can't read the real config.
FALLBACK_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,   # 7B has no GQA; bigger Llama-2 models do (e.g. 70B uses 8)
    "intermediate_size": 11008,
    "vocab_size": 32000,
}


@dataclass
class Op:
    """An operation performed by the model, for tallying and display purposes"""
    name: str
    bucket: str          # gemm | lut | vector | special | lookup
    detail: str = ""     # human-readable note, e.g. the matrix-multiply size
    repeat: int = 1      # how many times this op runs in one pass


def gemm(name, M, K, N, repeat=1, note=""):
    """Helper to describe a matrix multiply of shape (M x K) times (K x N)"""
    size = f"({M} x {K}) @ ({K} x {N})"
    if repeat > 1:
        size += f"   x{repeat} (one per attention head)"
    if note:
        size += f"   [{note}]"
    return Op(name, "gemm", size, repeat)


def load_config(model_name):
    """Read the model's config from HuggingFace; fall back to Llama-2-7B numbers"""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        return {
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            # older configs may not set num_key_value_heads -> defaults to num heads
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
        }, "read from HuggingFace config"
    except Exception as exc:  # no internet, not logged in, etc.
        return dict(FALLBACK_CONFIG), f"FALLBACK to hardcoded Llama-2-7B numbers ({type(exc).__name__})"


def build_decoder_block(cfg, seq_len):
    """Return the ordered op list for ONE decoder block (this repeats N times)"""
    H = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = H // n_heads
    kv_dim = n_kv * head_dim          # K/V projection output width (smaller than H if GQA)
    inter = cfg["intermediate_size"]
    S = seq_len

    ops = []

    # ---- attention ----
    ops.append(Op("input_layernorm (RMSNorm)", "vector", "rescale the 4096-wide vectors"))
    ops.append(gemm("q_proj  (Query)", S, H, H))
    ops.append(gemm("k_proj  (Key)",   S, H, kv_dim, note="narrower than Q if model uses GQA"))
    ops.append(gemm("v_proj  (Value)", S, H, kv_dim, note="narrower than Q if model uses GQA"))
    ops.append(Op("RoPE (rotary position)", "special", "rotates Q and K to encode word position"))
    ops.append(gemm("Q x K^T  (relevance scores)", S, head_dim, S, repeat=n_heads))
    ops.append(Op("scale + causal mask", "vector", "divide by sqrt(head_dim); hide future words"))
    ops.append(Op("softmax", "lut", "turn scores into percentages that sum to 100%"))
    ops.append(gemm("scores x V  (blend values)", S, S, head_dim, repeat=n_heads))
    ops.append(gemm("o_proj  (output projection)", S, H, H))
    ops.append(Op("residual add", "vector", "add attention result back to block input"))

    # ---- feed-forward (MLP) ----
    ops.append(Op("post_attention_layernorm (RMSNorm)", "vector", "rescale again"))
    ops.append(gemm("gate_proj", S, H, inter))
    ops.append(gemm("up_proj",   S, H, inter))
    ops.append(Op("SiLU(gate) * up", "lut", "smooth on/off curve, then multiply"))
    ops.append(gemm("down_proj", S, inter, H))
    ops.append(Op("residual add", "vector", "add MLP result back to block input"))

    return ops


def build_full_model(cfg, seq_len):
    """Return (setup_ops, one_block_ops, finish_ops, num_layers)"""
    H = cfg["hidden_size"]
    S = seq_len
    setup = [Op("embed_tokens", "lookup", f"look up {S} token IDs -> {S} vectors of width {H}")]
    block = build_decoder_block(cfg, seq_len)
    finish = [
        Op("final norm (RMSNorm)", "vector", "one last rescale"),
        gemm("lm_head  (next-word scores)", S, H, cfg["vocab_size"],
             note="score every possible next word"),
    ]
    return setup, block, finish, cfg["num_hidden_layers"]


def print_ops(title, ops):
    print(title)
    for op in ops:
        tag = f"[{op.bucket}]".ljust(10)
        line = f"   {tag} {op.name}"
        if op.detail:
            line = line.ljust(48) + f" -- {op.detail}"
        print(line)
    print()


def main():
    parser = argparse.ArgumentParser(description="List the operations a Llama-2 model performs.")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="HuggingFace model id")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length (number of tokens)")
    args = parser.parse_args()

    cfg, source = load_config(args.model)

    print("=" * 78)
    print(f"  Operation list for: {args.model}")
    print(f"  Config source     : {source}")
    print(f"  Sequence length   : {args.seq_len} tokens   (batch size 1)")
    print("=" * 78)
    print(f"  hidden={cfg['hidden_size']}  layers={cfg['num_hidden_layers']}  "
          f"heads={cfg['num_attention_heads']}  kv_heads={cfg['num_key_value_heads']}  "
          f"intermediate={cfg['intermediate_size']}  vocab={cfg['vocab_size']}")
    print()

    setup, block, finish, n_layers = build_full_model(cfg, args.seq_len)

    print_ops("SETUP (runs once):", setup)
    print_ops(f"ONE DECODER BLOCK  (this entire block repeats {n_layers} times):", block)
    print_ops("FINISH (runs once):", finish)

    # ---- bucket tally across the whole forward pass ----
    buckets = {}
    def add(ops, times):
        for op in ops:
            buckets[op.bucket] = buckets.get(op.bucket, 0) + op.repeat * times

    add(setup, 1)
    add(block, n_layers)
    add(finish, 1)

    print("=" * 78)
    print(f"  TOTAL operation tally for one forward pass ({n_layers} blocks):")
    print("=" * 78)
    labels = {
        "gemm": "matrix multiplies (GEMMs)",
        "lut": "lookup-table steps",
        "vector": "light vector steps",
        "special": "special case",
        "lookup": "embedding lookup",
    }
    for bucket in ["gemm", "lut", "vector", "special", "lookup"]:
        if bucket in buckets:
            print(f"   {buckets[bucket]:>6}   {labels[bucket]}")
    print()


if __name__ == "__main__":
    main()
