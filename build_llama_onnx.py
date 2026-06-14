import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from list_llama_ops import load_config

OPSET = 17
EPS = 1e-5


class Builder:
    """Accumulates ONNX nodes + initializers and threads tensors by name."""

    def __init__(self, random_weights=False):
        self.nodes = []
        self.initializers = []
        self.random = random_weights
        self.param_count = 0      # total trainable weight elements (for size reporting)
        self._counter = 0
        self._init_names = set()

    # naming 
    def _name(self, hint):
        self._counter += 1
        return f"{hint}_{self._counter}"

    # initializers 
    def const(self, name, array):
        """Register a uniquely-named initializer from a numpy array."""
        name = self._name(name)
        self.initializers.append(numpy_helper.from_array(array.astype(np.float32), name))
        return name

    def weight(self, name, shape):
        self.param_count += int(np.prod(shape))
        if self.random:
            arr = (np.random.randn(*shape) * 0.02).astype(np.float32)
            return self.const(name, arr)
        shp = self.int_const(name + "_shape", list(shape))
        out = self._name(name)
        # ConstantOfShape defaults to a float32 zero fill
        self.nodes.append(helper.make_node("ConstantOfShape", [shp], [out]))
        return out

    def int_const(self, name, values):
        name = self._name(name)
        arr = np.asarray(values, dtype=np.int64)
        self.initializers.append(numpy_helper.from_array(arr, name))
        return name

    # node wrappers 
    def _op(self, op_type, inputs, out_hint, **attrs):
        out = self._name(out_hint)
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    def matmul(self, a, b):
        return self._op("MatMul", [a, b], "matmul")

    def add(self, a, b):
        return self._op("Add", [a, b], "add")

    def mul(self, a, b):
        return self._op("Mul", [a, b], "mul")

    def div(self, a, b):
        return self._op("Div", [a, b], "div")

    def sqrt(self, a):
        return self._op("Sqrt", [a], "sqrt")

    def sigmoid(self, a):
        return self._op("Sigmoid", [a], "sigmoid")

    def neg(self, a):
        return self._op("Neg", [a], "neg")

    def softmax(self, a, axis=-1):
        return self._op("Softmax", [a], "softmax", axis=axis)

    def gather(self, data, indices, axis=0):
        return self._op("Gather", [data, indices], "gather", axis=axis)

    def reducemean(self, a, axes):
        # opset-17 ReduceMean takes axes as an attribute
        return self._op("ReduceMean", [a], "rmean", axes=axes, keepdims=1)

    def pow_scalar(self, a, exponent):
        exp = self.const("exp", np.array(exponent, dtype=np.float32))
        return self._op("Pow", [a, exp], "pow")

    def reshape(self, a, shape):
        shp = self.int_const("shape", shape)
        return self._op("Reshape", [a, shp], "reshape")

    def transpose(self, a, perm):
        return self._op("Transpose", [a], "transpose", perm=perm)

    def concat(self, inputs, axis):
        return self._op("Concat", inputs, "concat", axis=axis)

    def split2(self, a, axis):
        """Split a tensor into two equal halves along `axis`."""
        o0 = self._name("split")
        o1 = self._name("split")
        # opset-17 Split with no `split` input + two outputs => two equal halves
        self.nodes.append(helper.make_node("Split", [a], [o0, o1], axis=axis))
        return o0, o1


# Composite ops
def rmsnorm(b: Builder, x, H):
    """RMSNorm over the last (width-H) dimension."""
    var = b.reducemean(b.pow_scalar(x, 2.0), axes=[-1])      # [..., 1]
    eps = b.const("eps", np.array(EPS, dtype=np.float32))
    denom = b.sqrt(b.add(var, eps))
    normed = b.div(x, denom)
    w = b.weight("rms_w", [H])
    return b.mul(normed, w)


def rope(b: Builder, x, cos_name, sin_name):
    """Apply rotary embedding to x[n_heads, S, head_dim] using cos/sin[S, head_dim]."""
    x1, x2 = b.split2(x, axis=-1)            # each [n_heads, S, hd/2]
    rotated = b.concat([b.neg(x2), x1], axis=-1)   # rotate_half(x)
    return b.add(b.mul(x, cos_name), b.mul(rotated, sin_name))


def attention(b: Builder, x, cfg, S):
    H = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hd = H // n_heads
    kv_dim = n_kv * hd

    # projections: [S, H] @ [H, *] -> [S, *]
    q = b.matmul(x, b.weight("q_proj", [H, H]))
    k = b.matmul(x, b.weight("k_proj", [H, kv_dim]))
    v = b.matmul(x, b.weight("v_proj", [H, kv_dim]))

    # reshape to heads:  [S, n*hd] -> [n, S, hd]
    def to_heads(t, n):
        t = b.reshape(t, [S, n, hd])
        return b.transpose(t, perm=[1, 0, 2])

    q = to_heads(q, n_heads)
    k = to_heads(k, n_kv)
    v = to_heads(v, n_kv)

    # RoPE on Q and K (cos/sin broadcast over the head dim)
    cos = b.const("cos", np.cos(_rope_freqs(S, hd))[None, :, :])   # [1, S, hd]
    sin = b.const("sin", np.sin(_rope_freqs(S, hd))[None, :, :])
    q = rope(b, q, cos, sin)
    k = rope(b, k, cos, sin)

    # GQA: expand K/V from n_kv heads to n_heads if needed
    if n_kv != n_heads:
        rep = n_heads // n_kv
        k = _repeat_kv(b, k, n_kv, rep, S, hd)
        v = _repeat_kv(b, v, n_kv, rep, S, hd)

    # scores = Q @ K^T   -> [n_heads, S, S]
    kt = b.transpose(k, perm=[0, 2, 1])
    scores = b.matmul(q, kt)

    # scale + causal mask
    scale = b.const("scale", np.array(1.0 / np.sqrt(hd), dtype=np.float32))
    scores = b.mul(scores, scale)
    mask = np.triu(np.full((S, S), -1e9, dtype=np.float32), k=1)[None, :, :]  # [1,S,S]
    scores = b.add(scores, b.const("mask", mask))

    probs = b.softmax(scores, axis=-1)

    # context = probs @ V  -> [n_heads, S, hd] -> [S, H]
    ctx = b.matmul(probs, v)
    ctx = b.transpose(ctx, perm=[1, 0, 2])     # [S, n_heads, hd]
    ctx = b.reshape(ctx, [S, H])

    # output projection
    return b.matmul(ctx, b.weight("o_proj", [H, H]))


def _repeat_kv(b: Builder, t, n_kv, rep, S, hd):
    """Expand [n_kv, S, hd] -> [n_kv*rep, S, hd] by repeating each kv head `rep` times."""
    t = b.reshape(t, [n_kv, 1, S, hd])
    expanded = b._op("Expand", [t, b.int_const("exp_shape", [n_kv, rep, S, hd])], "expand")
    return b.reshape(expanded, [n_kv * rep, S, hd])


def mlp(b: Builder, x, cfg, S):
    H = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    gate = b.matmul(x, b.weight("gate_proj", [H, inter]))
    up = b.matmul(x, b.weight("up_proj", [H, inter]))
    silu = b.mul(gate, b.sigmoid(gate))     # SiLU(gate) = gate * sigmoid(gate)
    h = b.mul(silu, up)
    return b.matmul(h, b.weight("down_proj", [inter, H]))


def build_decoder_block_onnx(b: Builder, x, cfg, S):
    """Emit one decoder block; mirrors build_decoder_block in list_llama_ops.py."""
    H = cfg["hidden_size"]
    # attention sub-layer with residual
    a = rmsnorm(b, x, H)
    a = attention(b, a, cfg, S)
    x = b.add(x, a)
    # feed-forward sub-layer with residual
    m = rmsnorm(b, x, H)
    m = mlp(b, m, cfg, S)
    x = b.add(x, m)
    return x


def _rope_freqs(S, hd):
    """Standard RoPE angle table -> [S, hd] (each frequency duplicated across the pair)."""
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))  # [hd/2]
    pos = np.arange(S, dtype=np.float32)[:, None]            # [S, 1]
    angles = pos * inv_freq[None, :]                         # [S, hd/2]
    return np.concatenate([angles, angles], axis=-1)        # [S, hd]


# Full / block model assembly
def build_model(cfg, S, full, random_weights):
    b = Builder(random_weights=random_weights)
    H = cfg["hidden_size"]

    if full:
        ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [S])
        embed_w = b.weight("embed_tokens", [cfg["vocab_size"], H])
        x = b.gather(embed_w, "input_ids", axis=0)          # [S, H]
        for _ in range(cfg["num_hidden_layers"]):
            x = build_decoder_block_onnx(b, x, cfg, S)
        x = rmsnorm(b, x, H)                                 # final norm
        logits = b.matmul(x, b.weight("lm_head", [H, cfg["vocab_size"]]))
        out_name, out_dims = logits, [S, cfg["vocab_size"]]
        inputs = [ids]
    else:
        hid = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [S, H])
        x = build_decoder_block_onnx(b, "hidden", cfg, S)
        out_name, out_dims = x, [S, H]
        inputs = [hid]

    output = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_dims)
    graph = helper.make_graph(b.nodes, "llama", inputs, [output], b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    return model, out_name, out_dims, b.param_count


def main():
    parser = argparse.ArgumentParser(description="Build an ONNX graph from the Llama op list.")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="HuggingFace model id")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="sequence length (use a small value like 128 for quick runs)")
    parser.add_argument("--full", action="store_true",
                        help="build the whole model (embed -> N blocks -> norm -> lm_head)")
    parser.add_argument("--random", action="store_true",
                        help="use small random weight initializers instead of ConstantOfShape zeros "
                             "(only practical for a single block)")
    parser.add_argument("--out", default="llama.onnx", help="output .onnx path")
    parser.add_argument("--no-run", action="store_true", help="skip the onnxruntime sanity run")
    parser.add_argument("--run", action="store_true",
                        help="force the onnxruntime run even for --full (needs ~param-size RAM)")
    args = parser.parse_args()

    if args.random and args.full:
        print("warning: --random with --full materializes the full ~7B weights "
              "(tens of GB, exceeds the 2 GB protobuf limit); consider dropping --random")

    cfg, source = load_config(args.model)
    print(f"config: {source}")
    print(f"hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']} "
          f"heads={cfg['num_attention_heads']} kv_heads={cfg['num_key_value_heads']} "
          f"intermediate={cfg['intermediate_size']} vocab={cfg['vocab_size']}")
    scope = "full model" if args.full else "one decoder block"
    print(f"building {scope}, seq_len={args.seq_len}, weights={'random' if args.random else 'zeros'}")

    model, out_name, out_dims, param_count = build_model(cfg, args.seq_len, args.full, args.random)

    onnx.checker.check_model(model)
    print("onnx.checker: passed")

    onnx.save(model, args.out)
    n_nodes = len(model.graph.node)
    print(f"saved {args.out}  ({n_nodes} nodes, {len(model.graph.initializer)} initializers)")
    print(f"model weights: {param_count/1e9:.2f}B params (~{param_count*4/1e9:.1f} GB float32 at runtime)")

    do_run = not args.no_run
    if args.full and not args.run:
        do_run = False
        print(f"note: skipping onnxruntime run for --full "
              f"(would need ~{param_count*4/1e9:.0f} GB RAM to execute); pass --run to force, "
              f"or inspect structure in Netron")

    if do_run:
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed; skipping sanity run")
            return
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        if args.full:
            feed = {"input_ids": np.random.randint(0, cfg["vocab_size"], size=(args.seq_len,)).astype(np.int64)}
        else:
            feed = {"hidden": np.random.randn(args.seq_len, cfg["hidden_size"]).astype(np.float32)}
        result = sess.run([out_name], feed)[0]
        print(f"onnxruntime run: output '{out_name}' shape {list(result.shape)} (expected {out_dims})")


if __name__ == "__main__":
    main()
