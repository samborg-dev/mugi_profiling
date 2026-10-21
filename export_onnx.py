import argparse
import json
import os
import re
import shutil
from collections import Counter, OrderedDict

OPSET = 17
DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
LAYER = re.compile(r"layers\.(\d+)/([^/]+)")


def export_to_onnx(model_id, out_dir, task="auto", opset=None, dtype="fp32", device="cpu",
                   attn="eager", layers=None, validate=False, summarize=True):
    import torch
    from optimum.exporters.onnx import onnx_export_from_model
    from optimum.exporters.tasks import TasksManager

    if task == "auto":
        task = TasksManager.infer_task_from_model(model_id)
    model_kwargs = {"attn_implementation": attn}
    if layers:
        model_kwargs["num_hidden_layers"] = layers
    model = TasksManager.get_model_from_task(task, model_id, torch_dtype=getattr(torch, DTYPES[dtype]),
                                             device=device, **model_kwargs)
    model.eval()
    onnx_export_from_model(model, out_dir, opset=opset or OPSET, task=task, device=device,
                           do_validation=False)
    meta = {"model": model_id, "task": task, "dtype": dtype, "attn": attn,
            "layers": layers or getattr(model.config, "num_hidden_layers", None)}
    if validate:
        meta["validation"] = validate_against_torch(model, out_dir, dtype)
    if summarize:
        summarize_export(out_dir, meta)
    return out_dir


def validate_against_torch(model, out_dir, dtype, seq_len=16, seed=0):
    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(os.path.join(out_dir, "model.onnx"), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    rng = np.random.default_rng(seed)
    feeds = {"input_ids": rng.integers(0, model.config.vocab_size, size=(1, seq_len), dtype=np.int64),
             "attention_mask": np.ones((1, seq_len), dtype=np.int64),
             "position_ids": np.arange(seq_len, dtype=np.int64)[None, :]}
    feeds = {k: v for k, v in feeds.items() if k in names}
    missing = set(names) - set(feeds)
    if missing:
        raise RuntimeError(f"validation does not know how to feed inputs {sorted(missing)}")

    device = next(model.parameters()).device
    with torch.no_grad():
        ref = model(**{k: torch.from_numpy(v).to(device) for k, v in feeds.items()}).logits
    ref = ref.float().cpu().numpy()
    got = session.run(["logits"], feeds)[0].astype(np.float32)

    max_abs = float(np.abs(got - ref).max())
    scale = max(1.0, float(np.abs(ref).max()))
    tol = {"fp32": 1e-3, "fp16": 2e-2, "bf16": 5e-2}[dtype] * scale
    result = {"seq_len": seq_len, "max_abs_diff": max_abs, "tolerance": tol,
              "ref_max_abs": float(np.abs(ref).max()),
              "argmax_agreement": float((got.argmax(-1) == ref.argmax(-1)).mean()),
              "passed": max_abs <= tol}
    print(f"validation: {result}")
    if not result["passed"]:
        raise RuntimeError(f"ONNX output differs from PyTorch: max abs diff {max_abs:.3g} > {tol:.3g}")
    return result


def _dims(value_info):
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


def _tensor_bytes(tensor):
    import numpy as np
    from onnx import helper
    n = int(np.prod(tensor.dims)) if tensor.dims else 1
    return n * np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize


def summarize_export(out_dir, meta=None):
    import onnx
    from onnx import helper

    path = os.path.join(out_dir, "model.onnx")
    onnx.checker.check_model(path)
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    external = any(t.data_location == onnx.TensorProto.EXTERNAL for t in inits.values())

    ops = Counter(n.op_type for n in graph.node)
    per_layer = {}
    outside = Counter()
    gemms = []
    for node in graph.node:
        match = LAYER.search(node.name)
        if match:
            layer, module = int(match.group(1)), match.group(2)
            if node.name.endswith("/" + module):
                module = "residual"
            per_layer.setdefault(layer, OrderedDict()).setdefault(module, Counter())[node.op_type] += 1
        else:
            outside[node.op_type] += 1
        if node.op_type in ("MatMul", "Gemm"):
            weight = next((inits[i] for i in node.input if i in inits), None)
            gemms.append({
                "node": node.name,
                "layer": int(match.group(1)) if match else None,
                "kind": "weight" if weight is not None else "activation x activation",
                "weight_shape": list(weight.dims) if weight is not None else None,
            })

    layer_ids = sorted(per_layer)
    flat = {i: frozenset((m, op, n) for m, c in per_layer[i].items() for op, n in c.items())
            for i in layer_ids}
    common = Counter(flat.values()).most_common(1)[0][0] if layer_ids else frozenset()
    rep = next((i for i in layer_ids if flat[i] == common), None)
    first = per_layer[rep] if rep is not None else {}
    deviations = {}
    for i in layer_ids:
        if flat[i] != common:
            ref = {(m, op): n for m, op, n in common}
            mine = {(m, op): n for m, op, n in flat[i]}
            deviations[i] = {f"{m}/{op}": mine.get((m, op), 0) - ref.get((m, op), 0)
                             for m, op in set(ref) | set(mine) if mine.get((m, op), 0) != ref.get((m, op), 0)}
    same = not deviations
    summary = {
        **(meta or {}),
        "opset": next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        "ir_version": model.ir_version,
        "inputs": {i.name: [_dims(i), helper.tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type).name]
                   for i in graph.input if i.name not in inits},
        "outputs": {o.name: [_dims(o), helper.tensor_dtype_to_np_dtype(o.type.tensor_type.elem_type).name]
                    for o in graph.output},
        "nodes": len(graph.node),
        "initializers": len(inits),
        "weight_bytes": sum(_tensor_bytes(t) for t in inits.values()),
        "external_data": external,
        "op_counts": dict(ops.most_common()),
        "decoder_layers": len(layer_ids),
        "layers_identical": same,
        "representative_layer": rep,
        "layer_deviations": deviations,
        "per_layer_ops": {m: dict(c.most_common()) for m, c in first.items()},
        "outside_layer_ops": dict(outside.most_common()),
        "gemms_per_layer": [g for g in gemms if rep is not None and g["layer"] == rep],
        "gemms_outside_layers": [g for g in gemms if g["layer"] is None],
    }
    with open(os.path.join(out_dir, "op_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "op_summary.md"), "w") as f:
        f.write(_markdown(summary))
    if external:
        shutil.copyfile(path, os.path.join(out_dir, "model_graph_only.onnx"))
    return summary


def _validation_line(v):
    if not v:
        return "- Not validated against PyTorch in this export"
    return (f"- Validated against PyTorch on {v['seq_len']} random tokens: max |logit diff| "
            f"{v['max_abs_diff']:.2e} (tolerance {v['tolerance']:.2e}), "
            f"next-token argmax agreement {v['argmax_agreement']:.0%}")


def _markdown(s):
    def io(rows):
        return "\n".join(f"| `{k}` | {v[0]} | {v[1]} |" for k, v in rows.items())

    def counts(c):
        return ", ".join(f"{op} {n}" for op, n in c.items())

    def gemm_rows(gs):
        return "\n".join(
            f"| `{g['node']}` | {g['kind']} | {'×'.join(map(str, g['weight_shape'])) if g['weight_shape'] else '–'} |"
            for g in gs)

    lines = [
        f"# ONNX op summary: {s.get('model', '')}",
        "",
        f"- Task `{s.get('task')}`, dtype `{s.get('dtype')}`, attention `{s.get('attn')}`, "
        f"opset {s['opset']}, IR {s['ir_version']}",
        f"- {s['nodes']} nodes, {s['initializers']} initializers, "
        f"{s['weight_bytes'] / 1e9:.2f} GB of weights"
        + (" (external data file)" if s["external_data"] else " (embedded)"),
        f"- {s['decoder_layers']} decoder layers"
        + (", all with the identical op pattern below" if s["layers_identical"] else
           f", op pattern below taken from layer {s['representative_layer']}; differences: "
           + "; ".join(f"layer {i}: " + ", ".join(f"{k} {v:+d}" for k, v in sorted(d.items()))
                       for i, d in s["layer_deviations"].items())),
        _validation_line(s.get("validation")),
        "",
        "## Inputs",
        "",
        "| Name | Dims | Dtype |",
        "|---|---|---|",
        io(s["inputs"]),
        "",
        "## Outputs",
        "",
        "| Name | Dims | Dtype |",
        "|---|---|---|",
        io(s["outputs"]),
        "",
        "## Ops in one decoder layer, by submodule",
        "",
        "| Submodule | Ops |",
        "|---|---|",
        "\n".join(f"| `{m}` | {counts(c)} |" for m, c in s["per_layer_ops"].items()),
        "",
        "## Matrix multiplies in one decoder layer",
        "",
        "| Node | Kind | Weight shape (K×N) |",
        "|---|---|---|",
        gemm_rows(s["gemms_per_layer"]),
        "",
        "## Ops outside the decoder layers",
        "",
        counts(s["outside_layer_ops"]),
        "",
        "Matrix multiplies outside the layers:",
        "",
        "| Node | Kind | Weight shape (K×N) |",
        "|---|---|---|",
        gemm_rows(s["gemms_outside_layers"]),
        "",
        "## Op totals, whole graph",
        "",
        counts(s["op_counts"]),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export a HuggingFace model to ONNX and summarize its ops.")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--out", default=None, help="output dir (default: output/onnx/<model-name>)")
    parser.add_argument("--task", default="auto",
                        help="export task; auto infers it without a KV cache (e.g. text-generation)")
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attn", default="eager", help="attention implementation to load the model with")
    parser.add_argument("--layers", type=int, default=None, help="keep only the first N decoder layers")
    parser.add_argument("--validate", action="store_true",
                        help="compare ONNX Runtime outputs against PyTorch after export")
    parser.add_argument("--summarize_only", action="store_true",
                        help="skip the export and only re-summarize an existing --out dir")
    args = parser.parse_args()

    out = args.out or os.path.join("output", "onnx", args.model.rstrip("/").split("/")[-1])
    if args.summarize_only:
        summarize_export(out, {"model": args.model})
    else:
        export_to_onnx(args.model, out, task=args.task, opset=args.opset, dtype=args.dtype,
                       device=args.device, attn=args.attn, layers=args.layers, validate=args.validate)
    print(f"exported {args.model} -> {out}")
    print(open(os.path.join(out, "op_summary.md")).read())


if __name__ == "__main__":
    main()
