import argparse
import os


def export_to_onnx(model_id, out_dir, task="auto", opset=None, **kwargs):
    """Export any HuggingFace model to ONNX via Optimum.

    Args:
        model_id: HF hub id or local path.
        out_dir:  directory to write the ONNX (and its config) into.
        task:     export task; "auto" lets Optimum infer it from the model.
        opset:    ONNX opset; None uses Optimum's default. NOTE: confirm the opset the Mugi
                  backend parses with the team before locking this.
        **kwargs: forwarded to optimum.exporters.onnx.main_export (e.g. device, dtype).

    Returns:
        out_dir (now containing model.onnx + config files).
    """
    # Imported lazily so `--help` and import work without the (heavy) optimum/torch stack.
    from optimum.exporters.onnx import main_export

    main_export(model_id, output=out_dir, task=task, opset=opset, **kwargs)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Export any HF model to ONNX (model-agnostic).")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--out", default=None,
                        help="output dir (default: output/onnx/<model-name>, gitignored)")
    parser.add_argument("--task", default="auto", help="export task (default: auto-infer)")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset (default: Optimum's)")
    args = parser.parse_args()

    out = args.out or os.path.join("output", "onnx", args.model.rstrip("/").split("/")[-1])
    out = export_to_onnx(args.model, out, task=args.task, opset=args.opset)
    print(f"exported {args.model} -> {out}")


if __name__ == "__main__":
    main()
