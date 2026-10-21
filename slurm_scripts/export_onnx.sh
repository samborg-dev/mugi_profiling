#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --job-name=export_onnx
#SBATCH --error=export_onnx_%j.err
#SBATCH --output=export_onnx_%j.out

set -eu

cd ~/mugi_profiling

unset PYTHONPATH
module load pytorch-conda/2.8 || { echo "pytorch-conda/2.8 not available"; exit 1; }
export HF_HOME="${HF_HOME:-/work/hdd/bebv/$USER/hf}"

VENV="${VENV:-$HOME/venvs/onnx}"
if [ ! -x "$VENV/bin/python" ]; then
    TF_VERSION=$(python -c "import transformers; print(transformers.__version__)")
    python -m venv "$VENV" --system-site-packages
    "$VENV/bin/pip" install --quiet "optimum-onnx>=0.1.0" "transformers==$TF_VERSION"
fi
PY="$VENV/bin/python"
"$PY" -c "import torch, transformers, optimum, onnx, onnxruntime; print('torch', torch.__version__, 'transformers', transformers.__version__, 'onnx', onnx.__version__, 'onnxruntime', onnxruntime.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL="${MODEL:-NousResearch/Llama-2-7b-hf}"
ROOT="${ROOT:-/work/hdd/bebv/$USER/onnx}"
NAME="${NAME:-llama-2-7b-prefill}"
DELIVER="${DELIVER:-deliverables/onnx_llama2_7b}"
mkdir -p "$ROOT" "$DELIVER"

run() {
    echo "=== $*"
    "$PY" -u export_onnx.py "$@"
}

run --model hf-internal-testing/tiny-random-LlamaForCausalLM --out "$ROOT/tiny-smoke" --validate
run --model "$MODEL" --out "$ROOT/$NAME-fp32-1layer" --layers 1 --dtype fp32 --device cuda --validate
run --model "$MODEL" --out "$ROOT/$NAME-fp16-1layer" --layers 1 --dtype fp16 --device cuda --validate
run --model "$MODEL" --out "$ROOT/$NAME-fp16" --dtype fp16 --device cuda --validate

chmod -R g+rX "$ROOT/$NAME-fp16" "$ROOT/$NAME-fp16-1layer" "$ROOT/$NAME-fp32-1layer"

cp "$ROOT/$NAME-fp16/model_graph_only.onnx" "$DELIVER/"
cp "$ROOT/$NAME-fp16/op_summary.md" "$DELIVER/op_summary.md"
cp "$ROOT/$NAME-fp16/op_summary.json" "$DELIVER/op_summary.json"
cp "$ROOT/$NAME-fp16/config.json" "$DELIVER/config.json"
cp "$ROOT/$NAME-fp16-1layer/op_summary.md" "$DELIVER/op_summary_1layer.md"

echo "full model:   $ROOT/$NAME-fp16"
echo "1-layer fp16: $ROOT/$NAME-fp16-1layer"
du -sh "$ROOT/$NAME-fp16" "$ROOT/$NAME-fp16-1layer" "$ROOT/$NAME-fp32-1layer"
ls -l "$DELIVER"
echo "exit code: 0"
