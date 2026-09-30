ORIG="$HOME/mugi_profiling/original_pipeline"
WORK="$HOME/orig_compare"
TIMES="$WORK/unit_times.csv"
mkdir -p "$WORK/runs"

unset PYTHONPATH
module load pytorch-conda/2.8 || { echo "pytorch-conda/2.8 not available"; exit 1; }
export PYTHONPATH="$ORIG/import_stubs${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/work/hdd/bebv/$USER/hf}"

[ -f "$TIMES" ] || echo "pipeline,job_id,unit,start,loop_start,end,rc,gpus" > "$TIMES"

python -c "import torch, transformers, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_unit() {
    local pipeline="$1" unit="$2"
    shift 2
    local log="$WORK/runs/${pipeline}_${unit}_${SLURM_JOB_ID:-local}.log"
    local start end loop_start rc
    start=$(date +%s)
    "$@" 2> >(tee -a "$log.err" >&2) | awk '{ print systime(), $0; fflush() }' | tee -a "$log"
    rc=${PIPESTATUS[0]}
    end=$(date +%s)
    loop_start=$(grep -m1 -E "Looping through configurations|Inferencing configurations" "$log" | cut -d' ' -f1)
    echo "$pipeline,${SLURM_JOB_ID:-local},$unit,$start,${loop_start:-},$end,$rc,${SLURM_GPUS_ON_NODE:-}" >> "$TIMES"
    return "$rc"
}
