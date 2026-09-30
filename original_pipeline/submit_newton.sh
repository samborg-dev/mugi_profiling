#!/bin/bash
set -euo pipefail

ORIG="$HOME/mugi_profiling/original_pipeline"
LOGS="$HOME/orig_compare/logs"
mkdir -p "$LOGS"

JOBS="${JOBS:-18}"
dep=()
for i in $(seq 1 "$JOBS"); do
    id=$(sbatch --parsable --job-name="newton_$i" "${dep[@]}" \
        --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" "$ORIG/newton_chain.sbatch")
    echo "newton_$i $id"
    dep=(--dependency="afterok:$id")
done
