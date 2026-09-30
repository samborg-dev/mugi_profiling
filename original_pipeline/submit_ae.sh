#!/bin/bash
set -euo pipefail

ORIG="$HOME/mugi_profiling/original_pipeline"
LOGS="$HOME/orig_compare/logs"
mkdir -p "$LOGS"

CHUNKS="${CHUNKS:-split_1_c1 split_0_c1 split_0_c2 split_0_c3 split_0_c4 split_0_c5 split_0_c6 split_1_c2 split_1_c3 split_1_c4 split_1_c5 split_1_c6}"

ids=()
for chunk in $CHUNKS; do
    id=$(sbatch --parsable --job-name="ae_$chunk" --export=ALL,CHUNK="$chunk" \
        --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" "$ORIG/ae_chunk.sbatch")
    echo "ae_$chunk $id"
    ids+=("$id")
done

if [ -z "${NO_E2E:-}" ]; then
    deps=$(IFS=:; echo "${ids[*]}")
    id=$(sbatch --parsable --job-name=ae_e2e --dependency="afterok:$deps" \
        --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" "$ORIG/ae_e2e.sbatch")
    echo "ae_e2e $id (after all chunks succeed)"
fi
