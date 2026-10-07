#!/bin/bash
set -euo pipefail

ORIG="$HOME/mugi_profiling/original_pipeline"
REPO=https://github.com/UnaryLab/mugi_profiling.git

[ -d "$HOME/mugi_ae" ] || git clone --branch asplos_2026_ae "$REPO" "$HOME/mugi_ae"
[ -d "$HOME/mugi_newton" ] || git clone --branch newton "$REPO" "$HOME/mugi_newton"

ae_tip=$(git -C "$HOME/mugi_ae" rev-parse --short=7 HEAD)
nw_tip=$(git -C "$HOME/mugi_newton" rev-parse --short=7 HEAD)
[ "$ae_tip" = f83978e ] || { echo "mugi_ae is at $ae_tip, expected f83978e"; exit 1; }
[ "$nw_tip" = 1c57038 ] || { echo "mugi_newton is at $nw_tip, expected 1c57038"; exit 1; }

for clone in mugi_ae mugi_newton; do
    sed -i 's#meta-llama/Llama-2-7b-hf#NousResearch/Llama-2-7b-hf#' \
        "$HOME/$clone/config/model_config/llama/llama_2_7b.yaml"
done

snaps=("${HF_HOME:-/work/hdd/bebv/$USER/hf}"/hub/models--NousResearch--Llama-2-7b-hf/snapshots/*/)
[ "${#snaps[@]}" -eq 1 ] && [ -d "${snaps[0]}" ] || { echo "expected one cached NousResearch/Llama-2-7b-hf snapshot, found: ${snaps[*]}"; exit 1; }
mkdir -p "$HOME/mugi_ae/weights"
ln -sfn "${snaps[0]}" "$HOME/mugi_ae/weights/llama-2-7b-hf"
sed -i 's#name: NousResearch/Llama-2-7b-hf#name: weights/llama-2-7b-hf#' \
    "$HOME/mugi_ae/config/model_config/llama/llama_2_7b.yaml"

sed -i 's/^n_samples: .*/n_samples: 8/' \
    "$HOME/mugi_ae/config/parameter_config/parameter_config.yaml" \
    "$HOME/mugi_ae/config/parameter_config/end_to_end_config.yaml" \
    "$HOME/mugi_newton/config/parameter_config/parameter_config.yaml"

mkdir -p "$HOME/mugi_ae/config/chunks" "$HOME/orig_compare/logs"
cp "$ORIG"/ae_chunks/*.yaml "$HOME/mugi_ae/config/chunks/"

echo "mugi_ae $ae_tip, mugi_newton $nw_tip"
grep -H "name:\|batch_size" "$HOME/mugi_ae/config/model_config/llama/llama_2_7b.yaml" \
    "$HOME/mugi_newton/config/model_config/llama/llama_2_7b.yaml"
grep -H "n_samples" "$HOME/mugi_ae/config/parameter_config/parameter_config.yaml" \
    "$HOME/mugi_ae/config/parameter_config/end_to_end_config.yaml" \
    "$HOME/mugi_newton/config/parameter_config/parameter_config.yaml"
ls "$HOME/mugi_ae/config/chunks"
ls -l "$HOME/mugi_ae/weights/llama-2-7b-hf"
