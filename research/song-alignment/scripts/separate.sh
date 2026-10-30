#!/usr/bin/env bash
# Extract the vocal stem with Mel-Band RoFormer (Kim).
# usage: separate.sh INPUT_AUDIO OUTPUT_WAV [MODEL]
#   MODEL defaults to vocals_mel_band_roformer.ckpt; for BS-RoFormer use
#   model_bs_roformer_ep_317_sdr_12.9755.ckpt
# audio-separator runs in its own uvx environment (it pins different deps).
set -euo pipefail
in=$1 out=$2 model=${3:-vocals_mel_band_roformer.ckpt}
models=${AUDIO_SEPARATOR_MODELS:-$HOME/.cache/audio-separator/models}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
uvx --python 3.12 --from "audio-separator[gpu]" --with audioread audio-separator "$in" \
  -m "$model" --model_file_dir "$models" --output_dir "$tmp" \
  --output_format WAV --single_stem Vocals
mv "$tmp"/*.wav "$out"
