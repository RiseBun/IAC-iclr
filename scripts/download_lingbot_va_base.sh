#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)/model_checkpoints/lingbot_va_base}"
BASE="https://hf-mirror.com/robbyant/lingbot-va-base/resolve/main"
DOWNLOADER="${2:-$(pwd)/scripts/parallel_range_download.py}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$ROOT/transformer" "$ROOT/text_encoder" "$ROOT/vae" "$ROOT/tokenizer"

declare -A SIZES=(
  [transformer/diffusion_pytorch_model-00001-of-00003.safetensors]=4821987820
  [transformer/diffusion_pytorch_model-00002-of-00003.safetensors]=4821655760
  [transformer/diffusion_pytorch_model-00003-of-00003.safetensors]=535373816
  [text_encoder/model-00001-of-00003.safetensors]=4935812536
  [text_encoder/model-00002-of-00003.safetensors]=4983103192
  [text_encoder/model-00003-of-00003.safetensors]=1442935480
  [vae/diffusion_pytorch_model.safetensors]=2818777808
)

for file in "${!SIZES[@]}"; do
  output="$ROOT/$file"
  if [[ -s "$output" ]]; then
    echo "exists $output"
  else
    echo "download $file"
    "$PYTHON" "$DOWNLOADER" --url "$BASE/$file" --output "$output" \
      --size "${SIZES[$file]}" --workers 8 --chunk-mb 64 --retries 8
  fi
done

for file in \
  transformer/config.json \
  transformer/diffusion_pytorch_model.safetensors.index.json \
  text_encoder/config.json \
  text_encoder/model.safetensors.index.json \
  tokenizer/special_tokens_map.json \
  tokenizer/spiece.model \
  tokenizer/tokenizer.json \
  tokenizer/tokenizer_config.json \
  vae/config.json; do
  output="$ROOT/$file"
  if [[ -s "$output" ]]; then
    echo "exists $output"
  else
    curl -fL --retry 5 --connect-timeout 20 "$BASE/$file" -o "$output"
  fi
done

echo "complete $ROOT"
