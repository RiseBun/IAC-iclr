#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/slurmfs-4090node3/user_data/zchen897/model_registry/lingbot_va_base"
PART="/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_drivewam_partition"
OUT="/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_drivewam_outputs_new"
RUNNER="/mnt/slurmfs-4090node1/homes/zchen897/iac_new/scripts/run_drivewam_native_batch.py"
CHECKPOINT="/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/drivewam_runtime_smoke"
CONDA="/mnt/slurmfs-4090node1/homes/zchen897/miniforge3/etc/profile.d/conda.sh"
DL_PIDS="${DL_PIDS:-4159529 4182886 4183964 4183965 4183966}"
DL_LOGS="${DL_LOGS:-/mnt/slurmfs-4090node3/user_data/zchen897/model_registry/lingbot_va_base/dl_transformer1.log /mnt/slurmfs-4090node3/user_data/zchen897/model_registry/lingbot_va_base/dl_text2.log /mnt/slurmfs-4090node3/user_data/zchen897/model_registry/lingbot_va_base/dl_vae.log}"

echo "waiting for LingBot-VA downloader pids ${DL_PIDS}" >&2
while true; do
  active=0
  for pid in $DL_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then active=1; fi
  done
  [[ "$active" == 0 ]] && break
  sleep 30
done
for log in $DL_LOGS; do
  grep -q '^done:' "$log" || { echo "download did not complete: $log" >&2; exit 5; }
done

declare -A SIZES=(
  [transformer/diffusion_pytorch_model-00001-of-00003.safetensors]=4821987820
  [transformer/diffusion_pytorch_model-00002-of-00003.safetensors]=4821655760
  [transformer/diffusion_pytorch_model-00003-of-00003.safetensors]=535373816
  [text_encoder/model-00001-of-00003.safetensors]=4935812536
  [text_encoder/model-00002-of-00003.safetensors]=4983103192
  [text_encoder/model-00003-of-00003.safetensors]=1442935480
  [vae/diffusion_pytorch_model.safetensors]=2818777808
)
for rel in "${!SIZES[@]}"; do
  path="$BASE/$rel"
  [[ -f "$path" ]] || { echo "missing $path" >&2; exit 2; }
  actual=$(stat -c '%s' "$path")
  [[ "$actual" == "${SIZES[$rel]}" ]] || { echo "size mismatch $path: $actual != ${SIZES[$rel]}" >&2; exit 3; }
done
BASE_URL="https://hf-mirror.com/robbyant/lingbot-va-base/resolve/main"
for rel in transformer/config.json transformer/diffusion_pytorch_model.safetensors.index.json text_encoder/config.json text_encoder/model.safetensors.index.json tokenizer/special_tokens_map.json tokenizer/spiece.model tokenizer/tokenizer.json tokenizer/tokenizer_config.json vae/config.json; do
  if [[ ! -s "$BASE/$rel" ]]; then
    mkdir -p "$(dirname "$BASE/$rel")"
    curl -fL --retry 5 --connect-timeout 20 "$BASE_URL/$rel" -o "$BASE/$rel"
  fi
  [[ -s "$BASE/$rel" ]] || { echo "missing metadata $BASE/$rel" >&2; exit 4; }
done

mkdir -p "$OUT"
declare -a COUNTS=(187 186 186 186)
for gpu in 0 1 2 3; do
  shard="$PART/shards/shard_${gpu}"
  out="$OUT/shard_${gpu}"
  mkdir -p "$out"
  if [[ -s "$out/manifest.json" ]]; then
    echo "skip existing $out" >&2
    continue
  fi
  log="$OUT/shard_${gpu}.log"
  setsid bash -lc "source '$CONDA'; conda activate drivingworld; exec python '$RUNNER' --data '$shard' --checkpoint '$CHECKPOINT' --base '$BASE' --output '$out' --num-samples '${COUNTS[$gpu]}' --video-steps 4 --action-steps 4 --device cuda:${gpu}" >"$log" 2>&1 < /dev/null &
  echo "launched gpu=${gpu} pid=$! count=${COUNTS[$gpu]} log=$log" >&2
done
