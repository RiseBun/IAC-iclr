#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:?branch required: left or right}"
case "$BRANCH" in
  left|right) ;;
  *) echo "unsupported branch: $BRANCH" >&2; exit 2 ;;
esac

BASE="/mnt/slurmfs-4090node3/user_data/zchen897/model_registry/lingbot_va_base"
PART="${PART:-/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_drivewam_partition}"
OUT="${OUT:-/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_drivewam_ccfc_${BRANCH}}"
RUNNER="/mnt/slurmfs-4090node1/homes/zchen897/iac_new/scripts/run_drivewam_native_batch.py"
CHECKPOINT="/mnt/slurmfs-4090node1/homes/zchen897/work_dirs/drivewam_runtime_smoke"
CONDA="/mnt/slurmfs-4090node1/homes/zchen897/miniforge3/etc/profile.d/conda.sh"

mkdir -p "$OUT"
for gpu in 0 1 2 3; do
  shard="$PART/shards/shard_${gpu}"
  out="$OUT/shard_${gpu}"
  log="$OUT/shard_${gpu}.log"
  count=$(find "$shard" -maxdepth 1 -name '*.pkl' | wc -l)
  [[ "$count" == "0" ]] && continue
  if [[ -s "$out/manifest.json" ]]; then
    echo "skip existing gpu=$gpu out=$out" >&2
    continue
  fi
  mkdir -p "$out"
  setsid bash -lc "source '$CONDA'; conda activate drivingworld; exec python '$RUNNER' --data '$shard' --checkpoint '$CHECKPOINT' --base '$BASE' --output '$out' --num-samples '$count' --video-steps 4 --action-steps 4 --device cuda:${gpu} --command-override '$BRANCH'" >"$log" 2>&1 < /dev/null &
  echo "launched branch=$BRANCH gpu=$gpu pid=$! count=$count log=$log" >&2
done
