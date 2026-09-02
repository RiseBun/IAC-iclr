#!/usr/bin/env bash
set -euo pipefail

RAW=/mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/raw/perception_v2/validation
while [ "$(find "$RAW/camera_image" -maxdepth 1 -type f -name '*.parquet' | wc -l)" -lt 202 ]; do
  sleep 60
done

~/miniforge3/envs/navsim_seerdrive/bin/python ~/iac_new/scripts/prepare_waymo_level1_samples.py \
  --input "$RAW" \
  --output /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/frames/level1_v2 \
  --stride-frames 55 > /tmp/waymo_prepare_v2.log

cd ~/iac_new
PYTHONPATH=src python3 scripts/build_level1_benchmark_v1.py \
  --navsim /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/manifests/navsim_level1_inventory_4h8f.jsonl \
  --waymo /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/frames/level1_v2/manifest.jsonl \
  --output-root /mnt/slurmfs-4090node3/user_data/zchen897/iac_waymo/manifests/benchmark_v2 \
  --benchmark-per-dataset 500 \
  --dev-per-dataset 250 \
  --seed 20260902 > /tmp/waymo_benchmark_v2.log
