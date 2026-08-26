#!/usr/bin/env bash
set -euo pipefail

CALIBRATION_MANIFEST="${1:?usage: run_server_validation.sh CALIBRATION_MANIFEST TEST_MANIFEST OUTPUT_DIR [DEVICE]}"
TEST_MANIFEST="${2:?usage: run_server_validation.sh CALIBRATION_MANIFEST TEST_MANIFEST OUTPUT_DIR [DEVICE]}"
OUTPUT_DIR="${3:?usage: run_server_validation.sh CALIBRATION_MANIFEST TEST_MANIFEST OUTPUT_DIR [DEVICE]}"
DEVICE="${4:-cuda}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

python -m iac_new.check_splits \
  --calibration "${CALIBRATION_MANIFEST}" \
  --test "${TEST_MANIFEST}"

for config in \
  configs/navsim_raft_small_plane.json \
  configs/navsim_raft_large_plane.json \
  configs/navsim_raft_large_plane_fb.json; do
  name="$(basename "${config}" .json)"
  python -m iac_new.evaluate \
    --manifest "${CALIBRATION_MANIFEST}" \
    --config "${config}" \
    --output "${OUTPUT_DIR}/${name}_calibration_scores.jsonl" \
    --device "${DEVICE}"
  python -m iac_new.calibrate \
    --scores "${OUTPUT_DIR}/${name}_calibration_scores.jsonl" \
    --coverage 0.90 \
    --output "${OUTPUT_DIR}/${name}_calibration.json"
  python -m iac_new.evaluate \
    --manifest "${TEST_MANIFEST}" \
    --config "${config}" \
    --calibration "${OUTPUT_DIR}/${name}_calibration.json" \
    --output "${OUTPUT_DIR}/${name}_calibrated.jsonl" \
    --device "${DEVICE}"
done
