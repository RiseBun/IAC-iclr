#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${1:?usage: run_event78_repro.sh MANIFEST [RUN_DIR]}"
RUN_DIR="${2:-${ROOT}/results/repro_event78_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-${ROOT}/configs/navsim_continuous_decoder_plane.json}"

export PYTHONPATH="${ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${RUN_DIR}"
cd "${ROOT}"
cp "${BASH_SOURCE[0]}" "${RUN_DIR}/run_event78_repro.sh"
cp "${MANIFEST}" "${RUN_DIR}/input_manifest.jsonl"
cp "${CONFIG}" "${RUN_DIR}/input_config.json"
RUN_MANIFEST="${RUN_DIR}/input_manifest.jsonl"
RUN_CONFIG="${RUN_DIR}/input_config.json"

find "${ROOT}/src" "${ROOT}/scripts" "${ROOT}/configs" -type f \
  ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_DIR}/SOURCE_SHA256SUMS.txt"
sha256sum "${MANIFEST}" "${CONFIG}" > "${RUN_DIR}/INPUT_SHA256SUMS.txt"
"${PYTHON_BIN}" -m pip freeze > "${RUN_DIR}/PYTHON_FREEZE.txt"
nvidia-smi -q > "${RUN_DIR}/NVIDIA_SMI.txt"
"${PYTHON_BIN}" -c \
  'import cv2,numpy,torch,torchvision; print({"torch":torch.__version__,"torchvision":torchvision.__version__,"numpy":numpy.__version__,"opencv":cv2.__version__})' \
  > "${RUN_DIR}/RUNTIME_VERSIONS.txt"

"${PYTHON_BIN}" -m pytest -q --import-mode=importlib "${ROOT}/tests" | tee "${RUN_DIR}/pytest.log"

"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_continuous_decoder.py" \
  --manifest "${RUN_MANIFEST}" \
  --config "${RUN_CONFIG}" \
  --output "${RUN_DIR}/raft_scores.jsonl" \
  --device cuda \
  | tee "${RUN_DIR}/decoder.log"

# The validated 78-sample protocol uses the signed ego-heading trajectory
# directly. The experimental temporal-heading fallback is intentionally off.
"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_maneuver_events.py" \
  --scores "${RUN_DIR}/raft_scores.jsonl" \
  --manifest "${RUN_MANIFEST}" \
  --output "${RUN_DIR}/event_metrics.json" \
  | tee "${RUN_DIR}/event.log"

"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_continuous_motion_alignment.py" \
  --manifest "${RUN_MANIFEST}" \
  --scores "${RUN_DIR}/raft_scores.jsonl" \
  --reference-source logged_gt \
  --output "${RUN_DIR}/continuous_motion_strict.json" \
  | tee "${RUN_DIR}/continuous_motion.log"

echo "${RUN_DIR}"
