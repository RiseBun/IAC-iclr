#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${1:?usage: run_continuous_alignment_repro.sh MANIFEST DECODER_SCORES [RUN_DIR] [REFERENCE_SOURCE]}"
SCORES="${2:?usage: run_continuous_alignment_repro.sh MANIFEST DECODER_SCORES [RUN_DIR] [REFERENCE_SOURCE]}"
RUN_DIR="${3:-${ROOT}/results/continuous_alignment_$(date +%Y%m%d_%H%M%S)}"
REFERENCE_SOURCE="${4:-logged_gt}"

export PYTHONPATH="${ROOT}/src:${ROOT}"
mkdir -p "${RUN_DIR}"
cd "${ROOT}"

cp "${MANIFEST}" "${RUN_DIR}/input_manifest.jsonl"
cp "${SCORES}" "${RUN_DIR}/input_decoder_scores.jsonl"
sha256sum "${MANIFEST}" "${SCORES}" > "${RUN_DIR}/INPUT_SHA256SUMS.txt"
find "${ROOT}/src" "${ROOT}/scripts" -type f \
  ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_DIR}/SOURCE_SHA256SUMS.txt"
"${PYTHON_BIN}" -m pytest -q --import-mode=importlib "${ROOT}/tests" | tee "${RUN_DIR}/pytest.log"
"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_continuous_motion_alignment.py" \
  --manifest "${RUN_DIR}/input_manifest.jsonl" \
  --scores "${RUN_DIR}/input_decoder_scores.jsonl" \
  --reference-source "${REFERENCE_SOURCE}" \
  --output "${RUN_DIR}/continuous_motion_strict.json" \
  | tee "${RUN_DIR}/strict.log"
"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_continuous_motion_alignment.py" \
  --manifest "${RUN_DIR}/input_manifest.jsonl" \
  --scores "${RUN_DIR}/input_decoder_scores.jsonl" \
  --reference-source "${REFERENCE_SOURCE}" \
  --include-uncertain \
  --output "${RUN_DIR}/continuous_motion_with_uncertain.json" \
  | tee "${RUN_DIR}/with_uncertain.log"

echo "${RUN_DIR}"
