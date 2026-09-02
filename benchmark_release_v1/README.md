# IAC Benchmark · Level-1 Release v1

中文文档：[README_zh.md](README_zh.md)

IAC (Imagined-future and Action Consistency) is a reproducible benchmark for
testing whether a WAM's imagined future is aligned with the action it emits.
This release contains the **Level-1 continuous image-side measurement**: a
candidate-blind geometry probe extracts forward/lateral motion, heading and
curvature from an image sequence, then compares those signals with the WAM's
future action/trajectory. It is a measurement layer, not a claim of causality
by itself.

## Release contents

```text
benchmark_release_v1/
├── configs/         frozen RAFT-Large ground-plane configuration
├── datasets/       public, path-sanitised benchmark/dev manifests and audits
├── docs/            frozen benchmark protocol and Level-1 main table
├── scripts/         dataset builders, audits and scorers
├── src/iac_new/     reusable Level-1 geometry, flow and scoring library
├── tests/           deterministic unit and protocol tests
├── weights/         frozen RAFT-Large checkpoint + checksum
├── pyproject.toml
├── VERSION
├── README.md
└── README_zh.md
```

Raw NAVSIM/Waymo frames, private absolute paths, WAM checkpoints and generated
logs are deliberately excluded. They stay in the data storage area and are
referenced through the manifest interface, which prevents accidental leakage
and keeps the GitHub package portable.

## Frozen protocol

- The private evaluation input contains four history frames (`t <= 0`) and
  eight future frames at `0.5, …, 4.0 s`. The public manifests in this
  repository do **not** contain future images; they contain only protocol
  metadata and sanitised sample identity.
- Exact timestamps, camera intrinsics/extrinsics and distortion are required.
- Scene/log groups are disjoint between benchmark and dev; split construction
  is deterministic and audited.
- Current internal freeze: 500 NAVSIM + 80 Waymo benchmark records and
  250 NAVSIM + 20 Waymo dev records (580/270 total). The Waymo expansion is
  continuing outside this release; it will become v2 after audit.
- Strata include stop, braking, acceleration, lateral-turn and straight-cruise
  so performance is not dominated by straight driving.

The public JSONL manifests retain sample identity, split, stratum, timestamps,
calibration and history state, while omitting raw image paths and future ground
truth. Use the private data root configured by your local environment to attach
frames at run time.

## Install and verify

```bash
python -m pip install -e .
PYTHONPATH=src:. python -m pytest tests -q
sha256sum -c weights/SHA256SUMS.txt
```

The test suite is deterministic and covers calibration interfaces, temporal
geometry, flow reliability, trajectory decoding, split isolation and the
Level-1 continuous scorer.

## Run the Level-1 measurement

On the evaluation server, first run the frozen image decoder on WAM-generated
future-image records with `configs/plane.json`, producing a decoder-score JSONL.
The public manifests alone cannot be scored because their future images and
future reference states remain private. Action/trajectory fields are read only
at the final comparison stage and must not be passed to the image-side decoder.

```bash
python scripts/audit_wam_level1_outputs.py --generated <wam_generated_records.jsonl> \
  --output <wam_output_audit.json>
python scripts/evaluate_continuous_decoder.py \
  --manifest <private_wam_generated_records.jsonl> \
  --config configs/plane.json \
  --output <decoder_scores.jsonl>
python scripts/evaluate_continuous_motion_alignment.py \
  --manifest <private_evaluation_manifest.jsonl> \
  --scores <decoder_scores.jsonl> \
  --reference-source action \
  --require-eight-frame-four-second \
  --output <out_dir>
```

The report contains path-normalised lateral/heading/curvature errors,
candidate-blind observability and coverage-risk. Speed is diagnostic only in
v1; it is not a formal Level-1 score. This release does not claim CCFC or FCS:
those require a separate paired WAM intervention package with generated future
images and are outside the public v1 protocol.

## Dataset preparation

Use `scripts/build_level1_benchmark_v1.py` to construct deterministic splits
from private NAVSIM/Waymo records. `scripts/prepare_waymo_level1_samples.py`
converts Waymo Perception v2 shards to the 4+8-frame interface, and
`scripts/build_public_benchmark_manifest.py` creates a leakage-safe public
manifest. See `docs/BENCHMARK_DATASET_V1_ZH.md` for the data protocol and
`docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md` for the frozen main table.

## Data and license

This repository contains code, manifests and third-party optical-flow weights.
NAVSIM and Waymo data remain subject to their original access terms and are
not redistributed here. Check `weights/README.md` for checkpoint provenance and
upstream licenses.
