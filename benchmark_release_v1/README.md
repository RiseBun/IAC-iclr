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

The role of each directory is fixed as follows:

| Path | Role | Required for reproduction |
|---|---|---|
| `configs/` | Frozen image-side geometry configuration (`plane.json`) | Yes |
| `datasets/` | Leakage-safe public manifests and split audits | Yes (metadata) |
| `docs/` | Frozen protocol, dataset contract and main-table definitions | Yes (protocol) |
| `scripts/` | Manifest builders, WAM-output audits and Level-1 scorers | Yes |
| `src/iac_new/` | Reusable flow, geometry, posterior and scoring implementation | Yes |
| `tests/` | Deterministic unit and protocol checks | Recommended |
| `weights/` | Frozen RAFT-Large checkpoint, provenance and SHA-256 checksums | Yes for the default probe |
| `pyproject.toml`, `VERSION` | Python package metadata and release identity | Yes |

The package intentionally contains no raw camera frames, private paths, WAM
checkpoints or generated experiment logs. Those inputs are mounted through the
private manifest interface at evaluation time.

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

## Level-1 metric summary

The following is the **new frozen `benchmark_v1` experiment**, evaluated on
580 records (500 NAVSIM + 80 Waymo), with strict shape gating and shape
fallback disabled. The reference is logged future ego state, so these numbers
validate the image measurement layer; they are not WAM causal scores.

| Metric | Result | Interpretation |
|---|---:|---|
| Non-stop shape coverage | **440/468 = 94.0%** | At least one shape interval is observable on most moving samples |
| Stop recognition | **92/112 = 82.1%** | Stop is reported by the dedicated stop layer, not by a fake velocity estimate |
| Lateral-speed MAE / within tolerance | **0.095 m/s / 98.4%** | Reliable lateral-motion amplitude (`0.50 m/s` tolerance) |
| Yaw-rate MAE / within tolerance | **0.029 rad/s / 97.0%** | Reliable heading-change measurement (`0.15 rad/s` tolerance) |
| Curvature MAE / within tolerance | **0.022 1/m / 86.1%** | Usable curvature measurement (`0.06 1/m` tolerance), with a heavier tail |
| Turn-layer yaw increment | **Gate passed** (106/114 evaluable) | Future content improves over history, matched-shuffle and reversal controls |
| Turn-layer curvature increment | **Gate passed** (106/114 evaluable) | Curvature uses the correct future and its temporal order |
| Full-pool curvature increment | **Gate passed** | The curvature signal remains specific on the mixed benchmark pool |

The formal Level-1 comparison therefore uses lateral motion, yaw rate and
curvature. Absolute speed, acceleration and metric forward distance remain
diagnostic only. The complete definitions, tolerances and bootstrap-gate
wording are frozen in
[`docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md`](docs/LEVEL1_MAIN_TABLE_BENCHMARK_V1_ZH.md).

### Reliability and long-tail diagnostics

The 78-sample development audit is retained only to expose failure modes and
coverage. It reports mean interval observability **77.2%**, fully observable
sample rate **61.5%**, and overall core-pass rate **82.1%**. Strong turns are
observable (interval coverage **100%**) but have only **40.0%** core-pass rate;
the limitation is accumulated lateral error, not invisibility. On the
scene-aware non-overlap 25-sample audit, core-pass rate is **88.0%** and
interval coverage **58.5%**, but braking has only one sample. These diagnostics
must not be merged into the frozen 580-record headline or used to claim
causality.

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
