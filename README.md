# IAC: Event-Causal Evaluation for World Action Models

IAC asks a narrower and more causal question than video quality or task
success alone:

> Did the action-conditioned imagined future express the intended event, and
> did the planner actually use that future when choosing an action?

The benchmark does not reconstruct an exact metric trajectory from monocular
front-view video. It extracts observable maneuver events and evaluates three
links separately:

```text
conditioned action -> imagined event -> selected action -> realized event/success
        Event-CC              FUI                 Event-FCS
```

## Method

The frozen image-side probe is more than an optical-flow model:

```text
history + imagined front-camera frames
  -> RAFT-Large forward/backward flow
  -> consistency and dynamic suppression
  -> calibrated ground-plane ego geometry
  -> candidate-blind continuous motion decoder
  -> observability and abstention
  -> maneuver skeleton
  -> interval event posterior
```

V1's formal primary signal is lateral event support: `keep_lane`, `turn_left`,
and `turn_right`. Metric speed and exact trajectory recovery are not primary
claims. Optional depth, DINOv2, semantic, CoTracker, and SEA-RAFT modules remain
research ablations; they are not part of the reported 78-sample default.

## Three evaluation levels

1. **Level 1, action response:** fixed history, common generation seed, at
   least three materially distinct action branches, and a full K-by-K
   image-event/action-event matrix.
2. **Level 2, Event-CC + Event-FCS:** adds independently measured realized
   events and an explicit task-success boolean. FCS is always reported with
   foresight coverage and Joint-FCS.
3. **Level 3, causal closure (FUI):** reruns one fixed planner while permuting
   imagined futures across action slots. Planner identity and nuisance seed are
   held fixed, and independent null resamples measure stochastic action changes.

The readiness audit is fail-closed. Diagnostic decoded-trajectory adapters are
tagged with a non-formal evidence source and cannot silently enter the formal
benchmark.

## Install and test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
```

The RAFT image probe requires CUDA for practical evaluation. Event metrics and
their synthetic tests run on CPU.

## Benchmark workflow

Audit the data contract and freeze a scene-disjoint calibration/holdout split:

```bash
PYTHONPATH=src:. python scripts/audit_event_benchmark.py \
  --groups event_groups.jsonl \
  --output work/readiness.json \
  --split-output work/scene_split.json \
  --require-level 2
```

Run Event-CC, Event-FCS, and FUI when planner reruns are present:

```bash
PYTHONPATH=src:. python scripts/evaluate_event_causal_metrics.py \
  --groups event_groups.jsonl \
  --output work/event_causal_report.json
```

Evaluate the four interaction-level risk/clear causal chains:

```bash
PYTHONPATH=src:. python scripts/evaluate_causal_chains.py \
  --records configs/causal_chain_v1.example.jsonl \
  --output work/causal_chain_report.json \
  --require-ready
```

Mine a high-recall, annotation-only candidate pool from nuPlan mini logs:

```bash
PYTHONPATH=src:. python scripts/mine_nuplan_causal_candidates.py \
  --db-root /path/to/nuplan-v1.1/splits/mini \
  --sensor-root /path/to/sensor_blobs \
  --output work/nuplan_causal_candidates.jsonl \
  --max-per-chain 40
```

Scenario tags remain candidate provenance and are never promoted to formal labels.

Run the oracle, identical-future, and cyclic action-swap controls:

```bash
PYTHONPATH=src:. python scripts/run_event_control_suite.py \
  --groups event_groups.jsonl \
  --output work/control_suite.json
```

Validate the frozen probe against blinded human labels on actual WAM videos:

```bash
PYTHONPATH=src python scripts/build_blind_event_annotation_pack.py \
  --source epona /path/to/epona_manifest.jsonl /path/to/epona_event_groups.jsonl \
  --source drivewam /path/to/drivewam_manifest.jsonl /path/to/drivewam_event_groups.jsonl \
  --output-dir work/blind_event_annotations
```

Re-run the frozen 78-sample image-event probe with a local NAVSIM manifest:

```bash
PYTHON_BIN=/path/to/python \
  bash scripts/run_event78_repro.sh /path/to/event78_manifest.jsonl
```

## Current evidence

The frozen default recovered lateral events on the balanced NAVSIM set with:

- 78 samples, 312 intervals, 19 scenes
- accuracy: `0.977564`
- macro-F1: `0.974382`
- onset MAE: `0.0 s`

These numbers validate the image-to-event measurement component. They are not
the final joint WAM score. Existing Epona and DriveWAM reports are diagnostic
because their records predate the complete seed/history/planner-rerun contract.

See [the metric specification](docs/EVENT_CAUSAL_METRICS_V1.md), [method
provenance](docs/BEST_METHOD_PROVENANCE.md), and [snapshot
scope](docs/IAC_RAFT_EVENT_CAUSAL_V1_SCOPE.md). The independent WAM-video
validation protocol is specified in
[Measurement Validity V1](docs/MEASUREMENT_VALIDITY_V1.md). The recent AD-WAM
input/output survey, action-head comparison, and proposed event taxonomy are
documented in [AD-WAM Landscape and Event Taxonomy (2026-08-26, Chinese)](docs/AD_WAM_LANDSCAPE_AND_EVENT_TAXONOMY_20260826_ZH.md).
The fail-closed risk/clear record contract and scoring definition for the four
interaction chains are in [Causal Chain Protocol V1 (Chinese)](docs/CAUSAL_CHAIN_PROTOCOL_V1_ZH.md).
The first real nuPlan candidate-mining run and its evidence boundary are recorded
in [Causal Chain Candidate Mining (2026-08-26, Chinese)](docs/CAUSAL_CHAIN_CANDIDATE_MINING_20260826_ZH.md).

## Repository map

- `src/iac_new/`: image probe, event representation, metrics, readiness, and FUI
- `scripts/evaluate_continuous_decoder.py`: frozen image-to-motion decoder
- `scripts/evaluate_maneuver_events.py`: interval event recovery evaluation
- `scripts/audit_event_benchmark.py`: formal benchmark gate and scene split
- `scripts/evaluate_event_causal_metrics.py`: Event-CC, Event-FCS, and FUI
- `scripts/run_event_control_suite.py`: causal sanity-check controls
- `scripts/evaluate_causal_chains.py`: four interaction-chain readiness and scoring
- `scripts/mine_nuplan_causal_candidates.py`: nuPlan four-chain annotation candidate miner
- `configs/navsim_continuous_decoder_plane.json`: reported default configuration
- `configs/causal_chain_v1.example.jsonl`: executable four-chain risk/clear example
- `tests/`: unit and protocol tests

Model checkpoints, datasets, generated videos, and experiment outputs are not
stored in Git.
