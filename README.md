# IAC: Continuous Foresight-Action Evaluation for World Action Models

IAC asks a narrower and more causal question than video quality or task
success alone:

> Does the ego-motion imagined in the future video agree with the action head,
> and do both respond consistently under a controlled intervention?

The primary representation is a continuous ego-motion profile, not a recovered
global trajectory and not a text event:

```text
m(t) = [speed, acceleration, lateral speed, yaw rate, curvature]
```

The image branch is candidate-blind: action waypoints are withheld until the
future-image motion profile has been decoded. Events are derived afterward for
interpretation and stratification only.

The converged method, data gates, 78-sample results, and GitHub-rendered flow
chart are in [Continuous Foresight-Action Alignment V1](docs/CONTINUOUS_FORESIGHT_ALIGNMENT_V1_ZH.md).

Current work is focused on [Level 1 Continuous Alignment V2](docs/LEVEL1_CONTINUOUS_ALIGNMENT_V2_ZH.md):
continuous alignment must beat a strong history-only null, a history-speed
matched shuffled future, and a time-reversed future before a motion component
is accepted as carrying incremental foresight information.

The first [history-anchored longitudinal residual experiment](docs/LEVEL1_LONGITUDINAL_RESIDUAL_V3_ZH.md)
reduces raw RAFT speed error substantially, but its scene-disjoint test result
does not beat the strong history-only null. It is retained as a falsified,
reproducible baseline rather than reported as a successful metric.

The current [optimizer-internal longitudinal residual V4](docs/LEVEL1_LONGITUDINAL_OPTIMIZER_V4_ZH.md)
keeps speed as an evaluation-sufficient behavioral signal. It is stable on the
78-sample run, but the manifest is still 4 future frames/2 seconds and the
longitudinal incremental gate is not resolved; no causal success claim is made.

The native [8-frame/4-second validation](docs/LEVEL1_LONGITUDINAL_OPTIMIZER_V4_8F_ZH.md)
now uses real continuous NAVSIM frames. It confirms the long-horizon contract, but
also shows that longitudinal accuracy degrades at 4 seconds; speed remains a
diagnostic probe, not a formal causal metric.

The full-method search and the proposed persistent-scale/global-residual repair are
documented in [Level 1 longitudinal solution search and V5 plan](docs/LEVEL1_LONGITUDINAL_SOLUTION_SEARCH_20260827_ZH.md).

The cleaned overlap-aware NAVSIM package is in `datasets/navsim_level1_v5/`:
`navsim_level1_v5_all_78.jsonl` is the reproducible development source,
`navsim_level1_v5_eval_nonoverlap.jsonl` contains 25 scene-aware non-overlapping
windows for statistical claims, and `navsim_level1_v5_development_overlap.jsonl`
contains the remaining 53 windows for diagnostics only. All three retain the
candidate bank for protocol compatibility, but Level 1 must use only `logged`
after image decoding. The audit explicitly records that native realized futures
validate the measurement probe and do not support a WAM causal claim.

![Converged IAC continuous foresight-action architecture](docs/figures/continuous_foresight_alignment_v1.svg)

The Level 1 longitudinal branch and its fail-closed next-step boundary are shown in
[the updated flow chart](docs/figures/level1_continuous_alignment_v2.png).

For actual WAM-generated future images, use
`scripts/build_wam_level1_continuous_manifest.py`. It keeps historical calibration
inputs and the WAM action reference, but removes future realized state from the
image-side record.

## Method

The frozen image-side probe is more than an optical-flow model:

```text
history + imagined front-camera frames
  -> RAFT-Large forward/backward flow
  -> consistency and dynamic suppression
  -> calibrated ground-plane ego geometry
  -> candidate-blind continuous motion decoder
  -> observability and abstention
  -> continuous ego-motion posterior
  -> direct comparison with action-waypoint kinematics
  -> event interpretation layer
```

V1 reports speed, acceleration, lateral speed, yaw rate, and curvature together
with observability, abstention, posterior coverage, and coverage-risk curves.
RAFT-Large remains the default because it completed 78/78 samples; SEA-RAFT is
a challenger. Their paired speed difference is not statistically resolved.

For interaction events, an actor-relative extension adds instance association,
calibrated ground-contact/depth projection, robust temporal fitting, metric
distance and closing/lateral speed intervals, TTC, observability, and
abstention. It is implemented and synthetically tested, but is not yet a
formally validated 78-sample metric.

## Evaluation levels

1. **Level 0, measurement validity:** compare image-derived motion with an
   independent logged trajectory; this validates the probe, not WAM causality.
2. **Level 1, continuous future-action alignment:** compare `m_F(t)` from the
   generated future with `m_A(t)` from the held-out action head.
3. **Level 2, counterfactual consistency:** for paired risk/clear interventions,
   compare `Delta m_F(t)` with `Delta m_A(t)` under a fixed history and seed.
4. **Level 3, causal closure and FCS:** add planner future-swap reruns,
   independently realized state, and explicit task success.

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

Validate continuous motion on held-out logged waypoints:

```bash
PYTHONPATH=src:. python scripts/evaluate_continuous_motion_alignment.py \
  --manifest event78_manifest.jsonl \
  --scores raft_scores.jsonl \
  --reference-source logged_gt \
  --output continuous_motion.json
```

Evaluate true paired WAM counterfactuals only when both generated-image decoder
outputs and action-head trajectories are present:

```bash
PYTHONPATH=src:. python scripts/evaluate_counterfactual_continuous_alignment.py \
  --records wam_counterfactual_branches.jsonl \
  --require-eight-frame-four-second \
  --require-ready \
  --output counterfactual_continuous.json
```

The event-oriented commands below remain available as secondary interpretation
and legacy controls.

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

Build the opaque three-annotator risk-seed pack:

```bash
PYTHONPATH=src:. python scripts/build_blind_causal_seed_pack.py \
  --candidates work/nuplan_causal_candidates.jsonl \
  --output-dir work/causal_seed_blind_pack
```

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

Estimate and independently score actor-relative motion:

```bash
PYTHONPATH=src python scripts/estimate_actor_relative_motion.py \
  --tracks actor_tracks.jsonl --require-eight-frame-four-second \
  --output actor_relative_posteriors.jsonl
PYTHONPATH=src python scripts/evaluate_relative_motion_metrics.py \
  --records relative_motion_gold_pairs.jsonl --output relative_motion_metrics.json
```

Run the oracle-initialized CoTracker3 capability bound and render auditable overlays:

```bash
PYTHONPATH=src python scripts/evaluate_cotracker_actor_motion.py \
  --manifest actor_motion_reference_v2.jsonl --checkpoint /path/to/scaled_offline.pth \
  --output cotracker_oracle.json --device cuda
PYTHONPATH=src python scripts/render_cotracker_actor_motion.py \
  --manifest actor_motion_reference_v2.jsonl --report cotracker_oracle.json \
  --output-dir cotracker_overlays
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
The public/private split and annotation fields are defined in
[Causal Seed Annotation V1 (Chinese)](docs/CAUSAL_SEED_ANNOTATION_V1_ZH.md).
The actor-relative distance, closing-speed, TTC, and coverage-risk protocol is
specified in [Actor Relative Motion Capability V1
(Chinese)](docs/RELATIVE_MOTION_CAPABILITY_V1_ZH.md).
The first independent 40-record, four-chain, 8-frame/4-second reference-set
build is recorded in that document; it is NAVSIM sensor-backed NuPlan data,
not a Waymo result.

## Repository map

- `src/iac_new/`: image probe, event representation, metrics, readiness, and FUI
- `scripts/evaluate_continuous_decoder.py`: frozen image-to-motion decoder
- `scripts/evaluate_maneuver_events.py`: interval event recovery evaluation
- `scripts/audit_event_benchmark.py`: formal benchmark gate and scene split
- `scripts/evaluate_event_causal_metrics.py`: Event-CC, Event-FCS, and FUI
- `scripts/run_event_control_suite.py`: causal sanity-check controls
- `scripts/evaluate_causal_chains.py`: four interaction-chain readiness and scoring
- `scripts/mine_nuplan_causal_candidates.py`: nuPlan four-chain annotation candidate miner
- `scripts/build_blind_causal_seed_pack.py`: opaque four-chain risk-seed annotation pack
- `scripts/estimate_actor_relative_motion.py`: candidate-blind actor state posterior
- `scripts/evaluate_relative_motion_metrics.py`: metric accuracy and coverage-risk report
- `scripts/compare_flow_backends.py`: paired RAFT/SEA-RAFT decoder comparison
- `scripts/build_actor_motion_manifest.py`: build independent NuPlan actor-state reference set
- `scripts/audit_actor_motion_manifest.py`: fail-closed reference-set audit
- `scripts/evaluate_cotracker_actor_motion.py`: oracle actor tracker/geometry capability bound
- `scripts/render_cotracker_actor_motion.py`: representative and failure-case track overlays
- `configs/navsim_continuous_decoder_plane.json`: reported default configuration
- `configs/causal_chain_v1.example.jsonl`: executable four-chain risk/clear example
- `tests/`: unit and protocol tests

Model checkpoints, datasets, generated videos, and experiment outputs are not
stored in Git.
The first same-split V5 speed enhancement is documented in [Level 1 longitudinal
V5 result](docs/LEVEL1_LONGITUDINAL_V5_CURVATURE_20260827_ZH.md). It adds a
low-frequency second-difference penalty to the history-anchored residual while
preserving the V4 configuration as a separate baseline.
