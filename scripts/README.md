# IAC New: Lightweight Image-Side Probe

This is the first, deliberately incomplete stage of the new IAC direction. It
only answers:

> Given a short front-camera clip, can an independent image-side model recover
> which feasible ego trajectory the clip supports, with calibrated uncertainty?

It does **not** train or evaluate a WAM action head yet. It does not claim
closed-loop success. The output is a candidate-bank posterior and a conformal
prediction set that can later be used by the WAM counterfactual protocol.

## Default model

```text
calibrated per-frame metric depth + known camera calibration
        -> candidate rigid-flow prediction
RAFT-Large observed optical flow
        -> shared dynamic suppression + robust ROI residual E(video, trajectory)
        -> calibrated posterior q(trajectory | video)
```

The accurate default is cached UniDepth-L plus RAFT-Large. The evaluator reads
validated depth caches but does not import UniDepth, so the metric core remains
small and the depth extractor can run offline. UniDepth-S is the lightweight
branch; the plane backend is retained only as a no-depth baseline. Per-frame
depth estimation is not scene reconstruction or SLAM.

## 8-frame contract

The canonical input is four real history frames followed by four imagined
future frames. The last history frame is the anchor; a candidate contains one
`[x_m, y_m, yaw_rad]` knot for each future frame.

```json
{
  "sample_id": "scene__timestamp",
  "scene_id": "scene",
  "history_frame_paths": ["history_0.jpg", "history_1.jpg", "history_2.jpg", "anchor.jpg"],
  "history_times_s": [-1.5, -1.0, -0.5, 0.0],
  "future_frame_paths": ["future_1.jpg", "future_2.jpg", "future_3.jpg", "future_4.jpg"],
  "future_times_s": [0.5, 1.0, 1.5, 2.0],
  "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion": [],
  "camera_to_ego": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
  "metric_depth_path": "/path/to/depth/video_000.npz",
  "metric_depth_source": "unidepth-v2-vitl14",
  "gt_candidate_id": "logged",
  "candidates": [
    {"candidate_id": "logged", "prior": 1.0,
     "trajectory": [[3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [9.0, 0.0, 0.0], [12.0, 0.0, 0.0]]},
    {"candidate_id": "perturbed", "prior": 1.0,
     "trajectory": [[3.0, 0.3, 0.02], [6.0, 0.6, 0.05], [9.0, 1.1, 0.09], [12.0, 1.8, 0.14]]}
  ]
}
```

Legacy `frame_paths` anchor-plus-future rows remain accepted only for the
existing exploratory material. Formal experiments must use the explicit 4+4
contract.

The evaluator owns the candidate bank. A WAM action or score must never be
used to construct it. The bank is only an approximation to a continuous
trajectory space in this first stage.

To reuse the existing ViPE/rigid-flow material, convert its manifest and attach
the explicit depth cache path:

```bash
python scripts/convert_legacy_vipe_manifest.py \
  --legacy-manifest /path/to/vipe_holdout36/manifest.jsonl \
  --depth-cache-dir /path/to/depth_cache \
  --candidate-bank structured \
  --output /path/to/iac_new_holdout36.jsonl
```

The structured bank contains the logged trajectory, `0.8/0.9/1.1/1.2x`
forward-scale controls, and curvature controls for clips where curvature is
observable. Cache shape, intrinsics, extrinsics, and image size are checked
before scoring; metadata mismatch invalidates the sample.

## Server validation

Run from the repository root on Linux after installing the package:

```bash
cd iac_new
python -m pip install -e .
python -m iac_new.evaluate \
  --manifest /path/to/navsim_calibration.jsonl \
  --config configs/navsim_raft_large_unidepth_l_cached.json \
  --output /path/to/work/depth_l_raft_large.jsonl \
  --visualization-dir /path/to/work/trajectory_regions \
  --device cuda
```

Each output row contains:

- a posterior and calibrated prediction set over trajectory modes;
- each mode's full `(x, y, yaw, motion direction, speed, curvature)` states and
  speed range;
- a weighted discrete joint support over lateral position, yaw, and curvature;
- per-interval observability for all seven image transitions;
- future-interval dynamic suppression statistics and an ego-frame trajectory
  support plot. This is not a full BEV perception output.

With `--visualization-dir`, a metric-depth sample produces three audit images:

- `<sample>.png`: ego-frame trajectory modes, joint region, heading arrows,
  speed ranges, and interval observability. It plots hypotheses in vehicle
  coordinates; it is not a reconstructed BEV map;
- `<sample>_image_detection.png`: the seven source-image intervals with the
  road ROI, retained/suppressed pixel weights, observed optical flow, and the
  top mode's predicted rigid flow. Green is retained static evidence, amber is
  suppressed/uncertain evidence, cyan is observed flow, and orange is
  predicted rigid flow. Suppressed pixels are not automatically obstacles.
- `<sample>_metric_depth.png`: metric-depth color, final common scoring support,
  median depth, confidence, and support coverage for each future interval.
- `<sample>_trajectory_overlay.png`: the front image with the semantic
  drivable prior, dynamic-object contours, and selected trajectory hypotheses
  projected back into the camera. This is the primary trajectory visualization;
  the rigid-flow image is only a diagnostic.

The history panels are explicitly labeled `ROI/FB reliability` because no
future candidate is allowed to provide their dynamic weights. The boundary
and three imagined-future panels use the shared rigid-static weight.

The marginal min/max values in `trajectory_region` are summaries only. The
feasible region is the list of coupled `joint_support` tuples, not the Cartesian
product of those independent bounds.

For a continuous-support pilot, expand each reference trajectory into a
deterministic counterfactual bank before evaluation:

```bash
PYTHONPATH=src python scripts/densify_counterfactual_manifest.py \
  --manifest /path/to/manifest.jsonl \
  --output /path/to/manifest_dense.jsonl
python -m iac_new.evaluate \
  --manifest /path/to/manifest_dense.jsonl \
  --config configs/navsim_raft_large_unidepth_l_cached.json \
  --output /path/to/dense_scores.jsonl
PYTHONPATH=src python scripts/summarize_dense_counterfactual.py \
  --scores /path/to/dense_scores.jsonl
```

The default bank adds 45 smooth speed/lateral/curvature perturbations (duplicates of
the original reference are removed). Each result now contains weighted
`trajectory_region.continuous_support.knotwise_quantiles` in addition to the
joint candidate support. This is an empirical feasible cloud, not a learned
world model or a reconstructed scene.

## Joint WAM metrics

### Exact data flow and metric roles

The final interface separates the WAM condition from the image evidence:

```text
history images + history ego state + condition trajectory/action A
    -> WAM -> generated future images I_A
    -> image probe g(history, I_A, camera contract) -> posterior/support over trajectories
    -> compare the decoded support with A (and with independent realized state, when available)
```

The condition trajectory is never passed to `g` as an observation. It is used
only after decoding to ask whether the generated images support the requested
action. A single branch can therefore measure image-action compatibility, but
cannot establish causality: a model that emits the same future for every action
could still obtain a superficially plausible score. Causal CC requires at least
two branches with the same history and task:

```text
A -> I_A, B -> I_B
score(g(I_A), A) versus score(g(I_A), B)
score(g(I_B), B) versus score(g(I_B), A)
```

With a multi-modal support, the matched action is accepted when it lies in the
decoded acceptable set; it need not be the unique Top-1 mode. We report both
`acceptable_mass` and the swapped-action control. `realized_future_ego_state`
is an independent dataset reference and is never fed into the image probe.

### Candidate-blind continuous decoder

For the primary image-action score, use
`scripts/evaluate_continuous_decoder.py`. It fits piecewise speed and
curvature controls directly to observed flow and projects the resulting
continuous trajectory through the camera model. The candidate bank is hidden
from the decoder and is used only afterward for an independent comparison to
the logged trajectory. The output includes a local profile support tube plus
observability-weighted lateral, yaw, speed, curvature, and joint-coverage
metrics. This prevents a finite candidate list from turning the image probe
into a retrieval test.

For WAM outputs with no generated-camera calibration, use
`scripts/evaluate_wam_image_plane.py`. It applies the same candidate-blind
principle to RAFT flow in a fixed image-plane ROI and emits a normalized motion
proxy. On the 15 available DrivingWorld reciprocal pairs, relative speed
ordering was correct on `5/5` speed pairs. Signed lateral/yaw consistency is
marked provisional because camera-axis sign and metric scale require either
camera calibration or an independent ego-state reference. The server output
is `work_dirs/wam_image_plane_full_v3.json`.

The stronger candidate-blind Omega motion probe is summarized with
`scripts/aggregate_wam_probe_splits.py`. On the independent DrivingWorld
holdout it reaches `15/16` candidate selections (`93.75%`), action-image
diagonal Top-1 `0.8125`, and mean CC lift over the same-history action-swap
control `0.3612`. The calibration split is `11/12` (`91.67%`) with CC lift
`0.4628`. These are positive causal-response results, but still not FCS:
the generated frames have no independent realized future ego state or task
success label.

For native NavSim image histories and future frames, use
`scripts/evaluate_native_realized_state.py`. It decodes the future trajectory
from images only, then compares it against the separately logged
`realized_future_ego_state`; the logged state is never passed to the decoder
and the candidate bank is disabled. On the first 100 native windows with the
plane geometry baseline, the run completed with zero errors and obtained mean
realized-state compatibility `0.5107` (median `0.4963`) and mean normalized
state distance `0.7281`. This validates the independent state-side evaluation
path, but also shows that the current image decoder is not accurate enough for
strict FCS claims. Native NavSim records in this batch contain no explicit
`task_success`, so FCS is correctly reported as unavailable rather than
fabricated from a proxy.

On the first 20 windows, adding the lightweight SegFormer-B0 semantic gate
raised compatibility to `0.6150`, but the 100-window confirmation fell to
`0.4705` versus `0.5107` for the no-semantic baseline; only `28%` of paired
windows improved. The 20-window gain was therefore a sample effect, not a
model upgrade. Semantic masks must be confidence-gated with a baseline
fallback before they are allowed to affect the score.

A safer actor-only variant keeps geometric road support and applies SegFormer
only as a soft downweight on detected actor pixels. Its apparent 100-window
mean of `0.5276` was not a valid holdout gain: with the first 20 windows used
for calibration, actor-only scored `0.6074` on calibration but `0.5077` on the
remaining 80, versus `0.5105` for the geometric baseline. It is therefore not
selected as the default; full road-mask gating remains disabled and semantic
weights remain an explicitly calibrated ablation.

The explicit join audit confirms why FCS is not yet reportable for the current
WAM holdout: all 16 branches have action/image fields, but zero history state,
zero realized future state, and zero task labels. The NuPlan annotation table
cannot be attached because it shares no accepted identity key. The audit now
fails closed instead of allowing a scene-name or image-similarity join.

The evaluator exposes `--actor-weight` and `--disable-road-mask` overrides for
calibration sweeps without editing the manifest. These values must be frozen
on calibration data before reporting a disjoint holdout score.

```bash
PYTHONPATH=src python scripts/evaluate_native_realized_state.py \
  --manifest /path/to/navsim_native.jsonl \
  --config configs/navsim_continuous_decoder_plane.json \
  --output /path/to/native_realized_state.jsonl --device cuda
```

The image-side posterior is only a measurement component. It is not itself a
WAM causal metric. A valid paired WAM record fixes the history and task, then
contains at least two branches where the imagined future is intervened on and
the executed action is recorded separately:

```text
(same history, task, future condition A) -> imagined future A -> action A
(same history, task, future condition B) -> imagined future B -> action B
```

For the lightweight action-conditioned benchmark, cross-score every generated
future against the complete action bank from the same history. This produces a
`K x K` matrix whose rows are generated futures and columns are action
hypotheses. The diagonal must beat every swapped action; equal scores are
treated as indistinguishable, not as a Top-1 success.

```bash
PYTHONPATH=src python scripts/analyze_action_image_matrix.py \
  --manifest /path/to/wam_branches.jsonl \
  --scores /path/to/image_probe_scores.jsonl \
  --output /path/to/action_image_matrix.json
```

The report retains raw energy and row-normalized probability matrices, diagonal
Top-1, reciprocal rank, matched-vs-best-counterfactual CC margin, raw energy
margin, response total variation, and abstention coverage. A discrete
`supported/mismatched` decision is emitted only when `--decision-margin` comes
from a held-out calibration split. Existing paired reports with an
image-decoded ego trajectory can be upgraded with
`scripts/add_action_image_matrix_to_analysis.py`; those rows explicitly record
`evidence_source=image_decoded_ego_trajectory`.

`src/iac_new/wam_metrics.py` implements three auditable quantities:

- `ForesightActionCompatibility`: `exp(-d(future, action))`, where `d` is a
  normalized joint distance over lateral position, yaw, speed, and curvature;
- `CounterfactualResponseAlignment`: whether the action change points in the
  same direction as the imagined-future intervention, with zero score for an
  unchanged action;
- `ForesightConditionedSuccess`: task success conditioned on compatibility
  exceeding a frozen threshold, plus unconditional success and success lift.

Their operational definitions are:

* `Counterfactual Consistency (CC)`: for same-history interventions A and B,
  the generated image under A should support A more than B, and vice versa.
  In the discrete matrix this is the diagonal-versus-best-swap margin. In the
  set version it is the acceptable posterior mass plus a paired swapped-action
  control. A positive result means the action changes explain the image change;
  it is not a pixel-quality score.
* `Foresight-Conditioned Success (FCS)`: among branches whose decoded future
  is compatible with the independently logged realized future (or an explicitly
  declared realized action reference), compute the task-success rate:
  `FCS = P(task_success=1 | compatibility >= tau)`. Report unconditional
  success and `FCS - P(task_success=1)` as the lift. `tau` must be fixed on a
  calibration split. Without an independent realized future and task label,
  FCS is unavailable, not zero.

The calibration-free state protocol represents both the image-decoded future
and action condition as `[x, y, yaw, speed, yaw_rate]`. Camera calibration is
therefore optional for the causal score and remains required only for metric
projection back onto pixels. Inputs and references have distinct roles:

```text
history images + history ego state + action intervention -> WAM future images
future ego state (when available)                       -> evaluation only
```

Future ego state must never be fed to the image-motion probe. If a manifest
contains only the action condition, the report measures action-conditioned
state response; it cannot claim realized execution fidelity or closed-loop
success.

Dataset adapters live in `src/iac_new/state_protocol.py`:

- NuPlan `ego_pose` exports (`x/y`, quaternion, velocity, yaw rate);
- NAVSIM scene records (`ego2global`, `ego_dynamic_state`);
- Waymo-style lightweight exports (`pose` or `x/y/heading`, velocity).

All adapters emit the same `[T,5]` state array. An annotation row must be
joined to a WAM row by an explicit source key; scene names or image similarity
are not accepted as implicit matches. The NuPlan closed-loop exporter reads
the native simulation log and weighted aggregate score:

```bash
PYTHONPATH=src python scripts/export_nuplan_simulation_states.py \
  --simulation-root /path/to/nuplan/closed_loop_run \
  --output /path/to/nuplan_annotations.jsonl
PYTHONPATH=src python scripts/attach_dataset_states.py \
  --manifest /path/to/wam_manifest.jsonl \
  --annotations /path/to/nuplan_annotations.jsonl \
  --dataset nuplan \
  --output /path/to/wam_with_states.jsonl
```

The exporter uses an explicit score threshold for `task_success`; the
threshold and source are written into every row. This is a dataset task label,
not a visual similarity heuristic.

For NAVSIM and Waymo, export native records to JSONL and normalize them with
`scripts/export_json_state_annotations.py`. A row contains `records`, a
`history_count`, and an explicit WAM join key. NAVSIM may additionally carry a
native `pdm_score`; Waymo task success must be supplied explicitly because the
dataset does not define one universal downstream task-success label.

Run the paired protocol with:

```bash
PYTHONPATH=src python scripts/analyze_wam_pairs.py \
  --pairs /path/to/wam_pairs.jsonl \
  --output /path/to/wam_metrics.json
```

Audit a proposed pair file before scoring:

```bash
PYTHONPATH=src python scripts/audit_wam_pairs.py \
  --pairs /path/to/wam_pairs.jsonl
```

The audit exits nonzero unless every row has a pair identity, at least two
branches, imagined futures, executed actions, and task-success labels.

Audit the lighter calibration-free state protocol separately:

```bash
PYTHONPATH=src python scripts/audit_wam_state_protocol.py \
  --manifest /path/to/wam_holdout.jsonl
```

This reports action-response readiness separately from closed-loop readiness.
`history_ego_state` documents the WAM input; `realized_future_ego_state` is an
independent evaluation reference and is never supplied to the image probe.

Existing IAC/NAVSIM rows have no same-history counterfactual branches, so they
can calibrate the image probe but cannot produce a causal Counterfactual
Consistency score. A real benchmark must provide paired future interventions
or an approved action-conditioned simulator/data generator.

The server also contains a completed DrivingWorld reciprocal WAM run. Its
`wam_calibration` and `wam_holdout` manifests contain two branches per
`twin_id`, conditioned candidate motion, and generated future images. The
existing motion-probe output supplies an independent motion estimate for each
generated future. Audit these native WAM pairs with:

```bash
PYTHONPATH=src python scripts/analyze_wam_native_pairs.py \
  --manifest /path/to/wam_holdout.jsonl \
  --scores /path/to/wam_holdout_scores.jsonl \
  --output /path/to/wam_native_analysis.json
```

The adapter reports ego-state paired CC, an A/B action-swap control,
condition-switch accuracy, and per-intervention-family results. True FCS is
reported only when each included branch has an independently logged
`realized_future_ego_state` and an explicit `task_success`; otherwise it is
`unavailable`. The conditioned candidate motion is the intervention action and
the independent motion-probe estimate is the decoded imagined future. The old
WAM manifests therefore remain a causal response audit, with
`state_reference.source=action_condition` and
`independent_realized_future_ego_state=false`, rather than a closed-loop policy
success benchmark. The low-level function exposes
`allow_action_proxy=True` only for backward-compatible diagnostics, and marks
those results as `reference_kind=action_proxy`.

### Trajectory-image record schema

The strict record used by the trajectory-image consistency experiment is
`wam-trajectory-image-record-v1`. It requires the dataset provenance and state
fields below, plus the image/trajectory signals needed by the metric:

```text
dataset
source_key
scene_name
timestamp_us
history_ego_state              # [H,5]: x, y, yaw, speed, yaw_rate
realized_future_ego_state      # [T,5], independent reference
history_images                 # H paths
future_images                  # T generated future-image paths
future_times_s                 # T increasing timestamps
trajectory or action_condition # [T,3]: x, y, yaw
```

`task_success` is optional and is not used by the trajectory-image
consistency score. It is only needed by the separate FCS extension. Validate a
JSONL file before evaluation with:

```bash
PYTHONPATH=src python scripts/audit_trajectory_image_records.py \
  --records /path/to/records.jsonl \
  --output /path/to/record_audit.json
```

Dynamic suppression adds no semantic network. It downweights pixels that none
of the feasible rigid-flow candidates can explain, and applies the same weights
to every candidate. This is a lightweight proxy for dynamic/non-planar content;
it must be ablated against the disabled setting on the validation split.

The optional SegFormer branch reports semantic actor pixels separately from the
geometric motion proxy. `future_actor_motion` contains, per interval, the
actor fraction, the fraction of actor pixels with low rigid-flow weight, and a
diagnostic classification. A `car` class is not treated as moving merely
because it is a car; motion requires the residual evidence as well.

### WAM calibration audit

Native WAM reciprocal manifests currently contain rendered image paths and
action/probe motion, but no `intrinsics` or `camera_to_ego`. Audit them before
attempting image projection:

```bash
PYTHONPATH=src python scripts/audit_wam_calibration.py \
  --manifest /path/to/wam_holdout.jsonl \
  --output /path/to/wam_holdout_calibration_audited.jsonl
```

The script emits `complete`, `partial`, `missing`, or `invalid` calibration
counts. Only `complete` rows are allowed to use `metric_ego` projection; all
others are explicitly marked `image_plane_only`. A calibration index may be
provided with `--calibration-index` and is matched by `video_id`, `twin_id`,
`sample_id`, or source image path. The index should be populated from the
original NuPlan/Waymo camera records, not inferred from the rendered pixels.

## Optional semantic model

Install the optional perception dependencies when a Cityscapes SegFormer prior
is desired:

```bash
python -m pip install -e '.[perception]'
```

Use `configs/navsim_raft_large_unidepth_l_segformer.json`. SegFormer supplies a
candidate-independent road prior and coarse actor mask. Each candidate receives
a `semantic_feasibility.traversable_fraction` for its projected vehicle-width
corridor; this is reported alongside, rather than silently replacing, the
depth+flow causal evidence. The actor mask means vehicle/person class, not
proven motion: static and moving actors require temporal tracking. If the model
is unavailable, use the cached metric-depth config without the `perception`
block.

`configs/navsim_quality_v2_temporal_road_plane.json` is the experimental
dynamic-road ROI. SegFormer processes all eight frames, RAFT aligns the next
road mask back to the current frame, and only temporally agreed road evidence
survives. Actor masks from both frames are merged as occlusions. The configured
polygon is only a broad field-of-view bound; it is no longer presented as the
road. This branch must pass a scene-matched baseline ablation before it becomes
the default.

Fit calibration on a scene-disjoint calibration split, then rerun evaluation:

```bash
python -m iac_new.calibrate \
  --scores /path/to/work/depth_l_raft_large_calibration.jsonl \
  --coverage 0.90 \
  --output /path/to/work/depth_l_raft_large_calibration.json

python -m iac_new.evaluate \
  --manifest /path/to/navsim_scene_disjoint_test.jsonl \
  --config configs/navsim_raft_large_unidepth_l_cached.json \
  --calibration /path/to/work/depth_l_raft_large_calibration.json \
  --output /path/to/work/depth_l_raft_large_test.jsonl \
  --device cuda
```

Analyze rank stability and image-conditioned trajectory consistency only on an
independent validation/test split:

```bash
PYTHONPATH=src python scripts/analyze_dense_posterior.py \
  --scores /path/to/calibrated_dense_scores.jsonl \
  --manifest /path/to/dense_manifest.jsonl \
  --calibration /path/to/calibration.json \
  --output /path/to/dense_analysis.json
```

The third dense axis is a constant curvature increment: yaw perturbation grows
with traveled arc length. Rank monotonicity is computed on one-axis-at-a-time
slices. Counterfactual consistency is posterior mass inside a kinematic tube
around the logged trajectory, defined without image-flow energy or posterior.
The report includes 0.5x/1.0x/1.5x tolerance sensitivity. This is an
image-conditioned trajectory-support precursor, not yet full WAM causal
Counterfactual Consistency.

The frozen test-50 integration result is 98% for UniDepth-L + RAFT-Large with
shared dynamic suppression, 94% with suppression disabled, and 90% for
UniDepth-S + RAFT-Large. See `SERVER_PILOT_20260820.md` for the split and
earlier baselines. Forward-backward consistency remains diagnostic only.

## Non-goals for this stage

- no V-JEPA or semantic acceptability ranker;
- no WAM action decoder;
- no counterfactual success claim;
- no dense scene reconstruction or SLAM map;
- no automatic camera calibration when dataset calibration is available.

Online depth inference is intentionally outside this first integration. Cached
depth makes the geometry contract testable before adding model-loading and
deployment complexity.
