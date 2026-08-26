# Image-Side Server Experiment

## Question

Can a lightweight, WAM-action-independent image-side probe recover the
trajectory mode supported by a NAVSIM front-camera clip?

## Splits

Use scene-disjoint splits, preferably the existing NAVSIM holdout first and a
new split of at least 100 scenes afterwards:

- calibration: temperature and conformal threshold only;
- validation: choose the default RAFT configuration;
- test: frozen once before inspecting results.

Do not tune thresholds on the 36-scene holdout after reading per-scene
results. The existing report calls that result exploratory.

## Configurations

| ID | Model | Purpose |
|---|---|---|
| S | RAFT-Small + ground-plane flow | speed lower bound |
| L | RAFT-Large + ground-plane flow | proposed default |
| L-FB | RAFT-Large + forward-backward mask | robustness ablation |
| D-L | existing UniDepthV2-L + RAFT-Large | accuracy upper bound |

The first three are implemented here. D-L is run with the existing IAC tools
and copied into the comparison table; it is intentionally not a dependency of
this package.

## Required metrics

Report all of these, not only top-1 accuracy:

- top-1 candidate accuracy;
- exact candidate negative log likelihood after temperature calibration;
- 90% conformal coverage;
- mean and median prediction-set size;
- invalid/abstained fraction;
- accuracy by forward scale, lateral displacement, yaw change, speed, and
  static-region pixel fraction;
- per-mode speed-range error and joint-region coverage for `(y, yaw, curvature)`;
- observability and dynamic-suppressed fraction for every one of the seven
  intervals, reported separately for history, real-to-generated boundary, and
  imagined-future transitions;
- wall-clock seconds per clip, clips per second, and peak CUDA memory.

## Promotion gate

The default L configuration is promoted only if, on the frozen test split:

1. top-1 accuracy is at least 90% on the existing two-candidate controls;
2. mirror-curvature and straightened-curvature controls are at least 90%;
3. calibrated 90% coverage is at least 88%;
4. invalid fraction is at most 5%;
5. L-FB does not improve coverage by more than 2 percentage points at a cost
   above 1.8x runtime, unless it also reduces invalid cases materially;
6. L is at least 2x faster or materially lower-memory than D-L while retaining
   the above safety thresholds.

The 10% speed perturbation should be reported but not used as a hard pass/fail
criterion until its separation exceeds the measured depth/flow noise floor.

## Failure diagnosis

Every invalid clip must be categorized as one of:

- too few valid static ROI pixels;
- homography leaves the image;
- forward-backward inconsistency;
- low optical-flow magnitude;
- non-planar road or strong camera pitch;
- dynamic-object contamination;
- missing or inconsistent calibration.

For every validation failure, inspect the generated `_image_detection.png`.
Reject the dynamic suppression design if it systematically removes road edges
or retains independently moving vehicles, even when aggregate accuracy rises.

The purpose of this experiment is to decide whether the image-side signal is
strong enough to justify a later WAM action intervention. It is not yet a
WAM leaderboard result.

## Dataset State Integration (2026-08-21)

The state protocol now supports canonical `[x, y, yaw, speed, yaw_rate]`
annotations from NuPlan, NAVSIM, and Waymo-style JSON exports. NuPlan was
validated against the existing closed-loop simulation logs: 5 annotations were
exported, with 2 native weighted-score successes and 3 failures at threshold
0.5. The adapter preserves the source log, score, state timestamps, and task
label.

The existing 16-row DrivingWorld WAM holdout was joined against those
annotations using explicit keys. The result was 0 matches: the old WAM rows do
not contain source scene/timestamp lineage. They must not receive NuPlan
states by image similarity or scene guesswork. Until WAM generation writes a
`source_key` pointing to a dataset annotation, realized-state CC and FCS remain
unavailable; action-conditioned CC remains valid.

The synthetic realized-state protocol example is only an API regression test;
it reports realized-state CC = 1.0 and FCS = 1.0 by construction.

The attachment step now also requires future timestamp agreement (60 ms
tolerance) whenever both the WAM manifest and the dataset annotation provide
`frame_times_s`/`candidate_times_s` and `state_times_s`. A key match with a
different future clock is rejected instead of silently scoring a misaligned
trajectory.

After the strict FCS change, the native 8-pair WAM holdout was rerun as
`artifacts/wam_holdout_state_v4.json`: action-response CC was 0.5272, ego-state
action CC was 0.4331, and A/B swap control was 0.0. Realized-state CC and FCS
were correctly `null`/unavailable because all 16 branches still lack explicit
dataset lineage and logged `task_success`; these numbers must not be presented
as true realized-state WAM scores.

The trajectory-image-only contract is now encoded as
`wam-trajectory-image-record-v1`. It requires `dataset`, `source_key`,
`scene_name`, `timestamp_us`, `history_ego_state`,
`realized_future_ego_state`, `history_images`, `future_images`,
`future_times_s`, and `trajectory`/`action_condition`. `task_success` is
optional and is not used by the current trajectory-image score. The server
audit of `examples/trajectory_image_record_example.jsonl` passed 1/1 records.

Native-dataset server test (NuPlan simulation root
`2026.03.31.21.30.01`) exported 5 source-linked annotations. Every row has
`history_ego_state` and `realized_future_ego_state` shaped `[4,5]`, finite
values, strictly increasing future times near 0.5/1.0/1.5/2.0 seconds, and
native task labels (2 successes, 3 failures). Joining these annotations to the
existing 16-row DrivingWorld holdout still matched 0 rows, confirming that the
old WAM history artifacts are not native NuPlan samples and cannot be scored
against these states without regenerating them with source lineage.

### Server trial 2026-08-21

The server trial reran the complete state-side path and the native WAM action
baseline. Protocol tests passed 12/12. NuPlan exported 5 annotations with 2
successes and 3 failures. The existing 8-pair WAM holdout produced action-
response CC `0.5272`, ego-state action CC `0.4331`, and A/B swap control `0.0`.
Realized-state CC and FCS remained unavailable, and the explicit join audit
reported 0/16 matches. This is a valid negative readiness result: the metric
does not silently treat generated WAM frames as native NuPlan observations.

### NAVSIM batch trial 2026-08-21

NAVSIM native scene pickles were inspected on the server. Each scene record
contains `CAM_F0` paths, timestamps, `ego2global` pose, `ego_dynamic_state`,
and camera intrinsics/extrinsics. The server currently exposes JPG blobs for
one mini log (`2021.10.06.17.43.07_veh-28_00508_00877`); the remaining 57
trainval state pickles have no matching local sensor blob and are therefore
reported as unavailable instead of being silently joined.

The new exporter produced 605 strict 4-history/4-future windows from the
available log. Each window has 45 candidates: the native realized trajectory
plus a deterministic speed/lateral/curvature counterfactual bank. Native
record audit passed 605/605 and the image-manifest contract passed 605/605.

The streaming RAFT-Small + ground-plane batch completed all 605 windows in
1675.4 s (0.361 clips/s, 340.6 MB peak CUDA memory). The corrected audit is
reported below because the first summary did not retain exception rows.

These numbers validate the large-batch NAVSIM image/state pipeline and show
that the lightweight baseline recovers a broad feasible support, but does not
yet identify the logged trajectory as a unique image-explained action. A
post-run audit found that the original streaming summary mixed two failure
classes: 136 returned rows with `valid=false`, plus 117 rows that raised a
low-effective-pixel exception before a result row was written. Thus the
correct total abstain count is 253/605 (41.8%); among the 352 valid rows,
native rank mean/median is 3.91/4 and Top-1 is 2.56%, while coverage remains
1.0. Internal abstains are dominated by low flow magnitude (128 rows,
counting rows rather than repeated interval reasons), with 8 domain-break
rows; all 117 exceptions are low-effective-pixel failures.

The future stream in this trial is `navsim_native_realized`, so it is an
oracle reference for protocol validation, not a WAM generation score. A WAM
run must replace only `future_images` using the same `source_key` and retained
realized state before realized-state CC/FCS can be reported.

Visualization semantics are now fixed: green translucent support is the
convex hull of the scored prediction set, red is the NAVSIM realized/GT
trajectory, blue is the selected Top-1 trajectory, and yellow is only the
optical-flow measurement ROI. Abstained rows have no selected trajectory.

### Quality-first semantic pilot (2026-08-22)

The evaluator now supports a conservative soft semantic gate. SegFormer road
pixels retain full weight, off-road pixels receive a lower weight, and pixels
classified as actors receive the lowest weight. This is deliberately not a
hard road mask: a segmentation miss must not turn a usable frame into an
automatic abstain. The gate is recorded in each result as
`perception.constraint_mode=soft` and with per-interval road/actor fractions.

An 8-window NAVSIM pilot was run with the same RAFT-Large, forward/backward,
ground-plane geometry and the same ROI:

| configuration | valid | abstain | GT coverage | Top-1 | mean GT rank | median GT rank |
|---|---:|---:|---:|---:|---:|---:|
| L + FB baseline | 8/8 | 0 | 1.00 | 0/8 | 2.625 | 3.0 |
| L + FB + soft SegFormer | 8/8 | 0 | 1.00 | 0/8 | 2.500 | 2.5 |

This is a ranking improvement, not a solved recognizer. The full 605-window
run is intentionally not promoted yet: the existing batch has 45 candidates,
and its 2.56% Top-1 on valid rows shows that support recall is good while
candidate identification is weak. The next quality gate is therefore a
calibration-split scorer ablation that separates speed, lateral displacement,
and curvature evidence, followed by a holdout run. No threshold will be
lowered merely to reduce abstention.

The server could load the cached SegFormer checkpoint, but its Hugging Face
metadata requests were repeatedly reset; the pilot completed from the cached
weights. There is still no metric-depth cache aligned to these NAVSIM native
windows, so UniDepth-L must not be attached to this batch by filename or
sequence position. A depth comparison is valid only after exporting a
source-key-aligned cache with matching intrinsics, camera transform, target
size, and interval count.

### Camera-contract correction and action-image matrix (2026-08-22)

The NAVSIM exporter previously dropped non-zero lens distortion while RAFT
operated on raw distorted frames and candidate flow used a pinhole camera. The
native distortion is large (one inspected camera starts near `k1=-0.356`), so
this was a scoring-contract error, not cosmetic metadata. The exporter and
manifest converter now retain distortion and an explicit composed
`camera_to_ego`; the image pipeline undistorts before RAFT, semantics, and
overlay rendering.

On the same first eight NAVSIM windows, corrected RAFT-Large + plane + FB moved
logged Top-1 from `0/8` to `3/8`, with mean rank improving from `2.625` to
`2.125`. Adding the old per-frame soft SegFormer gate after correction reduced
Top-1 to `1/8` and mean rank to `2.5`. Therefore the earlier semantic result is
not evidence for promoting that gate. A new temporal-road branch now aligns
adjacent road masks with RAFT and merges actor occlusions; it awaits a matched
server ablation.

The existing DrivingWorld holdout was also re-expressed as the agreed
future-image by action matrix. Eight same-history pairs (16 branches) were
cross-scored using the image-decoded ego trajectory against both branch
actions. Coverage was `16/16`, diagonal Top-1 `0.8125`, MRR `0.90625`, mean
probability CC margin `0.28909`, raw normalized-state energy margin `0.68408`,
and mean response TV `0.30452`. All heading interventions were correctly
matched. Three lateral groups had one failed branch, showing that small lateral
response remains the weak axis. These are diagnostic results because the CC
decision margin has not yet been frozen on a separate calibration split.

### LiDAR oracle and temporal-road A/B (2026-08-22)

The NAVSIM test oracle was run on 50 scene-diverse native windows. Five windows
had camera JPGs but no complete local LiDAR sequence and were explicitly marked
`missing_lidar`; the remaining 45 records and 180 future intervals were
evaluable. On ground points with forward/backward-consistent RAFT evidence, the
median EPE was `1.184 px`, p75 EPE `2.013 px`, direction cosine `0.9984`, and
observed/predicted flow scale `0.619`. The aggregate scale is misleading because
75/180 intervals had predicted median flow below 1 px. Restricting to the 102
intervals with predicted median flow at least 3 px gives direction cosine
`0.9986` and scale `0.984`, showing that RAFT direction and scale are reliable
when motion is observable. The evaluator should therefore abstain or lower the
speed claim for near-static intervals rather than replace RAFT with a larger
backbone.

The temporal-road ROI was compared against a strict broad-ROI control on the
same first 50 distortion-corrected NAVSIM windows. The broad control had 47
valid rows, 3 abstains, Top-1 `6.38%`, and mean rank `3.617`. Temporal-road had
39 valid rows, 11 total abstains, Top-1 `7.69%`, and mean rank `3.692`. Among 39
rows valid under both settings, temporal-road improved 9 ranks, left 18
unchanged, and worsened 12; mean rank delta was `+0.103` (worse). The branch
therefore remains an ablation, not the default ROI. It can improve individual
curved/occluded samples but currently removes too much evidence and does not
improve aggregate identification.

Oracle visualization uses green RAFT observed vectors and red pose+LiDAR rigid
vectors. The first smoke image and 50-row artifacts are stored under the server
`artifacts/navsim_oracle_flow_50_viz` and `artifacts/navsim_oracle_flow_50*`
paths. The batch exporter now supports `--require-lidar`, and the oracle audit
continues around missing point clouds instead of discarding the whole batch.

### Candidate-blind continuous decoder (2026-08-22)

The primary image-side decoder was changed from finite candidate retrieval to
continuous piecewise speed/curvature fitting. It sees only observed flow,
camera geometry, and optional semantic/depth reliability weights; the candidate
bank is used only after decoding for an external logged-trajectory comparison.
On 20 NAVSIM real-future windows, ground-plane decoding recovered direction and
road shape well (mean lateral error `0.034 m`, yaw error `0.0036 rad`, curvature
error `0.0085 1/m`) but had `40.2%` mean relative speed error and `0.375` joint
tolerance coverage. Cached UniDepth-L reduced speed error to `27.3%` and raised
joint coverage to `0.604`, while heading cosine remained near `0.99997`. This
supports using a continuous trajectory support tube for image-action
compatibility, but confirms that absolute speed/scale needs explicit uncertainty
and must not be treated as exact. These are native-image upper-bound diagnostics,
not WAM causal scores.

### WAM candidate-blind motion probe split report (2026-08-23)

The server's fixed Omega/RAFT motion probe was evaluated through
`scripts/aggregate_wam_probe_splits.py`. The probe emits `predicted_motion`
from generated future images before candidate actions are used. Candidate
actions enter only in the paired post-hoc CC and action-swap control.

| split | pairs | probe accuracy | action-image Top-1 | mean CC | swap control | CC lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | 12 | 0.9167 | 0.8333 | 0.4833 | 0.0205 | 0.4628 |
| holdout | 8 | 0.9375 | 0.8125 | 0.4038 | 0.0426 | 0.3612 |

This is positive action-response evidence under same-history reciprocal
interventions. It is not Foresight-Conditioned Success: generated frames
still have no independent camera calibration, realized future ego state, or
task-success label. The compact report is
`work_dirs/wam_probe_split_report.json`; the corrected matrix visualization is
`work_dirs/wam_holdout_best_native_matrix_v2.png`.

### Native NavSim realized-state image loop (2026-08-23)

`scripts/evaluate_native_realized_state.py` now runs the same candidate-blind
image decoder on native NavSim records. The decoder receives four history
images, camera calibration, and timestamps; it does not receive the logged
future state or use the candidate trajectory bank. The independent reference
is `realized_future_ego_state`, converted to the common `[x,y,yaw]`
trajectory representation only after decoding.

The first 100-window plane-geometry run completed with `100/100` scored and
`0` errors:

| metric | value |
| --- | ---: |
| mean realized-state compatibility | 0.5107 |
| median realized-state compatibility | 0.4963 |
| mean normalized ego-state distance | 0.7281 |

This is a positive validation of the independent realized-state comparison
path, not a claim that the image decoder is already production-accurate. The
manifest has zero explicit `task_success` labels, so FCS remains unavailable
by protocol. The complete machine-readable report is
`work_dirs/native_realized_state_decoder100_summary.json`.

As an image-side ablation, the same first 20 windows were rerun with the
lightweight SegFormer-B0 road/actor soft gate
(`configs/navsim_quality_v1_soft_semantic_plane.json`). Compatibility rose
from `0.5113` to `0.6150`, while normalized state distance fell from `0.7597`
to `0.6038`; both runs scored every window without errors. The 100-window
confirmation reversed that apparent gain: compatibility was `0.4705` versus
`0.5107` for the no-semantic baseline, and only `28%` of paired windows
improved. This is a cross-domain failure mode of direct Cityscapes semantic
masking, not evidence for replacing the baseline. The next implementation is
confidence-adaptive semantic gating with per-window fallback to the geometric
decoder before considering heavier depth or 3D foundation models. The complete
semantic report is `work_dirs/native_realized_state_segformer100_summary.json`.

The actor-only follow-up keeps geometric road support and applies SegFormer
only as a soft downweight on actor pixels
(`configs/navsim_actor_only_semantic_plane.json`). Its apparent all-100 mean
compatibility `0.5276` was not a holdout gain: using the first 20 windows for
calibration gives `0.6074` on calibration but only `0.5077` on the remaining 80,
versus `0.5105` for the geometric baseline. It is therefore not selected as
the default. Full road-mask gating remains disabled; semantic weights stay an
explicit calibration ablation. The raw report is
`work_dirs/native_realized_state_actor_only100_summary.json`.

### Explicit WAM state join audit (2026-08-23)

The current WAM holdout contains 16 branches in 8 reciprocal pairs and is
action-response ready, but has `0` history states, `0` realized future states,
and `0` task-success labels. The NuPlan annotation table has the required
states and labels, but shares no accepted identity key with these WAM rows.
The new `scripts/audit_state_join_keys.py` reports this as not FCS-ready and
rejects scene-name-only or image-similarity matches. A future WAM export must
carry one exact `source_key` (or `video_id`/`sample_id` plus timestamp) through
generation to make realized-state CC and FCS legal.

The evaluator exposes `--actor-weight` and `--disable-road-mask` overrides for
calibration sweeps without changing the input manifest. The selected values
must be frozen on calibration data before a disjoint holdout is reported.
