# Best-method provenance

This file separates results that use different datasets and evaluation gates.
They must not be compared as if they were produced on one common split.

## Balanced event recovery (primary event result)

- Result: 78 samples / 312 intervals, lateral accuracy 0.977564 and macro-F1 0.974382.
- Exact config: `configs/navsim_continuous_decoder_plane.json`.
- Active image chain: RAFT-Large (32 updates), forward/backward consistency,
  fixed road ROI, calibrated camera geometry, ground-plane homography,
  candidate-blind continuous trajectory optimization, interval observability,
  maneuver extraction, and event posterior construction.
- Explicitly inactive: metric depth, SegFormer, DINOv2, temporal road filter,
  adaptive road plane, and speed in the primary score.

## SEA-RAFT matched flow ablation

- Split: the same balanced 78-sample manifest and decoder thresholds as the
  primary event result.
- RAFT-Large decoded 78/78; SEA-RAFT decoded 77/78.
- SEA-RAFT improved mean speed relative error from 0.307247 to 0.291543 and
  longitudinal diagnostic accuracy from 0.625000 to 0.672078.
- It regressed weighted joint error from 0.431435 to 0.646236, lateral MAE
  from 0.135017 m to 0.223629 m, and lateral event macro-F1 from 0.974382 to
  0.964305.
- Decision: RAFT-Large remains the frozen default. SEA-RAFT is a speed-side
  challenger, not the best overall backend.
- This 4-frame, approximately 2-second run does not validate actor-relative
  speed under the proposed 8-frame, 4-second protocol.

## Actor-relative state capability

- Protocol and implementation: `src/iac_new/relative_motion.py` and
  `docs/RELATIVE_MOTION_CAPABILITY_V1_ZH.md`.
- Implemented: calibrated ground-contact/depth projection, robust temporal
  fitting, distance/closing/lateral speed intervals, TTC, corridor entry,
  observability, abstention, and coverage-risk metrics.
- Evidence status: implementation complete and synthetically tested; no formal
  78-sample actor-ground-truth result has been claimed.

## Smooth-decoder ablation (different dense-curvature split)

- Selected setting: `smooth_010`.
- Exact config: `configs/navsim_continuous_decoder_plane_smooth.json`.
- Validation change versus its matched baseline: joint error 0.195451 to
  0.178157; soft compatibility 0.855745 to 0.867564; curvature error
  0.008480 to 0.007303 1/m.
- Added terms: speed smoothness 0.05, curvature smoothness 0.05, and lateral
  acceleration 0.02.
- Status: positive held-out validation ablation, not the config used by the
  balanced 78-sample event result.

## DINOv2 diagnostic

- Exact config: `configs/navsim_continuous_decoder_plane_dino_vits.json`.
- It improved Epona branch ranking on a separate WAM diagnostic, but did not
  establish a new default image-event recovery result.
- Status: optional diagnostic, not part of the primary event configuration.

The source tree contains the optional depth, semantic, temporal-road, and flow
reliability modules for continued research. A module being present does not
mean it was enabled for a reported metric.
