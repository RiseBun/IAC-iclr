# IAC progress: offline semantic ablation and DriveWAM assets

## What was verified

- SegFormer-B0 Cityscapes loads with `HF_HUB_OFFLINE=1` from the server cache.
- The primary 100-window NAVSIM baseline remains `0.5106745` realized-state compatibility.
- A fixed soft semantic weight applied to all controls is not safe: the 100-window result was `0.4730862`.
- A two-stage shape-only variant (geometric speed first, SegFormer only for curvature refinement) reduced the large regressions, but still measured `0.5027336` on the same 100 windows. It remains an ablation, not the default.
- The 100-window paired shape-only delta was `-0.00794`; 81 rows were unchanged, 13 improved, and 6 regressed. The regressions are concentrated in a few extreme-curvature samples.

## Current default

The benchmark default stays candidate-blind RAFT-Large + forward/backward consistency + ground-plane geometry + dynamic suppression + conservative continuous decoder. Speed is diagnostic and uncertainty-labeled, not part of the primary score. SegFormer is offline-capable but must be reported as an explicit ablation.

## Next quality work

1. Add curvature/turn observability diagnostics and a targeted abstain reason for extreme-curvature or low-support intervals.
2. Evaluate improvements on a calibration split, then freeze thresholds before holdout.
3. Run the same decoder on generated WAM future-image branches only after branch completeness is verified; keep IAC failure and weak WAM response as separate diagnoses.

## Curvature observability update

- The old first-100-window slice covered only three NAVSIM scenes, so it was not valid for selecting a general threshold.
- A new scene-balanced manifest has 95 windows: five temporally separated windows from each of 19 scenes. All 95 decode successfully with realized-state compatibility `0.6099077`.
- The decoder now records left/right static-flow angular contrast, temporal turn change, and support for every future interval without changing the primary score.
- On scene-disjoint validation, low left/right angular contrast predicts high curvature reconstruction error strongly (four evaluable folds: holdout AUC `0.94` to `1.00`). The evidence is useful, but the absolute error distribution differs by scene; it must remain a calibrated continuous uncertainty signal until a fixed physical error threshold is validated.
- `scripts/analyze_curvature_observability.py` is the reproducible calibration/holdout analysis. It assigns complete scenes to balanced folds and never selects the contrast threshold on the held-out scenes.
- With a fixed physical target of `0.01 1/m` curvature error, the four evaluable holdout folds still give mean AUC `0.973` and balanced accuracy `0.904`. The implementation therefore exposes `curvature_confidence` per interval; it is diagnostic/weighting metadata only and does not alter the decoded trajectory or primary compatibility score.

## CoTracker3 and benchmark engineering update

- The official CoTracker3 scaled-offline checkpoint is expected under `$IAC_MODEL_ROOT/cotracker3/scaled_offline.pth` (SHA256 `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`).
- A 95-row, 19-scene matched A/B gives RAFT low-contrast AUC `0.983` versus CoTracker low-evidence AUC `0.579`. CoTracker has valid tracks on average for `33.2%` of sampled road points. It remains diagnostic-only and is not fused into the default decoder.
- A fail-closed benchmark manifest audit and versioned metric card are now available. The 95 native single-branch records are valid for IAC image-to-trajectory capability (`0.6099` compatibility, 100% image-probe coverage), but correctly fail the counterfactual-pair and FCS gates because they contain no same-history branch pairs and no task-success labels.

## Two-gate WAM diagnosis

- Existing 15-frame DrivingWorld image-plane experiment: IAC gate passes with mean heading cosine/compatibility `0.999953` and mean lateral error `0.4592 m` under the `0.5 m` gate.
- The WAM future-response gate fails: action-image distance correlation `0.07734` and mean response ratio `0.09740`, both below the frozen `0.20` diagnostic thresholds.
- The defensible current conclusion is therefore `wam_future_response_weak`, not `iac_model_insufficient`. This is an image-response gate, not yet realized-state CC or FCS; the generated branches still need independent realized state and task-success joins for those claims.

## DriveWAM assets

The stable registry is `$IAC_MODEL_REGISTRY/registry.json`. Both official DriveWAM checkpoints are now available in the internal experiment environment:

- `drivewam_navsim`: 12,482,164,484 bytes
- `drivewam_physicalai`: 12,482,072,236 bytes

The DriveWAM code repository and its checkpoint variants are recorded separately. The LingBot-VA base dependency is not silently treated as installed; runtime integration still needs that base model and its documented configuration.
