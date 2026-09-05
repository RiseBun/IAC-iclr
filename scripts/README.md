# Script map

## Public evaluation entrypoints

These are the scripts needed to validate a submission and run the frozen
protocol once a private evaluation server has joined images and ground truth:

| Script | Role |
|---|---|
| `validate_wam_submission.py` | Fail-closed submission / leakage audit |
| `audit_wam_level1_outputs.py` | Accept 4- or 8-frame WAM futures before join |
| `build_wam_level1_continuous_manifest.py` | Join public identity with private frames |
| `evaluate_continuous_decoder.py` | Step 1 candidate-blind image probe |
| `evaluate_continuous_motion_alignment.py` | Align probed motion with native action / GT |
| `score_iac_submission.py` | Capability-stratified scorecard |
| `audit_benchmark_manifest.py` | Split / paired-manifest audit |
| `build_public_benchmark_manifest.py` | Strip private paths and future GT |
| `recompute_shape_cfac.py` | Shape-only CFAC aggregate helper |
| `analyze_ccfc_subscores.py` | CCFC direction / magnitude / temporal audit |
| `diagnose_cfac_ccfc_failures.py` | Longitudinal vs shape failure diagnosis |

Future-frame policy: history is usually 4 frames; generated futures may be
**4 or 8** points covering about 4.0 s. Pin with `--expected-future-count` when
needed.

## Model-specific / private-runner adapters

The following helpers reproduce the DriveWAM v3 pilot on a private NAVSIM mount.
They are **not** required to score an arbitrary WAM that already emits the
submission JSONL contract:

- `build_drivewam_inputs_v3.py`
- `build_drivewam_v3_level1_input.py`
- `build_drivewam_v3_ccfc_manifest.py`
- `build_drivewam_v3_missing_partition.py`
- `build_drivewam_fcs_staging.py`
- `prepare_drivewam_v3_reuse.py`
- `merge_drivewam_v3_outputs.py`
- `evaluate_v3_drivewam_cfac_fau.py`

Dataset construction utilities (`build_navsim_benchmark_v3.py`,
`export_navsim_records.py`, `prepare_waymo_level1_samples.py`,
`download_waymo_perception_v2.py`, …) likewise require licensed source data and
never ship raw frames.

A script that needs a private mount must fail with a missing-input error rather
than silently substituting a logged action or oracle state.
