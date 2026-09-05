# Script map

The release entrypoints are the following:

| Script | Scope |
|---|---|
| `validate_wam_submission.py` | fail-closed submission validation |
| `evaluate_continuous_decoder.py` | server-side Step 1 image probing |
| `evaluate_continuous_motion_alignment.py` | CFAC/FAU alignment against a joined record |
| `score_iac_submission.py` | capability-stratified scorecard generation |
| `audit_wam_level1_outputs.py` | WAM output schema and lineage audit |
| `audit_benchmark_manifest.py` | paired-manifest and split audit |
| `build_public_benchmark_manifest.py` | remove private paths and future GT |
| `build_wam_level1_continuous_manifest.py` | private evaluation join |

The remaining builders and adapters are compatibility utilities for constructing
private NAVSIM/Waymo inputs or reproducing historical pilots. They never ship
raw data and are not required to score a submission. Server-specific launch and
download wrappers are intentionally excluded. A script that requires a private
mount must fail with a missing-input error rather than silently using a logged
action or oracle state.
