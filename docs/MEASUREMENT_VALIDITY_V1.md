# Measurement Validity V1

## Question

The 78-sample NAVSIM result validates event recovery on real reference clips.
It does not establish that the frozen probe correctly reads WAM-generated
videos. This protocol tests that missing link against blinded human judgment.

## Design

- Mix videos from all evaluated WAMs before assigning opaque item ids.
- Remove model identity, action condition, task outcome, source paths, future
  timing, and probe predictions from the annotator-facing package.
- Center-crop and re-encode every frame as metadata-free `960x540` PNG so
  native resolution and file format cannot reveal the WAM identity.
- Keep the decoding key separate until all annotations are frozen.
- Use at least three annotators per item. Annotators label each future interval
  as `keep_lane`, `turn_left`, `turn_right`, or unobservable.
- Do not resolve disagreements using the conditioned action. Intervals without
  the preregistered consensus fraction remain excluded and reduce coverage.
- Resample complete scenes, not intervals, for confidence intervals.

The annotation target is perceived ego maneuver in the generated images. It is
not whether the video looks realistic and not whether the intended action was
reasonable.

## Outputs

The scorer reports:

- human pairwise agreement and generalized kappa;
- probe accuracy, macro-F1, NLL, multiclass Brier score, and 10-bin ECE;
- coverage over human-consensus intervals and all intervals;
- confidence-ranked risk/coverage and AURC;
- scene-clustered 95% bootstrap intervals;
- per-WAM results after unblinding;
- Event-CC computed independently from human event posteriors and probe event
  posteriors, including group-level CC-margin correlation and error.

The final comparison is essential. Low human Event-CC means the WAM did not
produce visually action-responsive futures. High human Event-CC with low probe
Event-CC means the image measurement failed. These failure modes must not be
combined.

## Build the blind pack

```bash
PYTHONPATH=src python scripts/build_blind_event_annotation_pack.py \
  --source epona /path/to/epona_manifest.jsonl /path/to/epona_event_groups.jsonl \
  --source drivewam /path/to/drivewam_manifest.jsonl /path/to/drivewam_event_groups.jsonl \
  --output-dir /path/to/blind_pack
```

Give annotators only the generated `public/` directory. Keep `private/` under
the benchmark owner's control and do not distribute any source manifest.

## Score frozen annotations

```bash
PYTHONPATH=src python scripts/score_event_measurement_validity.py \
  --private-key /path/to/blind_pack/private/private_key.jsonl \
  --annotations /path/to/annotator_1.jsonl \
  --annotations /path/to/annotator_2.jsonl \
  --annotations /path/to/annotator_3.jsonl \
  --output /path/to/measurement_validity_report.json
```

## Claim boundary

Passing this protocol establishes measurement validity on the sampled WAM
domains. It does not establish `imagined future -> selected action`; that still
requires the Level-3 FUI planner intervention. Thresholds for a formal pass/fail
claim must be preregistered before unblinding and justified relative to the
human agreement ceiling.
