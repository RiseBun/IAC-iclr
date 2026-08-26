# Event-Causal WAM Metrics V1

## Scope

V1 evaluates lateral events only: `keep_lane`, `turn_left`, and `turn_right`.
The frozen image probe is the reproducible RAFT-Large event baseline. Lane
changes have no positive support in the current 78-sample set, and monocular
longitudinal state is not qualified for the primary metric.

## Independent evidence contract

For the same history, every group contains at least two materially different
action branches. Each branch must provide:

```json
{
  "branch_id": "scene-1::left",
  "condition_action_id": "left",
  "imagined_event_posterior": [],
  "action_event_target": [],
  "realized_event_target": [],
  "realized_event_source": "ego_state",
  "task_success": true
}
```

- `imagined_event_posterior` comes only from the frozen image probe applied to
  the WAM future images.
- `action_event_target` is derived only from the conditioned action trajectory.
- `realized_event_target` is derived from independent ego/simulator telemetry.
- `task_success` must be an explicit boolean. Missing values are not imputed.

The complete input is JSONL with one group per line:

```json
{
  "counterfactual_group_id": "scene-1",
  "scene_id": "scene-1",
  "history_id": "history-sha256",
  "generation_seed": 7,
  "branches": []
}
```

## Event distance

Aligned interval posteriors are compared with normalized Jensen-Shannon
divergence. Distances are weighted by image observability. Abstained intervals
and intervals below the frozen observability threshold do not contribute, and
their omission is exposed as coverage.

```text
d_event(q, r) in [0, 1]
compatibility(q, r) = 1 - d_event(q, r)
```

V1 uses fixed aligned future intervals. Timing-tolerant onset/peak/exit scoring
is a future protocol version and must be calibrated before holdout use.

## Event Counterfactual Consistency

For K actions under one fixed history:

```text
M[k,j] = d_event(image_posterior_k, action_target_j)
ECC[k] = p[k,k] - max(p[k,j], j != k)
```

The report includes diagonal Top-1, MRR, CC margin, chance lift, cyclic
action-swap margin lift, branch coverage, and interval coverage. A valid result
requires a shared history, unique action ids, and complete K-by-K cross scores.

## Event-Foresight-Conditioned Success

For the executed branch, event compatibility is computed against independently
realized events. At a threshold tau, report all of:

```text
FCS(tau)       = P(success | compatibility >= tau)
foresight_cov  = P(compatibility >= tau)
Joint-FCS      = P(success and compatibility >= tau)
success_lift   = FCS(tau) - P(success)
```

Reporting FCS without coverage is invalid because a system could abstain on
nearly every episode.

## Causal claim boundary

Event CC establishes `action -> imagined future`. Event-FCS establishes that a
compatible imagined event co-occurs with realized success. Neither alone proves
`imagined future -> selected action`.

That final direction requires an Imagined-Future Swap: hold history, action
proposals, planner identity, and planner nuisance randomness fixed; permute the
futures supplied to the planner; rerun action selection; and compare the change
with independent null resampling and action-label controls. Level 3 therefore
requires `planner_id`, `planner_baseline_run_id`, `planner_nuisance_seed`, and
auditable rerun ids. Null trials must include distinct generated-future ids and
their image-probe event posteriors. Without those records, FUI is
`not_computed`, never inferred from offline branches.

## CLI

```bash
PYTHONPATH=src:. python scripts/audit_event_benchmark.py \
  --groups counterfactual_event_groups.jsonl \
  --output readiness.json \
  --split-output scene_split.json \
  --require-level 2

PYTHONPATH=src:. python scripts/evaluate_event_causal_metrics.py \
  --groups counterfactual_event_groups.jsonl \
  --output event_causal_report.json

PYTHONPATH=src:. python scripts/run_event_control_suite.py \
  --groups counterfactual_event_groups.jsonl \
  --output event_control_report.json
```
