# IAC Event-Causal Architecture

> **Status: secondary interpretation layer.** The current primary architecture
> compares continuous image-derived ego motion directly with action waypoint
> motion. See [Continuous Foresight-Action Alignment V1](CONTINUOUS_FORESIGHT_ALIGNMENT_V1_ZH.md).

## Objective

IAC is an image-side evidence model for WAM evaluation. Its output is not a
single metrically exact trajectory. It is a calibrated posterior over coarse
driving events and their time support. This makes the evaluation tolerant to
multiple reasonable trajectories while preserving a causal test.

## Pipeline

```text
history images + history ego state
              |
              v
  RAFT static-road motion + UniDepth-L auxiliary + road evidence
              |
              v
  ego-frame temporal road state / observability
              |
              v
  Event Posterior (per interval)
    lateral: straight | left | right
    lane: keep | change-left | change-right
    longitudinal: cruise | slow | stop
    timing: onset / peak / exit interval
              |
              v
  canonical action-event support
              |
  WAM: same history + action A/B/C -> future images
              |
              v
  cross-branch event support matrix
              |
              +--> image-action CC
              +--> swapped-action control / response lift
              +--> realized-state CC
              +--> FCS
```

## Direction evidence

The primary direction rule is applied in ego coordinates, not in pixel space:

```text
delta_yaw[t] = wrap(yaw[t] - yaw[t-1])
left         if delta_yaw[t] > +0.028 rad
right        if delta_yaw[t] < -0.028 rad
straight     otherwise
```

RAFT supplies temporal image evidence and the decoder supplies the ego-frame
motion state. Pixel-flow direction is diagnostic only because perspective,
camera mounting, and moving actors can reverse or rotate its apparent image
direction.

## Event posterior

For each interval the image probe emits probabilities, support intervals and
observability. A low-observability interval is `uncertain`/`abstain`, not a
negative action label. Speed remains diagnostic and does not contaminate the
main direction or curvature score.

```json
{
  "lateral": {"straight": 0.08, "left": 0.89, "right": 0.03},
  "lane": {"keep_lane": 0.94, "change_left": 0.04, "change_right": 0.02},
  "longitudinal": {"cruise": 0.71, "slow": 0.24, "stop": 0.05},
  "onset_time_s": {"q05": 0.8, "q50": 1.1, "q95": 1.6},
  "observability": {"near": 0.92, "mid": 0.78, "far": 0.55}
}
```

## Causal scoring

For a branch conditioned on event action `A`, let `q_A(E)` be the event
posterior from its generated future image. The diagonal support margin is:

```text
CC_event(A) = q_A(A) - max_{B != A} q_A(B)
```

For the same history with paired interventions `A` and `B`, compute the full
cross matrix `q_A(E_B)` and `q_B(E_A)`. Report diagonal top-1, mean reciprocal
rank, margin, and the lift over an action-swap control. This is the causal
part: the history and nuisance content are held fixed while the action is
changed.

The realized-state version compares the image event posterior with an
independent logged future ego-state event sequence. FCS is reported only when
`realized_future_ego_state` and `task_success` are present; missing fields are
an evaluation eligibility failure, not an assumed success.

## Why this is structurally different

1. It evaluates a set of semantically equivalent maneuvers, not one GT path.
2. It separates direction, lane, longitudinal and timing evidence instead of
   letting monocular speed error corrupt the turn score.
3. It uses a paired cross-branch matrix, so a model cannot receive credit by
   emitting the same plausible scene for every action.
4. It makes observability and abstention explicit and calibratable.

The existing RAFT + UniDepth-L image chain remains the evidence extractor. The
new contribution is the event-level causal interface and its paired scoring
protocol, which is lightweight and compatible with WAMs that expose different
native action representations.
