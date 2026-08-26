# IAC RAFT Event-Causal V1

This directory is the focused development snapshot for the current IAC
mainline: RAFT-Large motion evidence, ego-frame continuous decoding,
interval-level maneuver/event posteriors, and paired action counterfactual
consistency scoring.

## Mainline

- Default configuration: `configs/navsim_continuous_decoder_plane.json`
- Flow evidence: RAFT-Large with forward/backward consistency
- Geometry: ground-plane ego-frame decoder
- Primary evidence: lateral position, heading, curvature, and event support
- Speed: diagnostic only; it is excluded from the primary score
- Event interface: `src/iac_new/maneuver.py` and `src/iac_new/event_posterior.py`
- Causal interface: `src/iac_new/action_image_matrix.py`

## Evaluation protocol

For a fixed history, evaluate multiple action-conditioned future-image
branches. Decode each branch to an event posterior `q_A(E)`, build the full
cross-branch matrix, and report diagonal Top-1, reciprocal rank, CC margin,
and intervention/control lift. Realized-state CC and FCS are reported only
when independent realized state and task-success fields are present.

## Snapshot provenance

The bundled result artifacts are the 78-sample balanced NAVSIM event run
generated on 2026-08-26. The source snapshot is copied from the server
`iac_new` mainline after the event-posterior implementation was updated.
Re-run the event evaluator after any source or threshold change before using
the bundled results as a final metric.

The directory intentionally excludes model checkpoints, caches, historical
work directories, and WAM-specific generated media. Those remain in the
parent `iac_new` tree.
