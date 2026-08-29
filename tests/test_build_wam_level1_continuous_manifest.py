import numpy as np

from scripts.build_wam_level1_continuous_manifest import build_manifest
from scripts.evaluate_continuous_motion_alignment import _level1_input_audit


def _base():
    return [{
        "sample_id": "base-1",
        "history_frame_paths": ["h0", "h1", "h2", "h3"],
        "future_frame_paths": ["old"] * 8,
        "future_times_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "candidates": [{"candidate_id": "logged", "prior": 1.0, "trajectory": np.zeros((8, 3)).tolist()}, {"candidate_id": "alt", "prior": 1.0, "trajectory": np.ones((8, 3)).tolist()}],
        "metadata": {"source_key": "source-1", "realized_future_ego_state": [[1.0]]},
    }]


def _generated():
    return [{
        "branch_id": "source-1::branch=logged",
        "counterfactual_group_id": "group-1",
        "branch_role": "clear",
        "history_fingerprint": "history-1",
        "nuisance_seed": 7,
        "source_key": "source-1",
        "wam_model_id": "test-wam",
        "future_images": [f"g{i}.png" for i in range(8)],
        "future_images_source": "wam_generated",
        "wam_generation_status": "complete",
        "future_times_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "action_condition": {"trajectory": np.full((8, 3), 2.0).tolist()},
    }]


def test_builds_generated_level1_row_without_realized_future_state():
    row = build_manifest(_base(), _generated())[0]
    assert row["future_images_source"] == "wam_generated"
    assert row["gt_candidate_id"] == "wam_action_head"
    assert len(row["future_frame_paths"]) == 8
    assert "realized_future_ego_state" not in row["metadata"]
    assert row["metadata"]["action_waypoint_used_by_image_branch"] is False
    assert row["action_trajectory_source"] == "wam_action_head"
    assert np.asarray(row["action_trajectory"]).shape == (8, 3)
    assert row["counterfactual_group_id"] == "group-1"
    assert row["branch_role"] == "clear"
    assert row["history_fingerprint"] == "history-1"
    assert row["nuisance_seed"] == 7
    audit = _level1_input_audit(row, "action")
    assert audit["ready"] is True


def test_rejects_pending_generation():
    generated = _generated()
    generated[0]["wam_generation_status"] = "pending"
    try:
        build_manifest(_base(), generated)
    except ValueError as error:
        assert "not complete" in str(error)
    else:
        raise AssertionError("expected pending generation to fail")
