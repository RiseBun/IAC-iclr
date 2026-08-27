import numpy as np

from scripts.audit_wam_level1_outputs import audit_rows


def _row():
    return {
        "branch_id": "s::branch=logged",
        "source_key": "s",
        "future_images_source": "wam_generated",
        "wam_generation_status": "complete",
        "wam_model_id": "wam-test",
        "future_images": [f"frame-{i}.png" for i in range(8)],
        "future_times_s": np.arange(1, 9, dtype=float) / 2.0,
        "action_condition": {"trajectory": np.zeros((8, 3)).tolist()},
    }


def test_accepts_complete_protocol_row():
    report = audit_rows([_row()])
    assert report["formal_level1_input_ready"] is True
    assert report["complete_rows"] == 1


def test_rejects_realized_future_leakage():
    row = _row()
    row["metadata"] = {"realized_future_ego_state": [[1.0]]}
    report = audit_rows([row])
    assert report["formal_level1_input_ready"] is False
    assert "realized_future_state_leakage" in report["issues"][0]["issues"]


def test_rejects_wrong_frame_count():
    row = _row()
    row["future_images"] = row["future_images"][:4]
    report = audit_rows([row])
    assert report["formal_level1_input_ready"] is False
    assert "future_images_must_have_8_paths" in report["issues"][0]["issues"]
