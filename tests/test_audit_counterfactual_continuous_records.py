from scripts.audit_counterfactual_continuous_records import audit_records


def _branch(role: str, action):
    return {
        "counterfactual_group_id": "g-1",
        "branch_role": role,
        "history_fingerprint": "h-1",
        "wam_model_id": "wam-1",
        "nuisance_seed": 9,
        "future_images_source": "wam_generated",
        "action_trajectory_source": "native_action_head",
        "candidate_bank_used_by_decoder": False,
        "future_times_s": [0.5, 1.0],
        "action_trajectory": action,
    }


def test_audit_accepts_complete_pair():
    report = audit_records([
        _branch("clear", [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        _branch("risk", [[0.8, 0.0, 0.0], [1.5, 0.0, 0.0]]),
    ])
    assert report["formal_level2_input_ready"] is True


def test_audit_reports_missing_pair_without_throwing():
    report = audit_records([_branch("clear", [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])])
    assert report["formal_level2_input_ready"] is False
    assert report["groups_detail"][0]["issues"] == ["expected_exactly_one_clear_and_one_risk"]
