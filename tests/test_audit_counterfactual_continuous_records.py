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


def test_command_pair_is_structurally_ready_but_not_formal_foresight():
    first = _branch("command_0", [[1.0, 1.0, 0.1], [2.0, 2.0, 0.2]])
    second = _branch("command_2", [[1.0, -1.0, -0.1], [2.0, -2.0, -0.2]])
    for row in (first, second):
        row["intervention_type"] = "navigation_command_onehot"
    report = audit_records(
        [first, second], role_a="command_0", role_b="command_2"
    )
    assert report["structural_level2_input_ready"] is True
    assert report["formal_level2_input_ready"] is False
    assert report["claim_scope"] == "command_conditioned_action_image_consistency"


def test_internal_future_intervention_is_mediation_ready_but_not_semantic_hazard():
    rows = [
        _branch("future_native", [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        _branch("future_reverse", [[1.0, 0.2, 0.1], [2.0, 0.4, 0.2]]),
    ]
    for row in rows:
        row["intervention_type"] = "internal_future_latent_permutation"
    report = audit_records(rows, role_a="future_native", role_b="future_reverse")
    assert report["structural_level2_input_ready"] is True
    assert report["formal_foresight_input_ready"] is True
    assert report["semantic_hazard_input_ready"] is False
    assert report["claim_scope"] == "internal_foresight_mediation"
