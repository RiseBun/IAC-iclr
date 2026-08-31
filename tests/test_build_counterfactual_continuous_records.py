import numpy as np

from scripts.build_counterfactual_continuous_records import build_records


def _row(role: str, sample_id: str, x: float):
    return {
        "sample_id": sample_id,
        "counterfactual_group_id": "group-1",
        "branch_role": role,
        "history_fingerprint": "history-1",
        "wam_model_id": "wam-1",
        "nuisance_seed": 4,
        "future_times_s": [0.5, 1.0],
        "action_trajectory": [[x, 0.0, 0.0], [2.0 * x, 0.0, 0.0]],
        "action_trajectory_source": "wam_action_head",
        "metadata": {"history_ego_state": [[0.0, 0.0, 0.0, 2.0]]},
    }


def _score(sample_id: str):
    return {
        "sample_id": sample_id,
        "candidate_bank_used_by_decoder": False,
        "decoder": {
            "protocol": "candidate-blind-continuous-trajectory-v1",
            "trajectory": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        },
    }


def test_assembles_clear_risk_pair():
    rows, summary = build_records(
        [_row("clear", "a", 1.0), _row("risk", "b", 0.8)],
        [_score("a"), _score("b")],
    )
    assert summary["groups"] == 1
    assert len(rows) == 2
    assert {row["branch_role"] for row in rows} == {"clear", "risk"}


def test_rejects_unpaired_group():
    try:
        build_records([_row("clear", "a", 1.0)], [_score("a")])
    except ValueError as error:
        assert "expected_exactly_clear_and_risk" in str(error)
    else:
        raise AssertionError("expected unpaired group to fail")


def test_assembles_custom_named_pair_without_relabeling():
    rows, summary = build_records(
        [_row("command_0", "a", 1.0), _row("command_2", "b", 0.8)],
        [_score("a"), _score("b")],
        role_a="command_0",
        role_b="command_2",
    )
    assert summary["pair_roles"] == ["command_0", "command_2"]
    assert {row["branch_role"] for row in rows} == {"command_0", "command_2"}
