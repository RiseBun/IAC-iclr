import unittest

from iac_new.scorecard import (
    build_model_scorecard,
    claimed_cells,
    frozen_pilot_scorecard,
    validate_submission,
)


def _public() -> dict:
    return {
        "sample_id": "navsim:demo:scene:1",
        "source_key": "navsim:demo:scene:1",
        "split": "benchmark_v1",
    }


def _row(**extra):
    times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    traj = [[float(i), 0.0, 0.0] for i in range(8)]
    row = {
        "sample_id": "navsim:demo:scene:1",
        "wam_model_id": "toy_wam",
        "capability": "native_action_conditioned",
        "future_images_source": "wam_generated",
        "future_images": [f"f{i}.png" for i in range(8)],
        "future_times_s": times,
        "action_trajectory": traj,
        "action_source": "native_action_head",
    }
    row.update(extra)
    return row


class ScorecardTest(unittest.TestCase):
    def test_epona_does_not_claim_ccfc(self) -> None:
        self.assertEqual(claimed_cells("externally_controlled_video"), ("a2f",))
        card = build_model_scorecard(model_id="epona", capability="externally_controlled_video")
        self.assertEqual(card["cells"]["ccfc"]["status"], "unavailable")
        self.assertEqual(card["cells"]["a2f"]["status"], "missing")

    def test_native_missing_cells_are_missing_not_zero(self) -> None:
        card = build_model_scorecard(model_id="x", capability="native_action_conditioned")
        self.assertEqual(card["cells"]["ccfc"]["status"], "unavailable")
        self.assertNotIn("score", card["cells"]["ccfc"])

    def test_submission_must_match_public_ids(self) -> None:
        report = validate_submission([_row()], [_public()])
        self.assertTrue(report["ready"])
        bad = validate_submission([_row(sample_id="unknown")], [_public()])
        self.assertFalse(bad["ready"])
        self.assertIn("sample_id_not_in_public_split", bad["issues"][0]["issues"])

    def test_logged_action_is_rejected_for_native_models(self) -> None:
        report = validate_submission([_row(action_source="logged")], [_public()])
        self.assertFalse(report["ready"])
        self.assertIn("action_source_is_not_native", report["issues"][0]["issues"])

    def test_future_leak_is_rejected(self) -> None:
        report = validate_submission(
            [_row(realized_future_ego_state=[[0, 0, 0, 0, 0]] * 8)],
            [_public()],
        )
        self.assertIn("realized_future_state_leakage", report["issues"][0]["issues"])

    def test_frozen_pilots_withdraw_unvalidated_metric_scores(self) -> None:
        board = frozen_pilot_scorecard()
        self.assertEqual(board["protocol"], "iac-scorecard-v1")
        self.assertIn("ccfc", board["main_columns"])
        self.assertEqual(board["models"][1]["cells"]["ccfc"]["status"], "missing")
        self.assertEqual(board["models"][0]["cells"]["ccfc"]["status"], "unavailable")

    def test_primary_score_excludes_unvalidated_longitudinal_scale(self) -> None:
        board = frozen_pilot_scorecard()
        policy = board["primary_score_policy"]
        self.assertEqual(policy["included_motion_fields"], ["lateral_speed_mps", "yaw_rate_radps", "curvature_1pm"])
        self.assertEqual(policy["excluded_motion_fields"], ["speed_mps", "acceleration_mps2"])
        self.assertTrue(policy["excluded_fields_are_diagnostic_only"])
        drivewam = board["models"][1]["cells"]
        self.assertNotIn("score", drivewam["cfac"])
        self.assertIn("legacy_diagnostic_score", drivewam["cfac"])


class ScorecardScriptTest(unittest.TestCase):
    def test_frozen_board_has_four_models(self) -> None:
        payload = frozen_pilot_scorecard()
        self.assertEqual(len(payload["models"]), 4)
        self.assertEqual(
            [row["model_id"] for row in payload["models"]],
            ["worlddrive_tadwm", "drivewam_navsim", "epona_nuplan", "driveva_navsim"],
        )


if __name__ == "__main__":
    unittest.main()
