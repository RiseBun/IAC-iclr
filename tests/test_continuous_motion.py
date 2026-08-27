import unittest

import numpy as np

from iac_new.continuous_motion import (
    compare_counterfactual_motion_deltas,
    compare_motion_profiles,
    image_motion_profile,
    trajectory_to_motion_profile,
)
from iac_new.trajectory_decode import integrate_piecewise_controls
from scripts.evaluate_counterfactual_continuous_alignment import audit_pair


def decoder(speeds: list[float], *, curvature: float = 0.0) -> dict:
    times = np.asarray([0.5, 1.0, 1.5, 2.0])
    trajectory = integrate_piecewise_controls(
        times,
        speeds_mps=np.asarray(speeds),
        curvatures_1pm=np.full(4, curvature),
    )
    return {
        "protocol": "candidate-blind-continuous-trajectory-v1",
        "trajectory": trajectory.tolist(),
        "speed_support": [
            {"q05": speed - 0.4, "q50": speed, "q95": speed + 0.4, "observability": 0.8, "status": "usable"}
            for speed in speeds
        ],
    }


class ContinuousMotionTest(unittest.TestCase):
    def test_waypoints_become_interval_motion(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        trajectory = integrate_piecewise_controls(
            times, speeds_mps=np.full(4, 4.0), curvatures_1pm=np.zeros(4)
        )
        profile = trajectory_to_motion_profile(trajectory, times, initial_speed_mps=4.0)
        np.testing.assert_allclose([row["speed_mps"] for row in profile["rows"]], 4.0)
        np.testing.assert_allclose([row["acceleration_mps2"] for row in profile["rows"]], 0.0)
        np.testing.assert_allclose([row["lateral_speed_mps"] for row in profile["rows"]], 0.0)

    def test_action_never_enters_image_profile(self) -> None:
        profile = image_motion_profile(decoder([4.0] * 4), [0.5, 1.0, 1.5, 2.0], initial_speed_mps=4.0)
        self.assertEqual(profile["source"], "image_only_candidate_blind_decoder")
        self.assertFalse(profile["candidate_bank_used"])
        self.assertNotIn("action_trajectory", profile)

    def test_continuous_comparison_scores_magnitude(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        imagined = image_motion_profile(decoder([4.0] * 4), times, initial_speed_mps=4.0)
        action_trajectory = integrate_piecewise_controls(
            np.asarray(times), speeds_mps=np.full(4, 5.0), curvatures_1pm=np.zeros(4)
        )
        action = trajectory_to_motion_profile(action_trajectory, times, initial_speed_mps=4.0)
        result = compare_motion_profiles(imagined, action)
        self.assertAlmostEqual(result["metrics"]["speed_mps"]["mae"], 1.0)
        self.assertEqual(result["coverage"], 1.0)
        self.assertFalse(result["leakage_audit"]["action_waypoint_visible_to_image_decoder"])

    def test_abstained_intervals_do_not_count_as_failures(self) -> None:
        value = decoder([4.0] * 4)
        value["speed_support"][0]["status"] = "abstain"
        imagined = image_motion_profile(value, [0.5, 1.0, 1.5, 2.0])
        action = trajectory_to_motion_profile(value["trajectory"], [0.5, 1.0, 1.5, 2.0])
        result = compare_motion_profiles(imagined, action)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(result["metrics"]["speed_mps"]["count"], 3)

    def test_counterfactual_delta_checks_response_not_labels(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        clear_image = image_motion_profile(decoder([5.0] * 4), times, initial_speed_mps=5.0)
        risk_image = image_motion_profile(decoder([4.5, 4.0, 3.5, 3.0]), times, initial_speed_mps=5.0)
        clear_action = trajectory_to_motion_profile(decoder([5.0] * 4)["trajectory"], times, initial_speed_mps=5.0)
        risk_action = trajectory_to_motion_profile(decoder([4.5, 4.0, 3.5, 3.0])["trajectory"], times, initial_speed_mps=5.0)
        result = compare_counterfactual_motion_deltas(clear_image, risk_image, clear_action, risk_action)
        self.assertEqual(result["metrics"]["speed_mps"]["sign_agreement"], 1.0)
        self.assertAlmostEqual(result["metrics"]["speed_mps"]["delta_mae"], 0.0)

    def test_counterfactual_readiness_is_fail_closed(self) -> None:
        base = {
            "history_fingerprint": "history-1",
            "wam_model_id": "wam-1",
            "nuisance_seed": 7,
            "future_images_source": "wam_generated",
            "action_trajectory_source": "native_action_head",
            "candidate_bank_used_by_decoder": False,
            "future_times_s": [0.5, 1.0],
        }
        clear = dict(base, action_trajectory=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        risk = dict(base, action_trajectory=[[0.8, 0.0, 0.0], [1.5, 0.0, 0.0]])
        self.assertEqual(audit_pair("group-1", {"clear": clear, "risk": risk}), [])
        del risk["nuisance_seed"]
        self.assertIn("missing_nuisance_seed", audit_pair("group-1", {"clear": clear, "risk": risk}))


if __name__ == "__main__":
    unittest.main()
