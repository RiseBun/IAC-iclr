import unittest

import numpy as np

from iac_new.continuous_motion import (
    compare_counterfactual_se2_consistency,
    compare_motion_profiles,
    compare_pose_profiles,
    history_only_motion_profile,
    image_motion_profile,
    trajectory_to_motion_profile,
)
from iac_new.trajectory_decode import integrate_piecewise_controls


def decoder(speeds: list[float], curvature: float = 0.0) -> dict:
    times = np.asarray([0.5, 1.0, 1.5, 2.0])
    trajectory = integrate_piecewise_controls(
        times,
        speeds_mps=np.asarray(speeds),
        curvatures_1pm=np.full(4, curvature),
    )
    return {
        "protocol": "candidate-blind-continuous-trajectory-v1",
        "trajectory": trajectory.tolist(),
        "profile_support": [
            {
                "x_m": {"q05": float(point[0] - 0.2), "q50": float(point[0]), "q95": float(point[0] + 0.2)},
                "y_m": {"q05": float(point[1] - 0.1), "q50": float(point[1]), "q95": float(point[1] + 0.1)},
                "yaw_rad": {"q05": float(point[2] - 0.02), "q50": float(point[2]), "q95": float(point[2] + 0.02)},
            }
            for point in trajectory
        ],
        "speed_support": [
            {"q05": speed - 0.4, "q50": speed, "q95": speed + 0.4,
             "observability": 0.8, "status": "usable"}
            for speed in speeds
        ],
    }


class ContinuousMotionTest(unittest.TestCase):
    def test_waypoints_become_motion_profile(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        value = decoder([4.0] * 4, curvature=0.02)
        imagined = image_motion_profile(value, times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(value["trajectory"], times, initial_speed_mps=4.0)
        result = compare_motion_profiles(imagined, action)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["metrics"]["yaw_rate_radps"]["mae"], 0.0, places=6)
        self.assertEqual(result["coverage"], 1.0)

    def test_history_profile_is_past_only(self) -> None:
        profile = history_only_motion_profile(
            [[-1.0, 0.0, 0.0, 3.0, 0.1], [0.0, 0.0, 0.0, 3.0, 0.1]],
            [0.5, 1.0],
        )
        self.assertEqual(profile["source"], "history_only_constant_speed_yaw_rate")
        np.testing.assert_allclose([row["speed_mps"] for row in profile["rows"]], [3.0, 3.0])
        np.testing.assert_allclose([row["yaw_rate_radps"] for row in profile["rows"]], [0.1, 0.1])

    def test_pose_comparison_reports_shape_metrics(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        value = decoder([4.0] * 4, curvature=0.01)
        imagined = image_motion_profile(value, times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(value["trajectory"], times, initial_speed_mps=4.0)
        result = compare_pose_profiles(imagined, action)
        self.assertEqual(result["status"], "ok")
        self.assertIn("se2_pose", result["metrics"])
        self.assertGreaterEqual(result["coverage"], 0.0)

    def test_counterfactual_shape_priority_downweights_forward_scale(self) -> None:
        def profile(x_values, y_values, heading_values, source):
            return {
                "source": source,
                "rows": [
                    {
                        "time_s": float(index + 1),
                        "progress_m": float(x),
                        "lateral_offset_m": float(y),
                        "heading_rad": float(h),
                        "shape_status": "usable",
                        "observability": 1.0,
                    }
                    for index, (x, y, h) in enumerate(zip(x_values, y_values, heading_values))
                ],
            }

        clear_image = profile([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], "image_only_candidate_blind_decoder")
        risk_image = profile([0.0, 3.0], [0.0, 0.4], [0.0, 0.1], "image_only_candidate_blind_decoder")
        clear_action = profile([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], "native_action")
        risk_action = profile([0.0, 1.5], [0.0, 0.4], [0.0, 0.1], "native_action")
        metric = compare_counterfactual_se2_consistency(clear_image, risk_image, clear_action, risk_action)
        shape_priority = compare_counterfactual_se2_consistency(
            clear_image, risk_image, clear_action, risk_action, scale_mode="shape_priority"
        )
        self.assertLess(
            shape_priority["metrics"]["translation_response"]["weighted_mae"],
            metric["metrics"]["translation_response"]["mae"],
        )
        self.assertEqual(shape_priority["scale_mode"], "shape_priority")
        self.assertAlmostEqual(shape_priority["normalization"]["longitudinal_weight"], 0.25)


if __name__ == "__main__":
    unittest.main()
