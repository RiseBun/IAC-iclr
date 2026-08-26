import unittest

import numpy as np

from iac_new.maneuver import compare_maneuvers, extract_maneuver
from iac_new.trajectory_decode import integrate_piecewise_controls


class ManeuverTest(unittest.TestCase):
    def test_straight_then_left_is_speed_invariant(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        trajectory = integrate_piecewise_controls(
            times,
            speeds_mps=np.asarray([3.0, 3.0, 3.0, 3.0]),
            curvatures_1pm=np.asarray([0.0, 0.0, 0.04, 0.08]),
        )
        faster = integrate_piecewise_controls(
            times,
            speeds_mps=np.asarray([7.0, 7.0, 7.0, 7.0]),
            curvatures_1pm=np.asarray([0.0, 0.0, 0.04, 0.08]),
        )
        first = extract_maneuver(trajectory, times)
        second = extract_maneuver(faster, times)
        self.assertEqual(first["segment_types"], ["straight", "left"])
        self.assertEqual(second["segment_types"], ["straight", "left"])
        self.assertGreater(compare_maneuvers(first, second)["score"], 0.9)

    def test_left_right_mismatch_is_inconsistent(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        left = integrate_piecewise_controls(
            times, speeds_mps=np.full(4, 4.0), curvatures_1pm=np.full(4, 0.05)
        )
        right = integrate_piecewise_controls(
            times, speeds_mps=np.full(4, 4.0), curvatures_1pm=np.full(4, -0.05)
        )
        result = compare_maneuvers(extract_maneuver(left, times), extract_maneuver(right, times))
        self.assertEqual(result["score"], 0.0)

    def test_longitudinal_and_lane_change_events(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        trajectory = np.asarray([
            [0.0, 0.0, 0.0],
            [0.5, 0.2, 0.02],
            [1.5, 0.8, 0.04],
            [3.0, 1.1, 0.05],
        ])
        result = extract_maneuver(trajectory, times)
        self.assertEqual(result["longitudinal_action"][0], "stop")
        self.assertIn("lane_change_left", result["lane_change_action"])
        self.assertEqual(result["maneuver_class"], "lane_change_left")

    def test_scale_noise_does_not_create_a_turn(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        # Very short progress makes curvature per metre large even though the
        # observable heading barely changes.
        trajectory = np.asarray([
            [0.03, 0.0, 0.002],
            [0.06, 0.0, 0.005],
            [0.09, 0.0, 0.007],
            [0.12, 0.0, 0.010],
        ])
        result = extract_maneuver(trajectory, times)
        self.assertEqual(result["lateral_action"], ["straight"] * 4)


if __name__ == "__main__":
    unittest.main()
