import unittest

import numpy as np

from iac_new.trajectory_decode import integrate_piecewise_controls
from scripts.build_level1_relative_stratified_manifest import classify_relative_stratum


def row_for(speeds: list[float], *, curvature: float = 0.0) -> dict:
    times = np.arange(1, len(speeds) + 1, dtype=np.float64) * 0.5
    trajectory = integrate_piecewise_controls(
        times,
        speeds_mps=np.asarray(speeds, dtype=np.float64),
        curvatures_1pm=np.full(len(speeds), curvature, dtype=np.float64),
    )
    return {
        "sample_id": "synthetic",
        "future_times_s": times.tolist(),
        "gt_candidate_id": "logged",
        "candidates": [{"candidate_id": "logged", "trajectory": trajectory.tolist()}],
    }


class RelativeStratificationTest(unittest.TestCase):
    def test_net_longitudinal_change_has_priority(self) -> None:
        self.assertEqual(classify_relative_stratum(row_for([4.0, 4.5, 5.0, 5.5, 6.0])), "acceleration")
        self.assertEqual(classify_relative_stratum(row_for([6.0, 5.5, 5.0, 4.5, 4.0])), "braking")

    def test_lateral_motion_is_used_when_speed_is_stable(self) -> None:
        self.assertEqual(classify_relative_stratum(row_for([5.0] * 5, curvature=0.03)), "lateral_turn")

    def test_constant_straight_motion_is_fallback(self) -> None:
        self.assertEqual(classify_relative_stratum(row_for([5.0] * 5)), "straight_cruise")


if __name__ == "__main__":
    unittest.main()
