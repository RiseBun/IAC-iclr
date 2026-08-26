import unittest

import numpy as np

from iac_new.wam_metrics import (
    counterfactual_response_alignment,
    foresight_conditioned_success,
    paired_counterfactual_consistency,
    ego_state_trajectory,
    ego_state_action_compatibility,
    ego_state_response_alignment,
    realized_state_counterfactual_consistency,
)


def _traj(lateral: float, speed: float = 2.0) -> np.ndarray:
    return np.asarray([
        [speed * 0.5, lateral * 0.5, 0.0],
        [speed, lateral, 0.0],
        [speed * 1.5, lateral * 1.5, 0.0],
    ])


class WamMetricsTest(unittest.TestCase):
    def test_ego_state_reference_is_calibration_free(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5])
        states = ego_state_trajectory(_traj(0.5), times)
        self.assertEqual(states.shape, (3, 5))
        self.assertAlmostEqual(ego_state_action_compatibility(_traj(0.5), _traj(0.5), times), 1.0)

    def test_ego_state_response_rejects_reversed_action(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5])
        aligned = ego_state_response_alignment(_traj(0.0), _traj(1.0), _traj(0.0), _traj(1.0), times)
        reversed_pair = ego_state_response_alignment(_traj(0.0), _traj(1.0), _traj(1.0), _traj(0.0), times)
        self.assertGreater(aligned["alignment_score"], 0.9)
        self.assertEqual(reversed_pair["alignment_score"], 0.0)

    def test_realized_state_metric_is_high_for_matching_logged_future(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5])
        result = realized_state_counterfactual_consistency(
            _traj(0.0), _traj(1.0), _traj(0.0), _traj(1.0), times
        )
        self.assertGreater(result["realized_state_counterfactual_consistency"], 0.9)

    def test_aligned_pair_scores_above_reversed_pair(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5])
        aligned = counterfactual_response_alignment(
            _traj(0.0), _traj(1.0), _traj(0.0), _traj(1.0), times
        )
        reversed_pair = counterfactual_response_alignment(
            _traj(0.0), _traj(1.0), _traj(1.0), _traj(0.0), times
        )
        self.assertGreater(aligned["alignment_score"], 0.9)
        self.assertEqual(reversed_pair["alignment_score"], 0.0)

    def test_pair_consistency_contains_compatibility_and_response(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5])
        result = paired_counterfactual_consistency(
            {"imagined_future": _traj(0.0), "executed_action": _traj(0.0)},
            {"imagined_future": _traj(1.0), "executed_action": _traj(1.0)},
            times,
        )
        self.assertGreater(result["counterfactual_consistency"], 0.9)

    def test_fcs_is_conditioned_on_compatibility(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5])
        result = foresight_conditioned_success(
            [
                {"imagined_future": _traj(0.0), "executed_action": _traj(0.0), "task_success": True},
                {"imagined_future": _traj(1.0), "executed_action": _traj(0.0), "task_success": False},
            ],
            future_times_s=times,
            allow_action_proxy=True,
        )
        self.assertEqual(result["foresight_conditioned_success"], 1.0)
        self.assertGreater(result["success_lift"], 0.0)

    def test_fcs_requires_realized_state_by_default(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5])
        with self.assertRaises(ValueError):
            foresight_conditioned_success(
                [{"imagined_future": _traj(0.0), "executed_action": _traj(0.0), "task_success": True}],
                future_times_s=times,
            )
        result = foresight_conditioned_success(
            [{"imagined_future": _traj(0.0), "realized_future": _traj(0.0), "task_success": True}],
            future_times_s=times,
        )
        self.assertEqual(result["reference_kind"], "realized_state")


if __name__ == "__main__":
    unittest.main()
