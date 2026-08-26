import unittest

import numpy as np

from iac_new.event_metrics import (
    action_trajectory_event_target,
    event_counterfactual_matrix,
    event_foresight_conditioned_success,
    event_sequence_distance,
)


def _event(label, *, observability=1.0, abstain=False):
    labels = ("keep_lane", "turn_left", "turn_right")
    posterior = {item: 0.0 for item in labels}
    posterior[label] = 1.0
    return {
        "lateral_event": label,
        "lateral_posterior": posterior,
        "observability": observability,
        "abstain": abstain,
    }


class EventMetricsTest(unittest.TestCase):
    def test_action_target_is_independent_event_sequence(self):
        target = action_trajectory_event_target(
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.2, 0.06]]),
            np.asarray([0.5, 1.0]),
        )
        self.assertEqual(target[0]["lateral_event"], "keep_lane")
        self.assertEqual(target[1]["lateral_event"], "turn_left")

    def test_sequence_distance_supports_soft_posteriors(self):
        observed = [_event("keep_lane"), _event("turn_left")]
        target = [_event("keep_lane"), _event("turn_left")]
        result = event_sequence_distance(observed, target)
        self.assertEqual(result["distance"], 0.0)
        self.assertEqual(result["compatibility"], 1.0)
        self.assertEqual(result["coverage"], 1.0)

    def test_low_observability_reduces_coverage(self):
        result = event_sequence_distance(
            [_event("keep_lane", observability=0.1), _event("turn_left")],
            [_event("keep_lane"), _event("turn_left")],
        )
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["num_evaluable_intervals"], 1)

    def test_perfect_event_counterfactual_matrix(self):
        straight = [_event("keep_lane"), _event("keep_lane")]
        left = [_event("keep_lane"), _event("turn_left")]
        result = event_counterfactual_matrix([
            {
                "branch_id": "future_straight",
                "condition_action_id": "straight",
                "imagined_event_posterior": straight,
                "action_event_target": straight,
            },
            {
                "branch_id": "future_left",
                "condition_action_id": "left",
                "imagined_event_posterior": left,
                "action_event_target": left,
            },
        ])
        self.assertEqual(result["protocol"], "event-counterfactual-consistency-v1")
        self.assertEqual(result["diagonal_top1_accuracy"], 1.0)
        self.assertGreater(result["mean_cc_margin"], 0.0)
        self.assertGreater(result["cc_margin_lift_over_cyclic_swap"], 0.0)

    def test_swapped_event_futures_fail_counterfactual_matrix(self):
        straight = [_event("keep_lane"), _event("keep_lane")]
        left = [_event("keep_lane"), _event("turn_left")]
        result = event_counterfactual_matrix([
            {
                "condition_action_id": "straight",
                "imagined_event_posterior": left,
                "action_event_target": straight,
            },
            {
                "condition_action_id": "left",
                "imagined_event_posterior": straight,
                "action_event_target": left,
            },
        ])
        self.assertEqual(result["diagonal_top1_accuracy"], 0.0)
        self.assertLess(result["mean_cc_margin"], 0.0)

    def test_event_fcs_reports_coverage_and_joint_mass(self):
        straight = [_event("keep_lane"), _event("keep_lane")]
        left = [_event("keep_lane"), _event("turn_left")]
        result = event_foresight_conditioned_success([
            {
                "episode_id": "success",
                "imagined_event_posterior": straight,
                "realized_event_target": straight,
                "realized_event_source": "ego_state",
                "task_success": True,
            },
            {
                "episode_id": "mismatch",
                "imagined_event_posterior": straight,
                "realized_event_target": left,
                "realized_event_source": "simulator_state",
                "task_success": False,
            },
            {
                "episode_id": "ineligible",
                "imagined_event_posterior": straight,
                "realized_event_target": straight,
                "task_success": True,
            },
        ], compatibility_threshold=0.9)
        self.assertEqual(result["num_evaluable"], 2)
        self.assertEqual(result["num_compatible"], 1)
        self.assertEqual(result["foresight_conditioned_success"], 1.0)
        self.assertAlmostEqual(result["joint_fcs"], 1.0 / 3.0)
        self.assertAlmostEqual(result["evaluation_coverage"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
