import unittest

import numpy as np

from iac_new.action_image_matrix import (
    action_image_cross_matrix,
    decoded_intervention_delta_matrix,
    decoded_trajectory_cross_matrix,
)


def _branch(branch_id, condition, a_energy, b_energy, valid=True):
    return {
        "branch_id": branch_id,
        "condition_action_id": condition,
        "valid": valid,
        "abstain_reasons": [] if valid else ["low_flow_magnitude"],
        "candidate_scores": [
            {"candidate_id": "a", "energy": a_energy},
            {"candidate_id": "b", "energy": b_energy},
        ],
    }


class ActionImageMatrixTest(unittest.TestCase):
    def test_perfect_diagonal_has_positive_cc(self):
        result = action_image_cross_matrix([
            _branch("future_a", "a", 0.1, 2.0),
            _branch("future_b", "b", 2.0, 0.1),
        ])

        self.assertEqual(result["diagonal_top1_accuracy"], 1.0)
        self.assertEqual(result["mean_reciprocal_rank"], 1.0)
        self.assertGreater(result["mean_cc_margin"], 0.0)
        self.assertGreater(result["mean_pairwise_response_tv"], 0.0)

    def test_swapped_futures_have_negative_cc(self):
        result = action_image_cross_matrix([
            _branch("future_a", "a", 2.0, 0.1),
            _branch("future_b", "b", 0.1, 2.0),
        ])

        self.assertEqual(result["diagonal_top1_accuracy"], 0.0)
        self.assertLess(result["mean_cc_margin"], 0.0)

    def test_invalid_branch_is_abstained(self):
        result = action_image_cross_matrix([
            _branch("future_a", "a", 0.1, 2.0, valid=False),
            _branch("future_b", "b", 2.0, 0.1),
        ])

        self.assertEqual(result["num_abstain"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["branches"][0]["decision"], "abstain")

    def test_tied_actions_are_not_counted_as_top1(self):
        result = action_image_cross_matrix([
            _branch("future_a", "a", 1.0, 1.0),
            _branch("future_b", "b", 1.0, 1.0),
        ])

        self.assertEqual(result["diagonal_top1_accuracy"], 0.0)
        self.assertEqual(result["mean_reciprocal_rank"], 2.0 / 3.0)
        self.assertEqual(result["mean_cc_margin"], 0.0)
        self.assertEqual(result["mean_energy_margin"], 0.0)

    def test_missing_cross_score_is_rejected(self):
        branches = [
            _branch("future_a", "a", 0.1, 2.0),
            _branch("future_b", "b", 2.0, 0.1),
        ]
        branches[0]["candidate_scores"] = branches[0]["candidate_scores"][:1]

        with self.assertRaisesRegex(ValueError, "missing action scores"):
            action_image_cross_matrix(branches)

    def test_decoded_trajectory_builds_full_cross_matrix(self):
        times = np.asarray([0.5, 1.0])
        action_a = [[1.0, -0.2, -0.05], [2.0, -0.5, -0.1]]
        action_b = [[1.0, 0.2, 0.05], [2.0, 0.5, 0.1]]
        result = decoded_trajectory_cross_matrix([
            {"branch_id": "a", "imagined_future": action_a, "executed_action": action_a},
            {"branch_id": "b", "imagined_future": action_b, "executed_action": action_b},
        ], times)

        self.assertEqual(result["evidence_source"], "image_decoded_ego_trajectory")
        self.assertEqual(result["diagonal_top1_accuracy"], 1.0)
        self.assertEqual(len(result["energy_matrix"]), 2)
        self.assertEqual(len(result["energy_matrix"][0]), 2)

    def test_support_tube_accepts_nonmedian_consistent_action(self):
        times = np.asarray([0.5, 1.0])
        support = [
            {"time_s": 0.5, "x_m": {"q05": 0.8, "q50": 1.0, "q95": 1.2}, "y_m": {"q05": -0.2, "q50": 0.0, "q95": 0.2}, "yaw_rad": {"q05": -0.1, "q50": 0.0, "q95": 0.1}},
            {"time_s": 1.0, "x_m": {"q05": 1.8, "q50": 2.0, "q95": 2.2}, "y_m": {"q05": -0.3, "q50": 0.0, "q95": 0.3}, "yaw_rad": {"q05": -0.1, "q50": 0.0, "q95": 0.1}},
        ]
        action = [[1.15, 0.18, 0.08], [2.15, 0.25, 0.08]]
        result = decoded_trajectory_cross_matrix([
            {"branch_id": "a", "imagined_future": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "imagined_support": support, "executed_action": action},
            {"branch_id": "b", "imagined_future": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "imagined_support": support, "executed_action": [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]},
        ], times)
        self.assertTrue(result["support_aware"])
        self.assertEqual(result["energy_definition"], "normalized_ego_state_support_distance")
        self.assertEqual(result["energy_matrix"][0][0], 0.0)

    def test_intervention_delta_removes_shared_forward_scale(self):
        times = np.asarray([0.5, 1.0])
        branches = [
            {
                "branch_id": "logged",
                "imagined_future": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                "executed_action": [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "valid": True,
            },
            {
                "branch_id": "left",
                "imagined_future": [[1.0, 0.1, 0.02], [2.0, 0.2, 0.04]],
                "executed_action": [[10.0, 0.1, 0.02], [20.0, 0.2, 0.04]],
                "valid": True,
            },
        ]
        result = decoded_intervention_delta_matrix(branches, times)
        self.assertEqual(result["diagonal_top1_accuracy"], 1.0)
        self.assertTrue(result["absolute_scale_invariant"])


if __name__ == "__main__":
    unittest.main()
