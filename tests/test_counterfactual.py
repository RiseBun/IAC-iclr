import unittest

import numpy as np

from iac_new.counterfactual import dense_counterfactual_trajectories, densify_record


class CounterfactualTest(unittest.TestCase):
    def test_dense_bank_is_smooth_and_deterministic(self) -> None:
        reference = np.asarray(
            [[2.0, 0.0, 0.0], [4.0, 0.1, 0.01], [6.0, 0.2, 0.02], [8.0, 0.3, 0.03]]
        )
        first = dense_counterfactual_trajectories(reference)
        second = dense_counterfactual_trajectories(reference)
        self.assertEqual(len(first), 45)
        self.assertEqual([item[0] for item in first], [item[0] for item in second])
        self.assertTrue(np.allclose(first[0][1], second[0][1]))
        # The anchor remains unchanged; perturbation grows with future progress.
        self.assertTrue(np.allclose(first[0][1][0, 1:], reference[0, 1:]))
        self.assertLess(abs(first[0][1][1, 1] - reference[1, 1]), abs(first[0][1][-1, 1] - reference[-1, 1]))
        self.assertIn("curvature_offset_1pm", first[0][2])

    def test_densify_record_keeps_label_and_adds_metadata(self) -> None:
        row = {
            "sample_id": "s",
            "gt_candidate_id": "logged",
            "candidates": [
                {"candidate_id": "logged", "trajectory": [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]], "prior": 1.0},
            ],
            "metadata": {},
        }
        dense = densify_record(row)
        self.assertEqual(dense["gt_candidate_id"], "logged")
        self.assertEqual(len(dense["candidates"]), 45)
        self.assertEqual(dense["metadata"]["counterfactual_bank"]["num_generated"], 44)
        self.assertIn("counterfactual", dense["candidates"][1])
        self.assertEqual(dense["candidates"][0]["feasibility_label"], "known_valid")
        self.assertEqual(dense["candidates"][1]["feasibility_label"], "plausible")


if __name__ == "__main__":
    unittest.main()
