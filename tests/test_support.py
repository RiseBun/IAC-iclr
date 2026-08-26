import unittest

import numpy as np

from iac_new.support import (
    acceptable_set_metrics,
    classify_trajectory_candidates,
    counterfactual_consistency,
    independent_support_mask,
)


class IndependentSupportTest(unittest.TestCase):
    def test_support_is_kinematic_and_score_independent(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        gt = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        near = gt.copy()
        near[:, 1] = np.linspace(0.0, 0.2, 4)
        far = gt.copy()
        far[:, 1] = np.linspace(0.0, 1.0, 4)
        candidates = [
            {"candidate_id": "gt", "trajectory": gt},
            {"candidate_id": "near", "trajectory": near},
            {"candidate_id": "far", "trajectory": far},
        ]
        mask, metadata = independent_support_mask(candidates, "gt", times)
        self.assertEqual(mask.tolist(), [True, True, False])
        self.assertEqual(metadata["definition"], "logged_trajectory_kinematic_tube")

    def test_consistency_is_posterior_mass_inside_support(self) -> None:
        result = counterfactual_consistency(
            np.asarray([0.6, 0.3, 0.1]), np.asarray([True, True, False])
        )
        self.assertAlmostEqual(result["support_mass"], 0.9)
        self.assertAlmostEqual(result["outside_support_mass"], 0.1)

    def test_multi_modal_labels_do_not_treat_unknown_as_invalid(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        gt = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        near = gt.copy()
        near[:, 1] = np.linspace(0.0, 0.4, 4)
        offroad = gt.copy()
        offroad[:, 1] = 4.0
        candidates = [
            {"candidate_id": "gt", "trajectory": gt},
            {"candidate_id": "near", "trajectory": near},
            {"candidate_id": "offroad", "trajectory": offroad, "collision": True},
        ]
        labels, meta = classify_trajectory_candidates(candidates, "gt", times)
        self.assertEqual([row["label"] for row in labels], ["known_valid", "plausible", "known_invalid"])
        result = acceptable_set_metrics(np.asarray([0.2, 0.7, 0.1]), labels)
        self.assertAlmostEqual(result["acceptable_mass"], 0.9)
        self.assertAlmostEqual(result["known_invalid_mass"], 0.1)
        self.assertEqual(meta["label_counts"]["plausible"], 1)


if __name__ == "__main__":
    unittest.main()
