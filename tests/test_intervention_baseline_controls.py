import unittest

import numpy as np

from iac_new.action_image_matrix import decoded_intervention_delta_matrix
from scripts.analyze_intervention_baseline_controls import analyze


class InterventionBaselineControlsTest(unittest.TestCase):
    def test_reports_logged_and_alternate_baselines(self):
        t = np.asarray([0.5, 1.0])
        branches = [
            {"branch_id": "g::branch=logged", "imagined_future": [[1, 0, 0], [2, 0, 0]], "executed_action": [[1, 0, 0], [2, 0, 0]]},
            {"branch_id": "g::branch=left", "imagined_future": [[1, .1, .01], [2, .2, .02]], "executed_action": [[1, .1, .01], [2, .2, .02]]},
            {"branch_id": "g::branch=right", "imagined_future": [[1, -.1, -.01], [2, -.2, -.02]], "executed_action": [[1, -.1, -.01], [2, -.2, -.02]]},
        ]
        report = analyze({"groups": [{"branches": branches}]}, t)
        self.assertEqual(report["reports"]["logged"]["diagonal_top1_accuracy"], 1.0)
        self.assertIn("left", report["reports"])


if __name__ == "__main__":
    unittest.main()
