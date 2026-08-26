import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.analyze_wam_action_sensitivity import analyze


class AnalyzeWamActionSensitivityTest(unittest.TestCase):
    def test_measures_image_response_without_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = [], []
            for index in range(2):
                a = root / f"a{index}.png"
                b = root / f"b{index}.png"
                cv2.imwrite(str(a), np.zeros((8, 8, 3), dtype=np.uint8))
                cv2.imwrite(str(b), np.full((8, 8, 3), 255, dtype=np.uint8))
                first.append(str(a))
                second.append(str(b))
            report = analyze([
                {"counterfactual_group_id": "g", "branch_id": "a", "future_images": first, "action_trajectory": [[1, 0, 0], [2, 0, 0]]},
                {"counterfactual_group_id": "g", "branch_id": "b", "future_images": second, "action_trajectory": [[1, 1, 0], [2, 1, 0]]},
            ])
            self.assertEqual(report["pairs"], 1)
            self.assertGreater(report["mean_future_image_l1"], 0.9)
            self.assertGreater(report["mean_normalized_action_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
