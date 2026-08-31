import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.score_action_response_gate import score_gate


def _png(path: Path, value: int) -> str:
    cv2.imwrite(str(path), np.full((8, 8, 3), value, dtype=np.uint8))
    return str(path)


class ScoreActionResponseGateTest(unittest.TestCase):
    def test_left_right_pass_when_images_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = [_png(root / "l0.png", 0), _png(root / "l1.png", 0)]
            right = [_png(root / "r0.png", 255), _png(root / "r1.png", 255)]
            report = score_gate([
                {
                    "counterfactual_group_id": "g",
                    "branch_mode": "left",
                    "future_images": left,
                    "future_images_source": "driveva_generated",
                    "action_trajectory": [[1.0, -1.0, 0.2], [2.0, -2.0, 0.4]],
                },
                {
                    "counterfactual_group_id": "g",
                    "branch_mode": "right",
                    "future_images": right,
                    "future_images_source": "driveva_generated",
                    "action_trajectory": [[1.0, 1.0, -0.2], [2.0, 2.0, -0.4]],
                },
            ])
            self.assertTrue(report["passed"])
            self.assertGreater(report["mean_left_right_image_l1"], 0.9)

    def test_identical_images_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = [_png(root / "l0.png", 10)]
            right = [_png(root / "r0.png", 10)]
            report = score_gate([
                {
                    "counterfactual_group_id": "g",
                    "branch_mode": "left",
                    "future_images": left,
                    "future_images_source": "wam_generated",
                    "action_trajectory": [[1.0, -1.0, 0.2]],
                },
                {
                    "counterfactual_group_id": "g",
                    "branch_mode": "right",
                    "future_images": right,
                    "future_images_source": "wam_generated",
                    "action_trajectory": [[1.0, 1.0, -0.2]],
                },
            ])
            self.assertFalse(report["passed"])

    def test_model_specific_generated_source_is_model_agnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = [_png(root / "l0.png", 0)]
            right = [_png(root / "r0.png", 255)]
            report = score_gate([
                {
                    "counterfactual_group_id": "g",
                    "branch_mode": "left",
                    "future_images": left,
                    "future_images_source": "worlddrive_generated",
                    "action_trajectory": [[1.0, -1.0, 0.2]],
                },
                {
                    "counterfactual_group_id": "g",
                    "branch_mode": "right",
                    "future_images": right,
                    "future_images_source": "worlddrive_generated",
                    "action_trajectory": [[1.0, 1.0, -0.2]],
                },
            ])
            self.assertTrue(report["passed"])
            self.assertFalse(report["issues"])

    def test_named_command_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for branch, value, lateral in (("command_0", 255, 2.0), ("command_2", 0, -2.0)):
                rows.append({
                    "counterfactual_group_id": "command-pair",
                    "branch_id": branch,
                    "future_images": [_png(root / f"{branch}.png", value)],
                    "future_images_source": "worlddrive_generated",
                    "action_trajectory": [[1.0, lateral, 0.2 * np.sign(lateral)]],
                })
            report = score_gate(
                rows, min_l1=0.005, branch_a="command_0", branch_b="command_2"
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["groups_with_pair"], 1)


if __name__ == "__main__":
    unittest.main()
