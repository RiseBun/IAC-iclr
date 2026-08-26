import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_drivingworld_native_input import prepare_rows


class PrepareDrivingWorldInputTest(unittest.TestCase):
    def test_writes_15_history_and_future_action_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = []
            for index in range(15):
                path = root / f"history_{index}.png"
                Image.new("RGB", (8, 8), color=(index, 0, 0)).save(path)
                history.append(str(path))
            row = {
                "branch_id": "scene::branch=left",
                "counterfactual_group_id": "scene",
                "source_key": "scene",
                "history_images": history,
                "history_ego_state": [[float(index), 0.0, 0.0, 1.0, 0.0] for index in range(15)],
                "action_trajectory": [[1.0, 0.1, 0.01], [2.0, 0.2, 0.02]],
                "wam_model_id": "drivingworld",
            }
            rows = prepare_rows([row], root / "out")
            branch = Path(rows[0]["input_dir"])
            self.assertEqual(len(list(branch.glob("*.png"))), 17)
            self.assertEqual(np.load(branch / "pose.npy").shape, (1, 17, 2))
            self.assertEqual(np.load(branch / "yaw.npy").shape, (1, 17, 1))
            self.assertTrue(rows[0]["placeholder_future_images"])


if __name__ == "__main__":
    unittest.main()
