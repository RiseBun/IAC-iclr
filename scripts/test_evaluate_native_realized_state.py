import unittest

import numpy as np

from scripts.evaluate_native_realized_state import _native_to_decoder_row


class NativeRealizedStateAdapterTest(unittest.TestCase):
    def test_adapter_preserves_independent_state_and_uses_schema_only_candidates(self):
        row = {
            "source_key": "navsim/sample",
            "scene_name": "scene-1",
            "history_images": ["h0.jpg", "h1.jpg", "h2.jpg", "h3.jpg"],
            "future_images": ["f0.jpg", "f1.jpg"],
            "future_times_s": [0.5, 1.0],
            "camera_intrinsic": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            "camera_to_ego": np.eye(4).tolist(),
            "trajectory": [[1.0, 0.0, 0.0], [2.0, 0.1, 0.01]],
            "realized_future_ego_state": [[1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.1, 0.01, 0.0, 0.0]],
        }

        adapted = _native_to_decoder_row(row, 0)

        self.assertEqual(adapted["history_times_s"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(adapted["future_times_s"], [2.0, 2.5])
        self.assertEqual(adapted["gt_candidate_id"], "native_realized")
        self.assertEqual(len(adapted["candidates"]), 2)
        self.assertEqual(adapted["native_realized_future_ego_state"], row["realized_future_ego_state"])

    def test_schema_keeps_candidate_bank_out_of_decoder_inputs(self):
        row = {
            "history_images": ["h0", "h1", "h2", "h3"],
            "future_images": ["f0"],
            "future_times_s": [0.5],
            "camera_intrinsic": np.eye(3).tolist(),
            "camera_to_ego": np.eye(4).tolist(),
            "trajectory": [[1.0, 0.0, 0.0]],
            "realized_future_ego_state": [[1.0, 0.0, 0.0, 0.0, 0.0]],
        }
        adapted = _native_to_decoder_row(row, 0)
        self.assertEqual(adapted["candidates"][0]["trajectory"], adapted["candidates"][1]["trajectory"])


if __name__ == "__main__":
    unittest.main()
