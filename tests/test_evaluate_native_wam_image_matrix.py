import unittest

from scripts.evaluate_native_wam_image_matrix import build_decoder_record


class NativeWamImageMatrixTest(unittest.TestCase):
    def _row(self):
        return {
            "branch_id": "scene::branch=logged",
            "counterfactual_group_id": "scene",
            "source_key": "scene",
            "history_images": ["h0", "h1", "h2", "h3"],
            "future_images": ["f0", "f1", "f2", "f3"],
            "future_times_s": [0.5, 1.0, 1.5, 2.0],
            "intrinsics": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
            "camera_to_ego": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "action_trajectory": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        }

    def test_builds_candidate_blind_decoder_record(self):
        row = self._row()
        record = build_decoder_record(row, [{"candidate_id": row["branch_id"], "trajectory": row["action_trajectory"], "prior": 1.0}])
        self.assertEqual(record["history_frame_paths"], row["history_images"])
        self.assertEqual(record["future_frame_paths"], row["future_images"])
        self.assertEqual(record["gt_candidate_id"], row["branch_id"])
        self.assertEqual(record["metadata"]["source_key"], "scene")

    def test_rejects_missing_camera_model(self):
        row = self._row()
        del row["intrinsics"]
        with self.assertRaises(ValueError):
            build_decoder_record(row, [{"candidate_id": row["branch_id"], "trajectory": row["action_trajectory"], "prior": 1.0}])

    def test_accepts_model_specific_window(self):
        row = self._row()
        row["history_images"] = [f"h{i}" for i in range(15)]
        row["future_images"] = [f"f{i}" for i in range(5)]
        row["future_times_s"] = [0.5, 1.0, 1.5, 2.0, 2.5]
        record = build_decoder_record(row, [{"candidate_id": row["branch_id"], "trajectory": [[1.0, 0.0, 0.0]] * 5, "prior": 1.0}])
        self.assertEqual(len(record["history_frame_paths"]), 15)
        self.assertEqual(len(record["future_frame_paths"]), 5)


if __name__ == "__main__":
    unittest.main()
