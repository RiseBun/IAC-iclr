import tempfile
import unittest
from pathlib import Path

import numpy as np

from iac_new.protocol import validate_record


def _record() -> dict:
    return {
        "sample_id": "s",
        "frame_paths": ["a.jpg", "b.jpg", "c.jpg"],
        "frame_times_s": [0.0, 0.5, 1.0],
        "intrinsics": [[100.0, 0.0, 20.0], [0.0, 100.0, 15.0], [0.0, 0.0, 1.0]],
        "camera_to_ego": np.eye(4).tolist(),
        "gt_candidate_id": "a",
        "candidates": [
            {"candidate_id": "a", "trajectory": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]},
            {"candidate_id": "b", "trajectory": [[1.0, 0.2, 0.0], [2.0, 0.4, 0.0]]},
        ],
    }


class ProtocolTest(unittest.TestCase):
    def test_protocol_resolves_relative_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = validate_record(_record(), manifest_root=root)
            self.assertEqual(normalized["frame_paths"][0], str(root / "a.jpg"))
            self.assertEqual(normalized["candidates"][0]["trajectory"].shape, (2, 3))

    def test_protocol_preserves_camera_distortion(self) -> None:
        row = _record()
        row["distortion"] = [-0.35, 0.17, -0.002, 0.0004, -0.05]
        with tempfile.TemporaryDirectory() as directory:
            normalized = validate_record(row, manifest_root=Path(directory))
        np.testing.assert_allclose(normalized["distortion"], row["distortion"])

    def test_protocol_resolves_relative_metric_depth(self) -> None:
        row = _record()
        row["metric_depth_path"] = "depth/video_001.npz"
        row["metric_depth_source"] = "unidepth-test"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalized = validate_record(row, manifest_root=root)
            self.assertEqual(
                normalized["metric_depth_path"], str(root / "depth/video_001.npz")
            )
            self.assertEqual(normalized["metric_depth_source"], "unidepth-test")

    def test_protocol_rejects_non_monotonic_time(self) -> None:
        row = _record()
        row["frame_times_s"] = [0.0, 0.5, 0.4]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                validate_record(row, manifest_root=Path(directory))

    def test_explicit_history_future_contract(self) -> None:
        row = _record()
        row.pop("frame_paths")
        row.pop("frame_times_s")
        row["history_frame_paths"] = ["h0.jpg", "h1.jpg", "h2.jpg", "h3.jpg"]
        row["future_frame_paths"] = ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg"]
        row["history_times_s"] = [10.0, 10.5, 11.0, 11.5]
        row["future_times_s"] = [12.0, 12.5, 13.0, 13.5]
        row["candidates"] = [
            {"candidate_id": "a", "trajectory": [[1.0, 0.0, 0.0]] * 4},
            {"candidate_id": "b", "trajectory": [[1.0, 0.2, 0.0]] * 4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            normalized = validate_record(row, manifest_root=Path(directory))
        self.assertEqual(normalized["protocol_variant"], "history4_future4")
        self.assertEqual(len(normalized["frame_paths"]), 8)
        self.assertTrue(np.allclose(normalized["future_times_s"], [0.5, 1.0, 1.5, 2.0]))
        self.assertEqual(normalized["candidates"][0]["trajectory"].shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
