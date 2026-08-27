import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_actor_motion_manifest import audit


class ActorMotionManifestTest(unittest.TestCase):
    def test_audit_accepts_eight_frame_independent_record(self) -> None:
        row = {
            "protocol": "actor-motion-reference-v3",
            "sample_id": "s1",
            "chain_type": "cut_in_or_lead_brake",
            "history_frame_paths": ["h0", "h1", "h2", "h3"],
            "future_frame_paths": [f"f{i}" for i in range(8)],
            "future_times_s": [0.5 * (i + 1) for i in range(8)],
            "actor_tracks": [{
                "actor_id": "a1",
                "positions_ego_m": [[20.0 - i, 0.1] for i in range(8)],
                "visibility": [True] * 8,
                "lidar_visibility": [True] * 8,
                "image_visibility": [True] * 8,
                "ground_contact_pixels_uv": [[500.0, 700.0]] * 8,
                "actor_boxes_xyxy": [[450.0, 500.0, 550.0, 700.0]] * 8,
            }],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = audit(path)
        self.assertTrue(report["formal_ready"])

    def test_audit_rejects_action_fields(self) -> None:
        row = {
            "protocol": "actor-motion-reference-v2",
            "sample_id": "s1",
            "chain_type": "blocked_lane",
            "history_frame_paths": ["h0"] * 4,
            "future_frame_paths": [f"f{i}" for i in range(8)],
            "future_times_s": [0.5 * (i + 1) for i in range(8)],
            "actor_tracks": [{
                "positions_ego_m": [[1.0, 0.0]] * 8,
                "visibility": [True] * 8,
                "lidar_visibility": [True] * 8,
                "image_visibility": [True] * 8,
                "ground_contact_pixels_uv": [[500.0, 700.0]] * 8,
            }],
            "action_trajectory": [[1.0, 0.0, 0.0]],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = audit(path)
        self.assertFalse(report["formal_ready"])
        self.assertEqual(report["num_errors"], 1)


if __name__ == "__main__":
    unittest.main()
