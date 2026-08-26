import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_wam_probe_splits import split_report


class AggregateWamProbeSplitsTest(unittest.TestCase):
    def test_requires_candidate_blind_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "video_id": "v_a", "twin_id": "t", "frame_times_s": [0.2, 0.6],
                "supported_candidate_index": 0,
                "candidates": [{"candidate_index": 0, "motion": {"forward_rate_mps": 1, "lateral_rate_mps": 0, "yaw_rate_rps": 0}}, {"candidate_index": 1, "motion": {"forward_rate_mps": 1, "lateral_rate_mps": 1, "yaw_rate_rps": 0}}],
            }
            score = {"video_id": "v_a"}
            mp = root / "manifest.jsonl"; sp = root / "scores.jsonl"
            mp.write_text(json.dumps(manifest) + "\n")
            sp.write_text(json.dumps(score) + "\n")
            with self.assertRaises(ValueError):
                split_report(mp, sp)


if __name__ == "__main__":
    unittest.main()
