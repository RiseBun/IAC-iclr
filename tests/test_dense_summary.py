import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys


class DenseSummaryTest(unittest.TestCase):
    def test_summary_reports_gt_rank_and_topk(self) -> None:
        row = {
            "gt_candidate_id": "gt",
            "candidate_scores": [
                {"candidate_id": "a", "energy": 0.1},
                {"candidate_id": "gt", "energy": 0.2},
                {"candidate_id": "b", "energy": 0.3},
            ],
            "prediction_set_size": 2,
            "valid": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            script = Path(__file__).parents[1] / "scripts" / "summarize_dense_counterfactual.py"
            output = subprocess.check_output(
                [sys.executable, str(script), "--scores", str(path)], text=True
            )
        result = json.loads(output)
        self.assertEqual(result["median_gt_energy_rank"], 2.0)
        self.assertEqual(result["topk_gt_coverage"]["1"], 0.0)
        self.assertEqual(result["topk_gt_coverage"]["5"], 1.0)


if __name__ == "__main__":
    unittest.main()
