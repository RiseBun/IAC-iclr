import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.build_benchmark_readiness_report import build


class BenchmarkReadinessTest(unittest.TestCase):
    def test_report_is_conservative_until_realized_state_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(name, data):
                path = root / name
                path.write_text(json.dumps(data), encoding="utf-8")
                return path
            result = build(Namespace(
                capability=write("cap.json", {"rows": [{"model_id": "x", "suitable_for_counterfactual_image_cc": True}]}),
                sensitivity=write("sens.json", {"mean_future_image_l1": 0.1, "action_image_distance_correlation": 0.2}),
                controls=write("ctrl.json", {"reports": {"logged": {"diagonal_top1_accuracy": 0.5, "mean_cc_margin": 0.1}}}),
                split_report=write("split.json", {"holdout": {"pairs": 2}}),
                test_count=86,
            ))
            self.assertFalse(result["formal_benchmark_ready"])
            self.assertEqual(result["gates"]["realized_state_cc"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
