import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WamPairAuditTest(unittest.TestCase):
    def test_audit_fails_closed_without_pair_identity(self) -> None:
        row = {"pair_id": "p", "branches": [{}, {}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            script = Path(__file__).parents[1] / "scripts" / "audit_wam_pairs.py"
            result = subprocess.run([sys.executable, str(script), "--pairs", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(json.loads(result.stdout)["causal_metric_ready"])


if __name__ == "__main__":
    unittest.main()
