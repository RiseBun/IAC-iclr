import unittest

from scripts.audit_benchmark_manifest import audit


class BenchmarkAuditTest(unittest.TestCase):
    def test_shared_history_and_actions_are_required(self) -> None:
        rows = [
            {"twin_id": "p", "branch_id": "a", "scene_name": "s", "history_images": ["h0", "h1"], "future_images": ["f0"], "future_times_s": [0.5], "action_condition": [[0.0, 0.0, 0.0]]},
            {"twin_id": "p", "branch_id": "b", "scene_name": "s", "history_images": ["h0", "h1"], "future_images": ["f1"], "future_times_s": [0.5], "action_condition": [[0.1, 0.0, 0.0]]},
        ]
        result = audit(rows, require_realized=False, require_success=False)
        self.assertTrue(result["image_probe_ready"])
        self.assertTrue(result["action_response_ready"])
        self.assertFalse(result["fcs_ready"])

    def test_different_history_is_rejected(self) -> None:
        rows = [
            {"twin_id": "p", "branch_id": "a", "history_images": ["h0", "h1"], "future_images": ["f0"], "future_times_s": [0.5], "action_condition": [[0, 0, 0]]},
            {"twin_id": "p", "branch_id": "b", "history_images": ["other", "other-1"], "future_images": ["f1"], "future_times_s": [0.5], "action_condition": [[1, 0, 0]]},
        ]
        result = audit(rows, require_realized=False, require_success=False)
        self.assertFalse(result["image_probe_ready"])
        self.assertFalse(result["action_response_ready"])
        self.assertTrue(any(issue["field"] == "history_images" for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
