import unittest

from scripts.audit_state_join_keys import audit


class StateJoinAuditTest(unittest.TestCase):
    def test_exact_source_key_join_is_ready_when_state_and_success_exist(self):
        report = audit(
            [{"source_key": "scene:1", "video_id": "v1"}],
            [{
                "source_key": "scene:1",
                "realized_future_ego_state": [[1, 0, 0, 1, 0]],
                "task_success": True,
            }],
        )
        self.assertTrue(report["closed_loop_join_ready"])
        self.assertTrue(report["fcs_join_ready"])

    def test_scene_name_only_is_reported_but_not_full_join(self):
        report = audit(
            [{"scene_name": "scene-1", "video_id": "wam"}],
            [{
                "scene_name": "scene-1",
                "realized_future_ego_state": [[1, 0, 0, 1, 0]],
                "task_success": True,
            }],
        )
        self.assertFalse(report["fcs_join_ready"])
        self.assertEqual(report["shared_key_types"], {"scene_name": 1})

    def test_ambiguous_key_fails_closed(self):
        report = audit(
            [{"source_key": "same"}, {"source_key": "same"}],
            [
                {"source_key": "same", "realized_future_ego_state": [], "task_success": True},
                {"source_key": "same", "realized_future_ego_state": [], "task_success": True},
            ],
        )
        self.assertFalse(report["closed_loop_join_ready"])
        self.assertEqual(len(report["ambiguous_shared_keys"]), 1)


if __name__ == "__main__":
    unittest.main()
