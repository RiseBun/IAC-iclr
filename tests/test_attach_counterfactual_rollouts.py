import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.attach_counterfactual_rollouts import attach


class AttachCounterfactualRolloutsTest(unittest.TestCase):
    def _branch(self, branch_id="scene::logged"):
        return {
            "branch_id": branch_id,
            "future_times_s": [0.5, 1.0],
            "action_condition": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        }

    def _rollout(self, branch_id="scene::logged", **extra):
        row = {
            "branch_id": branch_id,
            "realized_future_ego_state": [[1.0, 0.0, 0.0, 2.0, 0.0], [2.0, 0.0, 0.0, 2.0, 0.0]],
            "task_score": 0.8,
            "task_success": True,
            "state_reference_source": "navsim_pdm_closed_loop",
            "action_injection_verified": True,
        }
        row.update(extra)
        return row

    def test_attaches_only_external_state(self):
        rows, summary = attach([self._branch()], [self._rollout()], require_success=True)
        self.assertTrue(summary["closed_loop_ready"])
        self.assertEqual(rows[0]["state_reference_source"], "navsim_pdm_closed_loop")

    def test_rejects_action_proxy(self):
        with self.assertRaises(ValueError):
            attach([self._branch()], [self._rollout(state_reference_source="action_condition")], require_success=True)

    def test_rejects_unverified_action(self):
        with self.assertRaises(ValueError):
            attach([self._branch()], [self._rollout(action_injection_verified=False)], require_success=True)


if __name__ == "__main__":
    unittest.main()
