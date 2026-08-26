import unittest

import numpy as np

from scripts.build_event_groups_from_wam import build_groups


class BuildEventGroupsTest(unittest.TestCase):
    def test_builds_independent_action_and_realized_targets(self):
        times = np.asarray([0.5, 1.0])
        trajectory = [[1.0, 0.0, 0.0], [2.0, 0.2, 0.06]]
        report = {"groups": [{"counterfactual_group_id": "g", "branches": [
            {
                "branch_id": "b",
                "condition_action_id": "a",
                "imagined_future": trajectory,
                "executed_action": trajectory,
                "realized_future": trajectory,
                "task_success": True,
                "valid": True,
            }
        ]}]}
        groups = build_groups(report, times=times)
        branch = groups[0]["branches"][0]
        self.assertEqual(branch["realized_event_source"], "simulator_state")
        self.assertEqual(branch["task_success"], True)
        self.assertEqual(branch["imagined_event_posterior"][1]["lateral_event"], "turn_left")


if __name__ == "__main__":
    unittest.main()
