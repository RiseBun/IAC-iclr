import unittest

from scripts.evaluate_event_causal_metrics import evaluate_groups


def _event(label):
    labels = ("keep_lane", "turn_left", "turn_right")
    return {
        "lateral_event": label,
        "lateral_posterior": {item: float(item == label) for item in labels},
        "observability": 1.0,
        "abstain": False,
    }


class EvaluateEventCausalMetricsTest(unittest.TestCase):
    def test_report_keeps_cc_fcs_and_fui_separate(self):
        straight = [_event("keep_lane"), _event("keep_lane")]
        left = [_event("keep_lane"), _event("turn_left")]
        report = evaluate_groups([{
            "counterfactual_group_id": "scene",
            "branches": [
                {
                    "branch_id": "straight",
                    "condition_action_id": "straight",
                    "imagined_event_posterior": straight,
                    "action_event_target": straight,
                    "realized_event_target": straight,
                    "realized_event_source": "ego_state",
                    "task_success": True,
                },
                {
                    "branch_id": "left",
                    "condition_action_id": "left",
                    "imagined_event_posterior": left,
                    "action_event_target": left,
                    "realized_event_target": left,
                    "realized_event_source": "simulator_state",
                    "task_success": True,
                },
            ],
        }], dimensions=("lateral",), minimum_observability=0.25,
            minimum_interval_coverage=0.5, temperature=0.1,
            compatibility_threshold=0.7)

        self.assertEqual(report["event_cc"]["diagonal_top1_accuracy"], 1.0)
        self.assertEqual(report["event_fcs"]["joint_fcs"], 1.0)
        self.assertEqual(report["fui"]["status"], "not_computed")


if __name__ == "__main__":
    unittest.main()
