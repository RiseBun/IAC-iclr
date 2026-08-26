import unittest

from iac_new.event_benchmark import (
    FORMAL_IMAGE_EVENT_SOURCE,
    audit_event_benchmark_groups,
    build_event_control_groups,
    evaluate_fui_group,
    scene_disjoint_split,
)
from scripts.evaluate_event_causal_metrics import evaluate_groups


def _event(label):
    labels = ("keep_lane", "turn_left", "turn_right")
    return [{
        "lateral_event": label,
        "lateral_posterior": {item: float(item == label) for item in labels},
        "observability": 1.0,
        "abstain": False,
    }]


def _group(group_id="g", scene_id="scene-a"):
    labels = ("keep_lane", "turn_left", "turn_right")
    branches = []
    for label in labels:
        branches.append({
            "branch_id": f"{group_id}:{label}",
            "condition_action_id": label,
            "imagined_event_source": FORMAL_IMAGE_EVENT_SOURCE,
            "generated_future_id": f"future:{label}",
            "imagined_event_posterior": _event(label),
            "action_event_target": _event(label),
            "realized_event_target": _event(label),
            "realized_event_source": "simulator_state",
            "task_success": True,
        })
    return {
        "counterfactual_group_id": group_id,
        "scene_id": scene_id,
        "history_id": f"history:{group_id}",
        "generation_seed": 7,
        "branches": branches,
    }


class EventBenchmarkTest(unittest.TestCase):
    def test_readiness_distinguishes_level2_from_fui(self):
        result = audit_event_benchmark_groups([_group()])
        self.assertTrue(result["formal_event_cc_fcs_ready"])
        self.assertFalse(result["causal_closure_fui_ready"])

    def test_scene_split_has_no_overlap_and_is_stable(self):
        groups = [_group("a", "s1"), _group("b", "s2"), _group("c", "s3")]
        first = scene_disjoint_split(groups, seed="fixed")
        second = scene_disjoint_split(groups, seed="fixed")
        self.assertEqual(first, second)
        self.assertEqual(first["scene_overlap"], [])

    def test_controls_have_expected_ordering(self):
        group = _group()
        reports = {}
        for control in ("oracle", "identical_future", "action_swap"):
            reports[control] = evaluate_groups(
                build_event_control_groups([group], control),
                dimensions=("lateral",),
                minimum_observability=0.25,
                minimum_interval_coverage=0.5,
                temperature=0.1,
                compatibility_threshold=0.7,
            )
        self.assertEqual(reports["oracle"]["event_cc"]["diagonal_top1_accuracy"], 1.0)
        self.assertAlmostEqual(reports["identical_future"]["event_cc"]["diagonal_top1_accuracy"], 1.0 / 3.0)
        self.assertEqual(reports["action_swap"]["event_cc"]["diagonal_top1_accuracy"], 0.0)

    def test_fui_follows_swapped_future_and_rejects_null_noise(self):
        group = _group()
        group["task_event_target"] = _event("turn_left")
        group["planner_baseline_selected_action_id"] = "turn_left"
        group["planner_id"] = "planner:v1"
        group["planner_baseline_run_id"] = "run:baseline"
        group["planner_nuisance_seed"] = 41
        group["fui_trials"] = [
            {
                "trial_id": "swap",
                "planner_run_id": "run:swap",
                "planner_nuisance_seed": 41,
                "intervention_type": "future_permutation",
                "future_assignment": {
                    "keep_lane": "keep_lane",
                    "turn_left": "turn_right",
                    "turn_right": "turn_left",
                },
                "selected_action_id": "turn_right",
            },
            {
                "trial_id": "null",
                "planner_run_id": "run:null",
                "planner_nuisance_seed": 41,
                "intervention_type": "null_resample",
                "future_assignment": {
                    "keep_lane": "keep_lane",
                    "turn_left": "turn_left",
                    "turn_right": "turn_right",
                },
                "generated_future_id_by_action": {
                    "keep_lane": "future:null:keep",
                    "turn_left": "future:null:left",
                    "turn_right": "future:null:right",
                },
                "imagined_event_posterior_by_source_action": {
                    "keep_lane": _event("keep_lane"),
                    "turn_left": _event("turn_left"),
                    "turn_right": _event("turn_right"),
                },
                "selected_action_id": "turn_left",
            },
        ]
        audit = audit_event_benchmark_groups([group])
        self.assertTrue(audit["causal_closure_fui_ready"])
        result = evaluate_fui_group(group)
        self.assertEqual(result["future_follow_rate"], 1.0)
        self.assertEqual(result["null_selection_change_rate"], 0.0)
        self.assertEqual(result["fui_lift"], 1.0)

        report = evaluate_groups(
            [group],
            dimensions=("lateral",),
            minimum_observability=0.25,
            minimum_interval_coverage=0.5,
            temperature=0.1,
            compatibility_threshold=0.7,
        )
        self.assertEqual(report["fui"]["status"], "computed")
        self.assertEqual(report["fui"]["fui_lift"], 1.0)


if __name__ == "__main__":
    unittest.main()
