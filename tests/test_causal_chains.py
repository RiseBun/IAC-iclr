import copy
import json
import unittest
from pathlib import Path

from iac_new.causal_chains import (
    CAUSAL_CHAIN_TEMPLATES,
    audit_causal_chain_records,
    evaluate_causal_chain_records,
)


def _stage(evidence_id, source, scores, **extra):
    return {
        "evidence_id": evidence_id,
        "source": source,
        "scores": scores,
        **extra,
    }


def _record(state="risk", chain_type="cut_in_or_lead_brake"):
    risk = state == "risk"
    return {
        "chain_episode_id": f"pair-1:{state}",
        "counterfactual_pair_id": "pair-1",
        "scene_id": "scene-1",
        "history_id": "history-1",
        "chain_type": chain_type,
        "world_state": state,
        "world_intervention_id": f"intervention:{state}",
        "generated_future_id": f"future:{state}",
        "planner_id": "planner:v1",
        "planner_run_id": f"run:{state}",
        "planner_nuisance_seed": 17,
        "trigger": {
            "evidence_id": f"trigger:{state}",
            "source": "simulator_state",
            "label": "vehicle_cut_in" if risk else "vehicle_keeps_lane",
        },
        "imagined_consequence": _stage(
            f"imagined:{state}",
            "frozen_iac_interaction_probe_v1",
            {"collision_risk": 0.95 if risk else 0.05, "no_conflict": 0.05 if risk else 0.95},
            observability=0.95,
            abstain=False,
        ),
        "selected_response": _stage(
            f"response:{state}",
            "planner_output",
            {"emergency_brake": 0.90 if risk else 0.05, "maintain_speed": 0.05 if risk else 0.90},
        ),
        "realized_outcome": _stage(
            f"outcome:{state}",
            "simulator_state",
            {"no_collision": 1.0, "safe_ttc": 1.0, "collision": 0.0},
            task_success=True,
        ),
    }


class CausalChainTest(unittest.TestCase):
    def test_exactly_four_preregistered_chain_types(self):
        self.assertEqual(
            set(CAUSAL_CHAIN_TEMPLATES),
            {
                "cut_in_or_lead_brake",
                "pedestrian_crossing",
                "blocked_lane",
                "unprotected_turn_or_merge",
            },
        )

    def test_perfect_risk_clear_pair_is_ready_and_aligned(self):
        records = [_record("risk"), _record("clear")]
        audit = audit_causal_chain_records(records)
        self.assertTrue(audit["formal_counterfactual_ready"])
        self.assertFalse(audit["four_chain_suite_ready"])
        result = evaluate_causal_chain_records(records)
        pair = result["pairs"][0]
        self.assertTrue(pair["directionally_aligned"])
        self.assertTrue(pair["joint_chain_success"])
        self.assertAlmostEqual(pair["imagined_risk_contrast"], 0.90)
        self.assertAlmostEqual(pair["protective_action_contrast"], 0.85)
        self.assertAlmostEqual(pair["causal_chain_score"], 0.85)

    def test_single_risk_record_is_not_counterfactual_ready(self):
        audit = audit_causal_chain_records([_record("risk")])
        self.assertFalse(audit["formal_counterfactual_ready"])
        self.assertIn(
            "pair_must_contain_exactly_risk_and_clear",
            audit["pairs"][0]["reasons"],
        )

    def test_unfrozen_imagined_source_fails_closed(self):
        records = [_record("risk"), _record("clear")]
        records[0]["imagined_consequence"]["source"] = "vlm_auto_tag"
        audit = audit_causal_chain_records(records)
        self.assertFalse(audit["rows"][0]["evidence_ready"])
        self.assertIn(
            "unfrozen_or_missing_imagined_source",
            audit["rows"][0]["evidence_reasons"],
        )

    def test_action_that_ignores_risk_has_zero_causal_score(self):
        records = [_record("risk"), _record("clear")]
        records[0]["selected_response"]["scores"] = {
            "emergency_brake": 0.05,
            "maintain_speed": 0.90,
        }
        result = evaluate_causal_chain_records(records)
        pair = result["pairs"][0]
        self.assertFalse(pair["directionally_aligned"])
        self.assertEqual(pair["causal_chain_score"], 0.0)

    def test_pair_must_hold_planner_noise_fixed(self):
        records = [_record("risk"), _record("clear")]
        records[1]["planner_nuisance_seed"] = 19
        audit = audit_causal_chain_records(records)
        self.assertFalse(audit["formal_counterfactual_ready"])
        self.assertIn(
            "pair_planner_nuisance_seed_not_held_fixed",
            audit["pairs"][0]["reasons"],
        )

    def test_unknown_labels_are_rejected(self):
        records = [_record("risk"), _record("clear")]
        broken = copy.deepcopy(records)
        broken[0]["selected_response"]["scores"]["teleport"] = 1.0
        audit = audit_causal_chain_records(broken)
        self.assertIn(
            "unknown_selected_response_label",
            audit["rows"][0]["evidence_reasons"],
        )

    def test_four_chain_example_is_suite_ready(self):
        records = [
            json.loads(line)
            for line in Path("configs/causal_chain_v1.example.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        result = evaluate_causal_chain_records(records)
        self.assertTrue(result["four_chain_suite_ready"])
        self.assertEqual(result["num_evaluable_pairs"], 4)
        self.assertIsNotNone(result["macro_mean_causal_chain_score"])

    def test_pedestrian_chain_requires_restart_after_stop(self):
        records = [
            json.loads(line)
            for line in Path("configs/causal_chain_v1.example.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if '"counterfactual_pair_id":"ped-1"' in line
        ]
        records[0]["selected_response"]["scores"]["restart"] = 0.0
        result = evaluate_causal_chain_records(records)
        self.assertEqual(result["pairs"][0]["causal_chain_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
