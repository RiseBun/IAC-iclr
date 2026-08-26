import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from iac_new.measurement_validation import (
    build_blind_annotation_pack,
    read_records,
    score_measurement_validity,
)


def _event(label, *, observability=1.0, abstain=False):
    labels = ("keep_lane", "turn_left", "turn_right")
    return {
        "lateral_event": label,
        "lateral_posterior": {item: float(item == label) for item in labels},
        "observability": observability,
        "abstain": abstain,
    }


class MeasurementValidationTest(unittest.TestCase):
    def test_annotation_pack_separates_public_tasks_from_private_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for index in range(3):
                path = root / f"secret_model_frame_{index}.png"
                path.write_bytes(f"frame-{index}".encode())
                frames.append(str(path))
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "branch_id": "secret-branch-left",
                "counterfactual_group_id": "secret-group",
                "scene_name": "secret-scene",
                "branch_mode": "left",
                "history_images": frames[:1],
                "future_images": frames[1:],
                "future_times_s": [0.5, 1.0],
            }) + "\n", encoding="utf-8")
            groups = root / "groups.jsonl"
            groups.write_text(json.dumps({
                "counterfactual_group_id": "secret-group",
                "branches": [{
                    "branch_id": "secret-branch-left",
                    "condition_action_id": "left",
                    "imagined_event_posterior": [_event("keep_lane"), _event("turn_left")],
                    "action_event_target": [_event("keep_lane"), _event("turn_left")],
                }],
            }) + "\n", encoding="utf-8")
            output = root / "pack"
            result = build_blind_annotation_pack([{
                "source_id": "secret-model",
                "manifest_path": manifest,
                "event_groups_path": groups,
            }], output, normalize_media=False)

            self.assertEqual(result["num_items"], 1)
            public_text = (output / "public" / "tasks.jsonl").read_text()
            self.assertNotIn("secret-model", public_text)
            self.assertNotIn("secret-branch", public_text)
            self.assertNotIn("secret-group", public_text)
            self.assertNotIn(str(root), public_text)
            private = read_records(output / "private" / "private_key.jsonl")[0]
            self.assertEqual(private["source_id"], "secret-model")
            task = read_records(output / "public" / "tasks.jsonl")[0]
            self.assertTrue(
                all((output / "public" / path).is_file() for path in task["frame_paths"])
            )
            checksums = (output / "public" / "SHA256SUMS.txt").read_text()
            self.assertEqual(len(checksums.splitlines()), 5)
            self.assertNotIn("private_key.jsonl", checksums)

    def test_scoring_reports_calibration_coverage_and_scene_bootstrap(self):
        private = [
            {
                "item_id": "a",
                "source_id": "model-a",
                "scene_id": "scene-1",
                "probe_event_posterior": [
                    _event("keep_lane"),
                    _event("turn_left", observability=0.1),
                ],
            },
            {
                "item_id": "b",
                "source_id": "model-b",
                "scene_id": "scene-2",
                "probe_event_posterior": [
                    _event("turn_right"),
                    _event("turn_left"),
                ],
            },
        ]
        annotations = []
        labels = {
            "a": ["keep_lane", "turn_left"],
            "b": ["turn_right", "turn_left"],
        }
        for annotator in ("r1", "r2", "r3"):
            for item_id in ("a", "b"):
                annotations.append({
                    "item_id": item_id,
                    "annotator_id": annotator,
                    "interval_labels": labels[item_id],
                    "interval_observable": [True, True],
                })
        result = score_measurement_validity(
            private,
            annotations,
            bootstrap_samples=20,
            seed=3,
        )
        self.assertEqual(result["num_scored_intervals"], 3)
        self.assertEqual(result["exclusion_counts"], {"low_probe_observability": 1})
        self.assertEqual(result["probe_metrics"]["accuracy"], 1.0)
        self.assertEqual(result["probe_metrics"]["coverage_over_human_consensus"], 0.75)
        self.assertEqual(result["human_agreement"]["generalized_kappa"], 1.0)
        self.assertEqual(result["scene_bootstrap"]["num_clusters"], 2)
        self.assertEqual(set(result["per_source"]), {"model-a", "model-b"})

    def test_human_event_cc_separates_wam_response_from_probe_fidelity(self):
        private = []
        annotations = []
        for label in ("keep_lane", "turn_left", "turn_right"):
            item_id = f"item-{label}"
            private.append({
                "item_id": item_id,
                "source_id": "wam-a",
                "scene_id": "scene-a",
                "counterfactual_group_id": "group-a",
                "condition_action_id": label,
                "probe_event_posterior": [_event(label)],
                "action_event_target": [_event(label)],
            })
            for annotator in ("r1", "r2", "r3"):
                annotations.append({
                    "item_id": item_id,
                    "annotator_id": annotator,
                    "interval_labels": [label],
                    "interval_observable": [True],
                })
        result = score_measurement_validity(
            private,
            annotations,
            bootstrap_samples=0,
        )
        comparison = result["human_probe_event_cc"]
        self.assertEqual(comparison["num_groups"], 1)
        self.assertEqual(comparison["human_diagonal_top1_accuracy"], 1.0)
        self.assertEqual(comparison["probe_diagonal_top1_accuracy"], 1.0)
        self.assertEqual(comparison["human_probe_cc_margin_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
