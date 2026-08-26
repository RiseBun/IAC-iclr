import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from iac_new.causal_annotation import build_blind_causal_seed_pack, read_jsonl


def _candidate(root: Path, candidate_id: str, chain_type: str):
    paths = []
    for index in range(4):
        path = root / f"secret-{candidate_id}-{index}.jpg"
        path.write_bytes(f"frame-{index}".encode())
        paths.append(str(path))
    return {
        "protocol": "iac-causal-candidate-v1",
        "candidate_id": candidate_id,
        "source_key": f"secret-source:{candidate_id}",
        "chain_type": chain_type,
        "candidate_trigger_tags": ["secret-scenario-tag"],
        "trigger_label_status": "candidate_only_requires_blind_confirmation",
        "counterfactual_pair_status": "not_constructed",
        "log_name": "secret-log",
        "history_images": paths[:2],
        "history_offsets_s": [-1.0, 0.0],
        "future_images": paths[2:],
        "future_offsets_s": [1.0, 2.0],
    }


class CausalAnnotationTest(unittest.TestCase):
    def test_pack_hides_candidate_identity_and_suggestion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [
                _candidate(root, "candidate-a", "pedestrian_crossing"),
                _candidate(root, "candidate-b", "blocked_lane"),
            ]
            output = root / "pack"
            result = build_blind_causal_seed_pack(
                candidates,
                output,
                normalize_media=False,
            )
            self.assertEqual(result["num_items"], 2)
            self.assertEqual(result["minimum_independent_annotators"], 3)

            public_text = (output / "public" / "tasks.jsonl").read_text()
            for secret in (
                "candidate-a",
                "candidate-b",
                "secret-source",
                "secret-scenario-tag",
                "secret-log",
            ):
                self.assertNotIn(secret, public_text)
            tasks = read_jsonl(output / "public" / "tasks.jsonl")
            self.assertEqual(tasks[0]["frame_offsets_s"], [-1.0, 0.0, 1.0, 2.0])
            self.assertTrue(
                all(
                    (output / "public" / path).is_file()
                    for task in tasks
                    for path in task["frame_paths"]
                )
            )
            private = read_jsonl(output / "private" / "private_key.jsonl")
            self.assertEqual({row["candidate_id"] for row in private}, {"candidate-a", "candidate-b"})
            template = read_jsonl(output / "public" / "annotations_template.jsonl")[0]
            self.assertIn("stage_observable", template)
            self.assertIn("resolution_state", template)
            self.assertEqual(template["ego_response_events"], [])
            checksums = (output / "public" / "SHA256SUMS.txt").read_text()
            self.assertNotIn("private_key.jsonl", checksums)

    def test_rejects_duplicate_candidate_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate(root, "duplicate", "pedestrian_crossing")
            with self.assertRaisesRegex(ValueError, "globally unique"):
                build_blind_causal_seed_pack(
                    [candidate, dict(candidate)],
                    root / "pack",
                    normalize_media=False,
                )

    def test_rejects_promoted_scenario_tag(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate(root, "candidate", "pedestrian_crossing")
            candidate["trigger_label_status"] = "ground_truth"
            with self.assertRaisesRegex(ValueError, "candidate-only"):
                build_blind_causal_seed_pack(
                    [candidate],
                    root / "pack",
                    normalize_media=False,
                )


if __name__ == "__main__":
    unittest.main()
