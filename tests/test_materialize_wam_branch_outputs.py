import unittest

from scripts.materialize_wam_branch_outputs import materialize


class MaterializeWamBranchOutputsTest(unittest.TestCase):
    def test_preserves_lineage_and_state(self):
        branches = [{
            "branch_id": "source::branch=logged",
            "source_key": "source",
            "history_ego_state": [[0, 0, 0, 1, 0]],
            "realized_future_ego_state": [[1, 0, 0, 1, 0]],
            "task_success": True,
        }]
        output = materialize(branches, [{
            "branch_id": "source::branch=logged",
            "generated_future_images": ["f0.jpg", "f1.jpg"],
            "wam_output_id": "backend-1",
        }])
        self.assertEqual(output[0]["future_images"], ["f0.jpg", "f1.jpg"])
        self.assertEqual(output[0]["source_key"], "source")
        self.assertEqual(output[0]["realized_future_ego_state"], [[1, 0, 0, 1, 0]])
        self.assertTrue(output[0]["task_success"])
        self.assertEqual(output[0]["future_images_source"], "wam_generated")

    def test_missing_or_duplicate_keys_fail_closed(self):
        branches = [{"branch_id": "a"}, {"branch_id": "b"}]
        with self.assertRaises(ValueError):
            materialize(branches, [{"branch_id": "a", "future_images": ["a.jpg"]}])
        with self.assertRaises(ValueError):
            materialize(
                [{"branch_id": "a"}],
                [
                    {"branch_id": "a", "future_images": ["a.jpg"]},
                    {"branch_id": "a", "future_images": ["a2.jpg"]},
                ],
            )


if __name__ == "__main__":
    unittest.main()
