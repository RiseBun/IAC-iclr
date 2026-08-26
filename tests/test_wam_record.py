import unittest

from iac_new.wam_record import validate_trajectory_image_record


def _record():
    return {
        "dataset": "nuplan",
        "source_key": "nuplan:log:sample",
        "scene_name": "log",
        "timestamp_us": 123,
        "history_images": ["h0.png", "h1.png"],
        "future_images": ["f0.png", "f1.png"],
        "history_ego_state": [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
        "realized_future_ego_state": [[1, 0, 0, 1, 0], [2, 0, 0, 1, 0]],
        "future_times_s": [0.5, 1.0],
        "action_condition": [[1, 0, 0], [2, 0, 0]],
    }


class WamRecordTest(unittest.TestCase):
    def test_normalizes_action_condition(self):
        result = validate_trajectory_image_record(_record())
        self.assertEqual(result["trajectory"], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertEqual(result["condition_trajectory"], result["trajectory"])

    def test_task_success_is_optional(self):
        result = validate_trajectory_image_record(_record())
        self.assertNotIn("task_success", result)

    def test_rejects_missing_realized_state(self):
        record = _record()
        del record["realized_future_ego_state"]
        with self.assertRaises(ValueError):
            validate_trajectory_image_record(record)


if __name__ == "__main__":
    unittest.main()
