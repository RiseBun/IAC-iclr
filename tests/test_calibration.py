import unittest

from iac_new.calibration import calibration_payload, calibration_status


class CalibrationTest(unittest.TestCase):
    def test_complete_calibration_is_projectable(self) -> None:
        row = {
            "intrinsics": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            "camera_to_ego": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]],
        }
        result = calibration_status(row)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["projectable"])
        self.assertEqual(result["projection_mode"], "metric_ego")

    def test_missing_calibration_never_gets_identity_fallback(self) -> None:
        result = calibration_payload({"video_id": "v"})
        self.assertEqual(result["status"], "missing")
        self.assertFalse(result["projectable"])
        self.assertEqual(result["projection_mode"], "image_plane_only")
        self.assertIn("missing_intrinsics", result["reasons"])
        self.assertNotIn("intrinsics", result)

    def test_invalid_calibration_is_not_partial(self) -> None:
        result = calibration_status({"intrinsics": [[1.0]], "camera_to_ego": [[1.0] * 4] * 4})
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["projectable"])


if __name__ == "__main__":
    unittest.main()
