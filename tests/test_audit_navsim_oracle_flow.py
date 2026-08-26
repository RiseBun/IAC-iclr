import sys
import types
import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from scripts.audit_navsim_oracle_flow import _ego_to_global, flow_metrics


class AuditNavsimOracleFlowTest(unittest.TestCase):
    def test_flow_metrics_identical_flow(self) -> None:
        observed = np.full((4, 5, 2), [2.0, -1.0], dtype=np.float32)
        metrics = flow_metrics(observed, observed.copy(), np.ones((4, 5), dtype=bool))

        self.assertEqual(metrics["num_points"], 20)
        self.assertAlmostEqual(metrics["median_epe_px"], 0.0)
        self.assertAlmostEqual(metrics["median_direction_cosine"], 1.0)
        self.assertAlmostEqual(metrics["median_observed_to_predicted_scale"], 1.0)

    def test_ego_pose_accepts_navsim_quaternion_fields(self) -> None:
        pose = _ego_to_global(
            {
                "ego2global_translation": [1.0, 2.0, 3.0],
                "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            }
        )

        np.testing.assert_allclose(pose, np.asarray([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))


if __name__ == "__main__":
    unittest.main()
