import unittest

import cv2
import numpy as np

from iac_new.flow_reliability import FlowReliabilityFusion, calibrate_historical_flow_bias, calibrate_historical_row_bias
from iac_new.temporal_geometry import HomographyBoundaryPropagator
from iac_new.road_relative import road_relative_posterior


class FlowBoundaryTest(unittest.TestCase):
    def test_flow_reliability_downweights_photometric_break(self):
        h, w = 32, 48
        first = np.zeros((h, w, 3), dtype=np.uint8)
        second = first.copy()
        cv2.rectangle(first, (8, 8), (24, 24), (180, 180, 180), -1)
        cv2.rectangle(second, (9, 8), (25, 24), (180, 180, 180), -1)
        flow = np.zeros((1, h, w, 2), dtype=np.float32)
        flow[..., 0] = 1.0
        result = FlowReliabilityFusion(tile_size=8).estimate(
            observed_flows=flow,
            frames=[first, second],
            consistency_masks=np.ones((1, h, w), dtype=bool),
        )
        self.assertEqual(result["weights"].shape, (1, h, w))
        self.assertGreater(result["valid_fraction"], 0.0)
        self.assertTrue(result["photometric_available"])

    def test_boundary_propagation_has_no_realized_state_dependency(self):
        masks = np.zeros((3, 72, 128), dtype=bool)
        for index in range(3):
            for row in range(24, 72):
                left = 28 + index + (72 - row) // 7
                right = 98 + index - (72 - row) // 7
                masks[index, row, left:right] = True
        result = HomographyBoundaryPropagator().update(
            masks,
            intrinsics=np.asarray([[100.0, 0.0, 64.0], [0.0, 100.0, 36.0], [0.0, 0.0, 1.0]]),
            camera_to_ego=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]),
            history_ego_state=np.asarray([[0.0, 0.0, 0.0, 4.0, 0.0]]),
            history_times_s=np.asarray([0.0]),
            future_times_s=np.asarray([0.5, 1.0, 1.5]),
        )
        self.assertTrue(result["available"])
        self.assertEqual(len(result["states"]), 3)
        self.assertFalse(result["uses_realized_future_state"])
        self.assertGreaterEqual(result["propagation"][1]["fused_keypoints"], 2)

    def test_historical_bias_calibration_is_candidate_blind(self):
        h, w = 40, 64
        flow = np.zeros((4, h, w, 2), dtype=np.float32)
        flow[..., 0] = 2.0
        flow[..., 1] = 0.25
        states = np.asarray([[-0.3, 0.0, 0.0], [-0.2, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result = calibrate_historical_flow_bias(
            full_flows=flow,
            history_ego_state=states,
            history_count=4,
            camera_to_ego=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]),
            intrinsics=np.asarray([[100.0, 0.0, 32.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]]),
            roi_mask=np.ones((h, w), dtype=bool),
            consistency_masks=np.ones((4, h, w), dtype=bool),
        )
        self.assertTrue(result["available"])
        self.assertIn("median_correction_px", result)
        self.assertNotIn("realized_future_ego_state", result)

    def test_historical_row_bias_shape_and_conservative_output(self):
        h, w = 40, 64
        flow = np.zeros((4, h, w, 2), dtype=np.float32)
        states = np.asarray([[-0.3, 0.0, 0.0], [-0.2, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result = calibrate_historical_row_bias(
            full_flows=flow, history_ego_state=states, history_count=4,
            camera_to_ego=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]),
            intrinsics=np.asarray([[100.0, 0.0, 32.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]]),
            roi_mask=np.ones((h, w), dtype=bool), bands=4,
        )
        self.assertEqual(result["corrected_flows"].shape, flow.shape)
        self.assertLessEqual(result["median_correction_px"], 0.8)

    def test_boundary_propagation_recovers_single_missing_mask(self):
        h, w = 72, 128
        masks = np.zeros((3, h, w), dtype=bool)
        for index in (0, 2):
            for row in range(24, h):
                left = 28 + (h - row) // 7
                right = 98 - (h - row) // 7
                masks[index, row, left:right] = True
        # The middle mask is fully missing.  A consistent small image-space
        # translation stands in for a static road flow field.
        flows = np.zeros((3, h, w, 2), dtype=np.float32)
        flows[..., 0] = 1.0
        result = HomographyBoundaryPropagator(min_current_confidence_for_propagation=0.95).update(
            masks,
            intrinsics=np.asarray([[100.0, 0.0, 64.0], [0.0, 100.0, 36.0], [0.0, 0.0, 1.0]]),
            camera_to_ego=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]),
            observed_flows=flows,
            static_weights=np.ones((3, h, w), dtype=np.float32),
            history_ego_state=np.asarray([[0.0, 0.0, 0.0, 4.0, 0.0]]),
            history_times_s=np.asarray([0.0]),
            future_times_s=np.asarray([0.5, 1.0, 1.5]),
        )
        self.assertTrue(result["available"])
        self.assertTrue(result["propagation"][1]["propagation_applied"])
        self.assertGreaterEqual(result["states"][1]["confidence"], 0.0)

    def test_interval_support_inflation_preserves_median(self):
        posterior = road_relative_posterior(
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.2, 0.1]]),
            np.asarray([0.5, 1.0]),
            lateral_inflation_by_interval=np.asarray([0.1, 0.2]),
            heading_inflation_by_interval=np.asarray([0.01, 0.02]),
            curvature_inflation_by_interval=np.asarray([0.02, 0.03]),
        )
        self.assertEqual(posterior["support"][0]["lateral_offset_range_m"]["q50"], 0.0)
        self.assertAlmostEqual(posterior["support"][1]["lateral_offset_range_m"]["q95"] - posterior["support"][1]["lateral_offset_range_m"]["q50"], 0.2)


if __name__ == "__main__":
    unittest.main()
