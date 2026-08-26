import unittest

import numpy as np

from iac_new.temporal_geometry import HomographyBoundaryPropagator, RoadStateFilter, TemporalScaleCalibrator
from iac_new.road_structure import causal_boundary_keypoint_filter, boundary_pixels_to_ego, fuse_ego_boundary_keypoints


class TemporalGeometryTest(unittest.TestCase):
    def test_causal_boundary_keypoint_filter_clips_outlier(self):
        masks = np.zeros((3, 64, 96), dtype=bool)
        for t in range(3):
            for y in range(28, 64):
                left, right = 22 + t, 74 + t
                if t == 2:
                    left, right = 55, 94
                masks[t, y, left:right] = True
        filtered, diag = causal_boundary_keypoint_filter(masks, max_jump_px=10.0, huber_scale_px=3.0)
        self.assertEqual(len(filtered), 3)
        self.assertGreater(diag["clipped_fraction"], 0.0)
        self.assertLess(float(filtered[2]["left_x"][-1]), 40.0)

    def test_causal_boundary_filter_warps_previous_keypoints(self):
        masks = np.zeros((2, 48, 80), dtype=bool)
        masks[0, 20:48, 20:60] = True
        masks[1, 20:48, 24:64] = True
        flows = np.zeros((2, 48, 80, 2), dtype=np.float32)
        flows[0, ..., 0] = 4.0
        filtered, diag = causal_boundary_keypoint_filter(masks, observed_flows=flows, max_jump_px=8.0, huber_scale_px=3.0)
        self.assertEqual(len(filtered), 2)
        self.assertLessEqual(diag["clipped_fraction"], 0.01)

    def test_boundary_pixels_to_ego_ground_intersection(self):
        boundary = {"valid": True, "rows": [36.0, 42.0], "left_x": [35.0, 35.0], "right_x": [65.0, 65.0], "image_height": 64}
        K = np.asarray([[50.0, 0.0, 50.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        T[2, 3] = 1.5
        result = boundary_pixels_to_ego(boundary, K, T)
        self.assertTrue(result["valid"])
        left = np.asarray(result["left_xy"])
        self.assertTrue(np.all(np.isfinite(left)))
        self.assertTrue(np.all(left[:, 0] > 0.0))

    def test_ego_boundary_fusion_shrinks_lateral_outlier(self):
        items = [{"valid": True, "left_xy": [[5.0, -1.0], [10.0, -1.0]], "right_xy": [[5.0, 1.0], [10.0, 1.0]]}, {"valid": True, "left_xy": [[5.0, -3.0], [10.0, -3.0]], "right_xy": [[5.0, 3.0], [10.0, 3.0]]}]
        fused, diag = fuse_ego_boundary_keypoints(items, max_lateral_jump_m=1.2, huber_scale_m=0.35)
        self.assertGreater(diag["clipped_fraction"], 0.0)
        self.assertGreater(float(fused[1]["left_xy"][0][1]), -3.0)

    def test_road_filter_smooths_center_offset(self):
        masks = np.zeros((4, 48, 96), dtype=bool)
        for index, offset in enumerate((0, 2, -2, 1)):
            for row in range(18, 48):
                left = max(5, 25 + offset + (48 - row) // 5)
                right = min(90, 70 + offset - (48 - row) // 5)
                masks[index, row, left:right] = True
        result = RoadStateFilter().update(masks)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["states"]), 4)
        self.assertIsNotNone(result["temporal_jitter_diagnostics"]["filtered_center_jitter_norm"])

    def test_global_flow_homography_recovers_translation(self):
        h, w = 48, 64
        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[..., 0] = 2.0
        flow[..., 1] = -1.0
        weights = np.ones((h, w), dtype=np.float32)
        matrix, ratio, count = HomographyBoundaryPropagator._global_flow_homography(
            flow, weights, max_points=1000, reprojection_threshold_px=0.1
        )
        self.assertIsNotNone(matrix)
        self.assertGreaterEqual(ratio, 0.99)
        self.assertGreaterEqual(count, 20)
        np.testing.assert_allclose(matrix[:2, 2], [2.0, -1.0], atol=1e-3)

    def test_global_flow_homography_can_restrict_to_road_support(self):
        h, w = 32, 48
        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[..., 0] = 2.0
        flow[..., 1] = -1.0
        road = np.zeros((h, w), dtype=bool)
        road[12:, 10:38] = True
        matrix, ratio, count = HomographyBoundaryPropagator._global_flow_homography(
            flow, np.ones((h, w), dtype=np.float32), support_mask=road,
            max_points=1000, reprojection_threshold_px=0.1
        )
        self.assertIsNotNone(matrix)
        self.assertEqual(count, int(road.sum()))
        self.assertGreaterEqual(ratio, 0.99)

    def test_scale_calibrator_returns_speed_interval(self):
        h, w, t = 24, 32, 3
        flow = np.zeros((t, h, w, 2), dtype=np.float64)
        flow[..., 0] = 2.0
        depth = np.full((t, h, w), 10.0, dtype=np.float64)
        weights = np.ones((t, h, w), dtype=np.float64)
        result = TemporalScaleCalibrator().estimate(
            observed_flows=flow,
            depths_m=depth,
            static_weights=weights,
            intrinsics=np.asarray([[100.0, 0.0, 16.0], [0.0, 100.0, 12.0], [0.0, 0.0, 1.0]]),
            future_times_s=np.asarray([0.5, 1.0, 1.5]),
            history_ego_state=np.asarray([[0.0, 0.0, 0.0, 4.0], [-1.0, 0.0, 0.0, 4.0]]),
            history_times_s=np.asarray([-0.5, 0.0]),
        )
        self.assertTrue(result["available"])
        self.assertEqual(len(result["rows"]), 3)
        self.assertIn("q05", result["rows"][0]["speed_interval_mps"])

    def test_low_support_abstains(self):
        flow = np.zeros((1, 10, 10, 2), dtype=np.float64)
        depth = np.full((1, 10, 10), 10.0, dtype=np.float64)
        weights = np.zeros((1, 10, 10), dtype=np.float64)
        weights[0, 0, 0] = 1.0
        result = TemporalScaleCalibrator(min_valid_fraction=0.2).estimate(
            observed_flows=flow,
            depths_m=depth,
            static_weights=weights,
            intrinsics=np.eye(3),
            future_times_s=np.asarray([0.5]),
            history_ego_state=None,
            history_times_s=None,
        )
        self.assertEqual(result["rows"][0]["speed_status"], "abstain")

    def test_missing_history_is_explicitly_uncalibrated(self):
        flow = np.ones((1, 12, 12, 2), dtype=np.float64)
        depth = np.full((1, 12, 12), 8.0, dtype=np.float64)
        result = TemporalScaleCalibrator().estimate(
            observed_flows=flow,
            depths_m=depth,
            static_weights=np.ones((1, 12, 12), dtype=np.float64),
            intrinsics=np.asarray([[100.0, 0.0, 6.0], [0.0, 100.0, 6.0], [0.0, 0.0, 1.0]]),
            future_times_s=np.asarray([0.5]),
            history_ego_state=None,
            history_times_s=None,
        )
        self.assertFalse(result["scale_calibrated"])
        self.assertEqual(result["rows"][0]["speed_status"], "uncalibrated")

    def test_history_prior_without_depth_is_not_scale_calibrated(self):
        flow = np.ones((1, 12, 12, 2), dtype=np.float64)
        result = TemporalScaleCalibrator().estimate(
            observed_flows=flow,
            depths_m=None,
            static_weights=np.ones((1, 12, 12), dtype=np.float64),
            intrinsics=np.eye(3),
            future_times_s=np.asarray([0.5]),
            history_ego_state=np.asarray([[0.0, 0.0, 0.0, 4.0]]),
            history_times_s=np.asarray([0.0]),
        )
        self.assertTrue(result["history_speed_prior_available"])
        self.assertFalse(result["scale_calibrated"])
        self.assertEqual(result["rows"][0]["speed_status"], "uncertain")

    def test_far_support_controls_observability_and_uncertainty(self):
        masks = np.zeros((2, 72, 128), dtype=bool)
        for row in range(20, 72):
            masks[:, row, 30:98] = True
        missing = masks.copy()
        missing[1, :46] = False
        clean = RoadStateFilter(far_support_reference_fraction=0.03).update(masks)
        degraded = RoadStateFilter(far_support_reference_fraction=0.03).update(missing)
        self.assertGreater(clean["states"][1]["far_range_observability"], degraded["states"][1]["far_range_observability"])
        self.assertGreater(degraded["states"][1]["lateral_uncertainty_m"], clean["states"][1]["lateral_uncertainty_m"])

    def test_flow_warp_propagator_uses_local_motion_without_homography(self):
        mask = np.zeros((2, 32, 48), dtype=bool)
        mask[:, 14:, 12:36] = True
        flow = np.zeros((1, 32, 48, 2), dtype=np.float32)
        flow[..., 0] = 1.0
        result = HomographyBoundaryPropagator(
            propagation_method="flow_warp",
            propagate_far_missing=False,
        ).update(
            mask,
            intrinsics=np.eye(3),
            camera_to_ego=np.eye(4),
            observed_flows=flow,
            static_weights=np.ones(flow.shape[:-1], dtype=np.float32),
            future_times_s=np.asarray([1.0, 2.0]),
        )
        self.assertTrue(result["propagation"][1]["flow_warp_used"])
        self.assertFalse(result["propagation"][1]["homography_used"])


if __name__ == "__main__":
    unittest.main()
