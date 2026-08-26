import unittest

import numpy as np

from iac_new.scoring import (
    dynamic_suppression_weights,
    finite_sample_quantile,
    interval_observability,
    mass_prediction_set,
    posterior_from_energies,
    polygon_mask,
    predict_candidate_flows,
    score_candidate,
)
from iac_new.region import build_trajectory_region
from iac_new.geometry import (
    adjacent_camera_transforms,
    candidate_camera_poses,
    ground_plane_homography,
    homography_flow,
)


class ScoringTest(unittest.TestCase):
    def test_speed_observability_is_diagnostic_only(self) -> None:
        observed = np.zeros((2, 4, 4, 2), dtype=np.float32)
        observed[0, ..., 0] = 2.0
        observed[1, ..., 0] = 0.2
        result = interval_observability(
            observed_flows=observed,
            roi_mask=np.ones((4, 4), dtype=bool),
            consistency_masks=None,
            dynamic_weights=None,
            minimum_flow_scale_px=0.5,
            static_weight_threshold=0.5,
            min_effective_pixel_fraction=0.10,
            role="future",
        )
        self.assertTrue(result[0]["direction_observable"])
        self.assertEqual(result[0]["speed_status"], "usable")
        self.assertFalse(result[1]["direction_observable"])
        self.assertEqual(result[1]["speed_status"], "abstain")

    def test_curvature_observability_reports_lateral_structure(self) -> None:
        observed = np.zeros((2, 10, 20, 2), dtype=np.float32)
        observed[..., 1] = 4.0
        observed[:, :, :10, 0] = -1.0
        observed[:, :, 10:, 0] = 1.0
        result = interval_observability(
            observed_flows=observed,
            roi_mask=np.ones((10, 20), dtype=bool),
            consistency_masks=None,
            dynamic_weights=None,
            minimum_flow_scale_px=0.5,
            static_weight_threshold=0.5,
            min_effective_pixel_fraction=0.10,
            role="future",
            curvature_min_lateral_contrast_rad=0.1,
            curvature_min_flow_gradient_px=0.01,
            curvature_reliable_lateral_contrast_rad=0.1,
        )
        self.assertTrue(result[0]["curvature_spatial_evidence"])
        self.assertGreater(result[0]["curvature_lateral_contrast_rad"], 0.1)
        self.assertGreater(result[0]["curvature_flow_gradient_px"], 0.01)
        self.assertEqual(result[0]["curvature_status"], "usable")
        self.assertAlmostEqual(result[0]["curvature_confidence"], 1.0)

    def test_posterior_prefers_lower_energy(self) -> None:
        probabilities = posterior_from_energies(
            np.asarray([0.1, 1.0, 2.0]), temperature=0.2
        )
        self.assertEqual(int(np.argmax(probabilities)), 0)
        self.assertTrue(np.isclose(probabilities.sum(), 1.0))

    def test_mass_set_contains_top_mode(self) -> None:
        selected = mass_prediction_set(np.asarray([0.8, 0.1, 0.1]), 0.9)
        self.assertEqual(selected, [0, 1])

    def test_finite_sample_quantile_is_conservative(self) -> None:
        threshold = finite_sample_quantile(np.asarray([0.1, 0.2, 0.3]), 0.9)
        self.assertEqual(threshold, 0.3)

    def test_polygon_mask_has_inside_and_outside(self) -> None:
        mask = polygon_mask(
            100, 100, [[0.1, 0.9], [0.9, 0.9], [0.6, 0.5], [0.4, 0.5]]
        )
        self.assertTrue(mask[80, 50])
        self.assertFalse(mask[10, 50])

    def test_exact_planar_candidate_has_lower_energy(self) -> None:
        height, width = 60, 80
        intrinsics = np.asarray(
            [[120.0, 0.0, width / 2], [0.0, 120.0, height / 2], [0.0, 0.0, 1.0]]
        )
        camera_to_ego = np.eye(4)
        camera_to_ego[2, 3] = 1.5
        exact = np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        poses = candidate_camera_poses(exact, camera_to_ego)
        transforms = adjacent_camera_transforms(poses)
        observed = []
        for pose, transform in zip(poses[:-1], transforms):
            homography = ground_plane_homography(intrinsics, transform, pose)
            flow, _ = homography_flow(homography, height, width)
            observed.append(flow)
        observed_flows = np.stack(observed)
        roi = np.ones((height, width), dtype=bool)
        exact_score = score_candidate(
            candidate_id="exact",
            trajectory=exact,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            observed_flows=observed_flows,
            roi_mask=roi,
            consistency_masks=None,
            min_valid_pixels=100,
            minimum_flow_scale_px=1.0,
        )
        wrong_score = score_candidate(
            candidate_id="wrong",
            trajectory=np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            observed_flows=observed_flows,
            roi_mask=roi,
            consistency_masks=None,
            min_valid_pixels=100,
            minimum_flow_scale_px=1.0,
        )
        self.assertLess(exact_score.energy, wrong_score.energy)
        self.assertLess(exact_score.median_epe_px, 1e-5)

    def test_dynamic_weights_are_shared_and_suppress_unexplained_pixels(self) -> None:
        observed = np.zeros((1, 4, 5, 2), dtype=np.float32)
        observed[:, 1, 2] = [20.0, 0.0]
        predicted = np.zeros((2, 1, 4, 5, 2), dtype=np.float32)
        weights, residual = dynamic_suppression_weights(
            observed_flows=observed,
            predicted_flows=predicted,
            roi_mask=np.ones((4, 5), dtype=bool),
            consistency_masks=None,
            absolute_threshold_px=2.0,
            relative_threshold=0.0,
        )
        self.assertEqual(weights.shape, (1, 4, 5))
        self.assertLess(float(weights[0, 1, 2]), float(weights[0, 0, 0]))
        self.assertTrue(np.allclose(residual[0, 1, 2], 20.0))

    def test_common_geometry_mask_gives_candidates_the_same_support(self) -> None:
        height, width = 60, 80
        intrinsics = np.asarray(
            [[120.0, 0.0, width / 2], [0.0, 120.0, height / 2], [0.0, 0.0, 1.0]]
        )
        camera_to_ego = np.eye(4)
        camera_to_ego[2, 3] = 1.5
        exact = np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        observed, exact_valid = predict_candidate_flows(
            trajectory=exact,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            height=height,
            width=width,
        )
        faster = exact * 2.0
        faster_flow, faster_valid = predict_candidate_flows(
            trajectory=faster,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            height=height,
            width=width,
        )
        common = exact_valid & faster_valid
        exact_score = score_candidate(
            candidate_id="exact",
            trajectory=exact,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            observed_flows=observed,
            roi_mask=np.ones((height, width), dtype=bool),
            consistency_masks=None,
            min_valid_pixels=100,
            minimum_flow_scale_px=1.0,
            predicted_flows=observed,
            common_geometry_masks=common,
        )
        faster_score = score_candidate(
            candidate_id="faster",
            trajectory=faster,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            observed_flows=observed,
            roi_mask=np.ones((height, width), dtype=bool),
            consistency_masks=None,
            min_valid_pixels=100,
            minimum_flow_scale_px=1.0,
            predicted_flows=faster_flow,
            common_geometry_masks=common,
        )
        self.assertEqual(
            exact_score.intervals[0]["valid_pixel_fraction"],
            faster_score.intervals[0]["valid_pixel_fraction"],
        )

    def test_region_keeps_lateral_yaw_curvature_joint_points(self) -> None:
        candidates = [
            {"candidate_id": "straight", "trajectory": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])},
            {"candidate_id": "left", "trajectory": np.asarray([[1.0, 0.2, 0.1], [2.0, 0.5, 0.2]])},
        ]
        modes, region = build_trajectory_region(
            candidates=candidates,
            probabilities=np.asarray([0.7, 0.3]),
            selected_indices=[0, 1],
            future_times_s=np.asarray([0.5, 1.0]),
            target_coverage=0.9,
        )
        self.assertEqual(modes[0]["speed_range_mps"], [2.0, 2.0])
        point = region["joint_lateral_yaw_curvature"][1]["joint_support"][1]
        self.assertIn("lateral_y_m", point)
        self.assertIn("yaw_rad", point)
        self.assertIn("curvature_1pm", point)
        self.assertIn("continuous_support", region)
        self.assertIn("knotwise_quantiles", region["continuous_support"])


if __name__ == "__main__":
    unittest.main()
