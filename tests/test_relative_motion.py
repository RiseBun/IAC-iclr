import unittest

import numpy as np

from iac_new.relative_motion import (
    ActorPixelTrack,
    ActorRelativeTrack,
    estimate_actor_relative_motion,
    evaluate_relative_motion_metrics,
    ground_contact_pixels_to_ego,
    project_actor_pixel_track,
    validate_actor_future_window,
)


class RelativeMotionTest(unittest.TestCase):
    def test_ground_contact_projection(self) -> None:
        intrinsics = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        camera_to_ego[:3, 3] = [0.0, 0.0, 1.5]
        points, valid = ground_contact_pixels_to_ego(
            np.asarray([[50.0, 65.0]]), intrinsics, camera_to_ego
        )
        self.assertTrue(valid[0])
        np.testing.assert_allclose(points[0], [10.0, 0.0, 0.0], atol=1e-6)

    def test_estimates_closing_and_lateral_speed(self) -> None:
        times = np.arange(0.5, 4.5, 0.5)
        positions = np.column_stack([20.0 - 3.0 * times, 3.0 - 0.5 * times])
        result = estimate_actor_relative_motion(ActorRelativeTrack(
            actor_id="vehicle-1",
            class_label="vehicle",
            times_s=times,
            positions_ego_m=positions,
            confidence=np.full(len(times), 0.9),
        ))
        self.assertTrue(result["available"])
        self.assertEqual(result["status"], "usable")
        first = result["support"][0]
        self.assertAlmostEqual(first["closing_speed_mps"]["q50"], 3.0, places=5)
        self.assertAlmostEqual(first["lateral_speed_mps"]["q50"], -0.5, places=5)
        self.assertAlmostEqual(first["ttc_s"], 18.5 / 3.0, places=5)
        self.assertAlmostEqual(first["corridor_conflict_ttc_s"], 18.5 / 3.0, places=5)
        self.assertTrue(result["support"][6]["inside_corridor"])
        self.assertEqual(result["support"][6]["time_to_corridor_s"], 0.0)
        self.assertAlmostEqual(result["support"][6]["corridor_conflict_ttc_s"], 3.1666666667, places=5)

        away = estimate_actor_relative_motion(ActorRelativeTrack(
            actor_id="vehicle-away",
            class_label="vehicle",
            times_s=times,
            positions_ego_m=np.column_stack([20.0 - 3.0 * times, 1.5 + 0.5 * times]),
        ))
        self.assertIsNone(away["support"][0]["corridor_conflict_ttc_s"])

    def test_abstains_without_temporal_support(self) -> None:
        result = estimate_actor_relative_motion(ActorRelativeTrack(
            actor_id="ped-1",
            class_label="pedestrian",
            times_s=np.asarray([0.5, 1.0]),
            positions_ego_m=np.asarray([[8.0, 2.0], [7.0, 1.5]]),
        ))
        self.assertFalse(result["available"])
        self.assertEqual(result["abstain_reason"], "insufficient_temporal_support")

    def test_pixel_track_projects_with_ground_and_depth_sources(self) -> None:
        intrinsics = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        camera_to_ego[:3, 3] = [0.0, 0.0, 1.5]
        track = ActorPixelTrack(
            actor_id="vehicle-2",
            class_label="vehicle",
            times_s=np.asarray([0.5, 1.0, 1.5]),
            pixels_uv=np.asarray([[50.0, 65.0], [50.0, 65.0], [50.0, 65.0]]),
        )
        relative, metadata = project_actor_pixel_track(
            track, intrinsics, camera_to_ego, depth_m=np.asarray([10.0, np.nan, 8.0])
        )
        np.testing.assert_allclose(relative.positions_ego_m[[0, 2], 0], [10.0, 8.0], atol=1e-6)
        self.assertEqual(metadata["projection_sources"], ["metric_depth", "ground_plane", "metric_depth"])
        self.assertEqual(metadata["candidate_bank_used"], False)

    def test_metric_report_includes_coverage_risk(self) -> None:
        rows = [
            {
                "predicted_distance_m": 10.0,
                "reference_distance_m": 11.0,
                "predicted_closing_speed_mps": 3.0,
                "reference_closing_speed_mps": 2.5,
                "predicted_ttc_s": 3.3,
                "reference_ttc_s": 4.4,
                "observability": 0.9,
                "abstain": False,
            },
            {
                "predicted_distance_m": 20.0,
                "reference_distance_m": 20.0,
                "predicted_closing_speed_mps": -1.0,
                "reference_closing_speed_mps": -1.0,
                "predicted_ttc_s": None,
                "reference_ttc_s": None,
                "observability": 0.5,
                "abstain": False,
            },
            {"abstain": True},
        ]
        report = evaluate_relative_motion_metrics(rows)
        self.assertAlmostEqual(report["coverage"], 2.0 / 3.0)
        self.assertAlmostEqual(report["distance_mae_m"], 0.5)
        self.assertAlmostEqual(report["closing_speed_mae_mps"], 0.25)
        self.assertEqual(len(report["coverage_risk"]), 4)
        self.assertEqual(report["num_reference_dangerous"], 0)
        self.assertIsNone(report["dangerous_ttc_recall"])

    def test_missing_metric_fields_fail_closed(self) -> None:
        report = evaluate_relative_motion_metrics([
            {"predicted_distance_m": 10.0, "abstain": False},
            {"predicted_distance_m": float("nan"), "abstain": False},
        ])
        self.assertEqual(report["num_scored"], 0)
        self.assertEqual(report["coverage"], 0.0)

    def test_formal_window_and_leakage_gate(self) -> None:
        times = validate_actor_future_window(np.arange(0.5, 4.5, 0.5))
        self.assertEqual(len(times), 8)
        with self.assertRaises(ValueError):
            validate_actor_future_window(np.asarray([0.5, 1.0, 1.5, 2.0]))
        report = evaluate_relative_motion_metrics([{
            "predicted_distance_m": 1.0,
            "reference_distance_m": 1.0,
            "predicted_closing_speed_mps": 1.0,
            "reference_closing_speed_mps": 1.0,
            "candidate_bank_used": True,
        }])
        self.assertEqual(report["num_leakage_excluded"], 1)
        self.assertFalse(report["formal_ready"])
        self.assertEqual(report["num_scored"], 0)


if __name__ == "__main__":
    unittest.main()
