import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from iac_new.evaluate import evaluate_record
from iac_new.perception import PerceptionObservation
from iac_new.scoring import predict_candidate_flows


class _SyntheticExtractor:
    model_size = "synthetic"
    use_forward_backward = False

    def __init__(self, observation: SimpleNamespace) -> None:
        self.observation = observation

    def observe(self, *args, **kwargs) -> SimpleNamespace:
        return self.observation


class _SyntheticPerception:
    def observe(self, frame_paths, *, target_size, intrinsics, distortion):
        width, height = target_size
        count = len(frame_paths)
        return PerceptionObservation(
            traversable_masks=np.ones((count, height, width), dtype=bool),
            actor_masks=np.zeros((count, height, width), dtype=bool),
            class_maps=np.zeros((count, height, width), dtype=np.int16),
            class_names=("road",),
            backend="synthetic",
            model_id="synthetic",
        )


class EvaluateTest(unittest.TestCase):
    def test_cached_metric_depth_backend_selects_exact_candidate(self) -> None:
        height, width = 60, 80
        intrinsics = np.asarray(
            [[120.0, 0.0, width / 2], [0.0, 120.0, height / 2], [0.0, 0.0, 1.0]]
        )
        camera_to_ego = np.eye(4)
        camera_to_ego[2, 3] = 1.5
        exact = np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        wrong = exact * 2.0
        depths = np.full((2, height, width), 12.0, dtype=np.float32)
        future_flow, _ = predict_candidate_flows(
            trajectory=exact,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            height=height,
            width=width,
            depths_m=depths,
        )
        observation = SimpleNamespace(
            forward=future_flow,
            consistency_masks=None,
            intrinsics=intrinsics,
            source_size=(width, height),
            target_size=(width, height),
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "depth.npz"
            np.savez_compressed(
                cache_path,
                depth_m=depths,
                confidence=np.ones_like(depths),
                intrinsics=intrinsics,
                target_size=np.asarray([width, height]),
                camera_to_ego=camera_to_ego,
            )
            record = {
                "sample_id": "metric",
                "scene_id": "metric",
                "frame_paths": ["anchor.jpg", "f1.jpg", "f2.jpg"],
                "frame_times_s": np.asarray([0.0, 0.5, 1.0]),
                "future_times_s": np.asarray([0.5, 1.0]),
                "anchor_time_s": 0.0,
                "history_count": 1,
                "protocol_variant": "legacy_anchor_future",
                "intrinsics": intrinsics,
                "distortion": np.asarray([]),
                "camera_to_ego": camera_to_ego,
                "metric_depth_path": str(cache_path),
                "metric_depth_source": "synthetic",
                "candidates": [
                    {"candidate_id": "exact", "trajectory": exact, "prior": 1.0},
                    {"candidate_id": "wrong", "trajectory": wrong, "prior": 1.0},
                ],
                "gt_candidate_id": "exact",
                "metadata": {},
            }
            config = {
                "image": {"width": width, "height": height},
                "geometry": {
                    "backend": "cached_metric_depth",
                    "min_depth_m": 1.0,
                    "max_depth_m": 100.0,
                    "confidence_quantile": 0.0,
                    "observed_flow_quantile": 1.0,
                },
                "mask": {
                    "polygon_normalized": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    "min_valid_pixels": 100,
                },
                "dynamic_suppression": {"enabled": False, "minimum_weight": 0.05},
                "observability": {"min_effective_pixel_fraction": 0.01},
                "score": {
                    "energy_metric": "median_epe_px",
                    "temperature": 1.0,
                    "target_coverage": 0.9,
                    "minimum_flow_scale_px": 0.1,
                },
            }
            result = evaluate_record(record, _SyntheticExtractor(observation), config, None)
        self.assertEqual(result["geometry_backend"], "cached_metric_depth")
        self.assertEqual(result["top_candidate_id"], "exact")
        self.assertLess(result["candidate_scores"][0]["energy"], 1e-5)

    def test_optional_perception_constrains_geometry_and_is_reported(self) -> None:
        height, width = 60, 80
        intrinsics = np.asarray(
            [[120.0, 0.0, width / 2], [0.0, 120.0, height / 2], [0.0, 0.0, 1.0]]
        )
        camera_to_ego = np.eye(4)
        camera_to_ego[2, 3] = 1.5
        exact = np.asarray([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        future_flow, _ = predict_candidate_flows(
            trajectory=exact,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            height=height,
            width=width,
        )
        observation = SimpleNamespace(
            forward=future_flow,
            consistency_masks=None,
            intrinsics=intrinsics,
            source_size=(width, height),
            target_size=(width, height),
        )
        record = {
            "sample_id": "perception",
            "scene_id": "perception",
            "frame_paths": [f"frame_{index}.jpg" for index in range(3)],
            "frame_times_s": np.asarray([0.0, 0.5, 1.0]),
            "future_times_s": np.asarray([0.5, 1.0]),
            "anchor_time_s": 0.0,
            "history_count": 1,
            "protocol_variant": "legacy_anchor_future",
            "intrinsics": intrinsics,
            "distortion": np.asarray([]),
            "camera_to_ego": camera_to_ego,
            "candidates": [
                {"candidate_id": "exact", "trajectory": exact, "prior": 1.0},
                {"candidate_id": "wrong", "trajectory": exact * 2.0, "prior": 1.0},
            ],
            "gt_candidate_id": "exact",
            "metadata": {},
        }
        config = {
            "image": {"width": width, "height": height},
            "perception": {"enabled": True, "use_traversable_mask": True},
            "mask": {
                "polygon_normalized": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                "min_valid_pixels": 100,
            },
            "dynamic_suppression": {"enabled": False, "minimum_weight": 0.05},
            "observability": {"min_effective_pixel_fraction": 0.01},
            "score": {
                "energy_metric": "median_epe_px",
                "temperature": 1.0,
                "target_coverage": 0.9,
                "minimum_flow_scale_px": 0.1,
            },
        }
        result = evaluate_record(
            record, _SyntheticExtractor(observation), config, None, perception=_SyntheticPerception()
        )
        self.assertEqual(result["perception"]["backend"], "synthetic")
        self.assertTrue(result["perception"]["used_as_geometry_constraint"])
        self.assertIn("semantic_feasibility", result["mode_summaries"][0])

    def test_full_record_outputs_joint_region_and_seven_intervals(self) -> None:
        height, width = 60, 80
        intrinsics = np.asarray(
            [[120.0, 0.0, width / 2], [0.0, 120.0, height / 2], [0.0, 0.0, 1.0]]
        )
        camera_to_ego = np.eye(4)
        camera_to_ego[2, 3] = 1.5
        exact = np.asarray(
            [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0], [0.8, 0.0, 0.0]]
        )
        wrong = exact.copy()
        wrong[:, 1] = [0.05, 0.12, 0.22, 0.35]
        future_flow, _ = predict_candidate_flows(
            trajectory=exact,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            height=height,
            width=width,
        )
        all_flow = np.concatenate([np.zeros_like(future_flow[:1]).repeat(3, axis=0), future_flow])
        observation = SimpleNamespace(
            forward=all_flow,
            consistency_masks=None,
            intrinsics=intrinsics,
            source_size=(width, height),
            target_size=(width, height),
        )
        record = {
            "sample_id": "synthetic",
            "scene_id": "synthetic",
            "frame_paths": [f"frame_{index}.jpg" for index in range(8)],
            "frame_times_s": np.asarray([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
            "future_times_s": np.asarray([0.5, 1.0, 1.5, 2.0]),
            "anchor_time_s": 0.0,
            "history_count": 4,
            "protocol_variant": "history4_future4",
            "intrinsics": intrinsics,
            "distortion": np.asarray([]),
            "camera_to_ego": camera_to_ego,
            "candidates": [
                {"candidate_id": "exact", "trajectory": exact, "prior": 1.0},
                {"candidate_id": "wrong", "trajectory": wrong, "prior": 1.0},
            ],
            "gt_candidate_id": "exact",
            "metadata": {},
        }
        config = {
            "image": {"width": width, "height": height},
            "mask": {
                "polygon_normalized": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                "min_valid_pixels": 100,
            },
            "dynamic_suppression": {
                "enabled": True,
                "absolute_threshold_px": 2.0,
                "relative_threshold": 0.1,
                "minimum_weight": 0.05,
            },
            "observability": {
                "static_weight_threshold": 0.5,
                "min_effective_pixel_fraction": 0.02,
                "boundary_relative_threshold": 0.35,
            },
            "score": {
                "temperature": 0.2,
                "target_coverage": 0.9,
                "minimum_flow_scale_px": 0.1,
            },
        }
        result = evaluate_record(record, _SyntheticExtractor(observation), config, None)
        self.assertEqual(result["top_candidate_id"], "exact")
        self.assertTrue(result["valid"])
        self.assertIn("low_flow_magnitude", result["observability"][0]["status"])
        self.assertEqual(len(result["observability"]), 7)
        self.assertEqual(len(result["mode_summaries"]), 2)
        self.assertEqual(
            len(result["trajectory_region"]["joint_lateral_yaw_curvature"]), 4
        )
        self.assertAlmostEqual(result["mode_summaries"][0]["speed_range_mps"][0], 0.4)


if __name__ == "__main__":
    unittest.main()
