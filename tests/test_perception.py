import sys
import types
import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from iac_new.perception import (
    PerceptionObservation,
    semantic_motion_summary,
    temporal_road_consensus,
    trajectory_traversability,
)


def _front_camera_to_ego() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    transform[:3, 3] = [0.0, 0.0, 1.5]
    return transform


class PerceptionTest(unittest.TestCase):
    def test_temporal_road_consensus_follows_curved_image_support(self) -> None:
        road = np.zeros((2, 8, 10), dtype=bool)
        road[0, 3:, 2:6] = True
        road[1, 3:, 3:7] = True
        actors = np.zeros_like(road)
        actors[1, 5:7, 5:7] = True
        flow = np.zeros((1, 8, 10, 2), dtype=np.float32)
        flow[..., 0] = 1.0
        observation = PerceptionObservation(
            traversable_masks=road,
            actor_masks=actors,
            class_maps=np.zeros_like(road, dtype=np.int16),
            class_names=("road", "car"),
            backend="synthetic",
            model_id="synthetic",
        )

        result = temporal_road_consensus(
            observation, flow, road_dilation_px=0, actor_dilation_px=0
        )

        np.testing.assert_array_equal(result.traversable_masks[0], road[0])
        self.assertTrue(result.actor_masks[0, 5, 4])
        self.assertIn("temporal_consensus", result.backend)
    def test_semantic_motion_keeps_actor_and_motion_evidence_separate(self) -> None:
        actors = np.zeros((1, 4, 5), dtype=bool)
        actors[0, 1:3, 1:3] = True
        weights = np.ones((1, 4, 5), dtype=np.float32)
        weights[0, 1, 1] = 0.1
        weights[0, 2, 2] = 0.1
        result = semantic_motion_summary(actors, weights)
        self.assertAlmostEqual(result[0]["actor_fraction"], 4.0 / 20.0)
        self.assertAlmostEqual(result[0]["actor_dynamic_fraction"], 0.5)
        self.assertEqual(result[0]["classification"], "likely_dynamic_actor")

    def test_traversability_prefers_corridor_inside_road_mask(self) -> None:
        height, width = 120, 200
        intrinsics = np.asarray(
            [[120.0, 0.0, width / 2], [0.0, 120.0, height / 2], [0.0, 0.0, 1.0]]
        )
        road = np.zeros((height, width), dtype=bool)
        road[:, 65:135] = True
        straight = np.asarray([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        outside = np.asarray([[4.0, 4.0, 0.0], [8.0, 4.0, 0.0]])
        straight_score = trajectory_traversability(
            straight, road, _front_camera_to_ego(), intrinsics
        )
        outside_score = trajectory_traversability(
            outside, road, _front_camera_to_ego(), intrinsics
        )
        self.assertGreater(straight_score["traversable_fraction"], 0.9)
        self.assertLess(outside_score["traversable_fraction"], 0.5)
        self.assertGreater(straight_score["visible_corridor_samples"], 0)


if __name__ == "__main__":
    unittest.main()
