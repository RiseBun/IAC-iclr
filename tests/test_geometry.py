import unittest

import numpy as np

from iac_new.geometry import (
    candidate_camera_poses,
    ground_plane_homography,
    homography_flow,
    rigid_flow_from_depth,
    se2_to_transform,
)


class GeometryTest(unittest.TestCase):
    def test_se2_translation_and_yaw(self) -> None:
        transform = se2_to_transform(2.0, -1.0, np.pi / 2.0)
        self.assertTrue(
            np.allclose(transform[:2, :2], [[0.0, -1.0], [1.0, 0.0]], atol=1e-7)
        )
        self.assertTrue(np.allclose(transform[:3, 3], [2.0, -1.0, 0.0]))

    def test_identity_ground_plane_has_zero_flow(self) -> None:
        intrinsics = np.asarray(
            [[100.0, 0.0, 20.0], [0.0, 100.0, 15.0], [0.0, 0.0, 1.0]]
        )
        camera_to_ego = np.eye(4)
        camera_to_ego[2, 3] = 1.5
        poses = candidate_camera_poses(np.asarray([[0.0, 0.0, 0.0]]), camera_to_ego)
        transform = np.linalg.inv(poses[1]) @ poses[0]
        homography = ground_plane_homography(intrinsics, transform, poses[0])
        flow, valid = homography_flow(homography, 30, 40)
        self.assertTrue(valid.all())
        self.assertLess(np.nanmax(np.abs(flow)), 1e-6)

    def test_candidate_pose_count_matches_knots(self) -> None:
        camera_to_ego = np.eye(4)
        poses = candidate_camera_poses(
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.1, 0.01]]), camera_to_ego
        )
        self.assertEqual(len(poses), 3)

    def test_identity_metric_depth_has_zero_flow(self) -> None:
        intrinsics = np.asarray(
            [[100.0, 0.0, 20.0], [0.0, 100.0, 15.0], [0.0, 0.0, 1.0]]
        )
        flow, valid = rigid_flow_from_depth(
            np.full((30, 40), 12.0, dtype=np.float32), intrinsics, np.eye(4)
        )
        self.assertTrue(valid[1:-1, 1:-1].all())
        self.assertGreater(float(valid.mean()), 0.9)
        self.assertLess(float(np.nanmax(np.abs(flow))), 1e-6)


if __name__ == "__main__":
    unittest.main()
