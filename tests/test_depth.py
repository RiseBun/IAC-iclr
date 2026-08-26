import tempfile
import unittest
from pathlib import Path

import numpy as np

from iac_new.depth import load_cached_metric_depth, metric_depth_reliability_masks


class MetricDepthTest(unittest.TestCase):
    def _write_cache(self, path: Path, intrinsics: np.ndarray, camera: np.ndarray) -> None:
        np.savez_compressed(
            path,
            depth_m=np.full((2, 6, 8), 10.0, dtype=np.float16),
            confidence=np.arange(96, dtype=np.float32).reshape(2, 6, 8),
            intrinsics=intrinsics,
            target_size=np.asarray([8, 6]),
            camera_to_ego=camera,
        )

    def test_loads_scales_and_masks_cached_depth(self) -> None:
        intrinsics = np.asarray([[20.0, 0.0, 4.0], [0.0, 20.0, 3.0], [0.0, 0.0, 1.0]])
        camera = np.eye(4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.npz"
            self._write_cache(path, intrinsics, camera)
            observation = load_cached_metric_depth(
                {"sample_id": "s", "metric_depth_path": str(path), "camera_to_ego": camera},
                {"depth_scale_divisor": 2.0},
                expected_intervals=2,
                expected_size=(8, 6),
                expected_intrinsics=intrinsics,
            )
        self.assertTrue(np.allclose(observation.depths_m, 5.0))
        masks, diagnostics = metric_depth_reliability_masks(
            observation,
            np.zeros((2, 6, 8, 2), dtype=np.float32),
            min_depth_m=1.0,
            max_depth_m=100.0,
            confidence_quantile=0.25,
            observed_flow_quantile=0.997,
        )
        self.assertEqual(masks.shape, (2, 6, 8))
        self.assertGreater(diagnostics[0]["valid_fraction"], 0.7)

    def test_rejects_cache_extrinsic_mismatch(self) -> None:
        intrinsics = np.eye(3)
        camera = np.eye(4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.npz"
            self._write_cache(path, intrinsics, camera)
            wrong = camera.copy()
            wrong[0, 3] = 1.0
            with self.assertRaisesRegex(ValueError, "camera_to_ego"):
                load_cached_metric_depth(
                    {"sample_id": "s", "metric_depth_path": str(path), "camera_to_ego": wrong},
                    {},
                    expected_intervals=2,
                    expected_size=(8, 6),
                    expected_intrinsics=intrinsics,
                )


if __name__ == "__main__":
    unittest.main()
