import unittest

import numpy as np

from scripts.export_navsim_records import _camera_to_ego, _lidar_path


class ExportNavsimRecordsTest(unittest.TestCase):
    def test_lidar_path_requires_existing_blob(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "log" / "MergedPointCloud" / "sample.pcd"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"pcd")
            self.assertEqual(
                _lidar_path({"lidar_path": "log/MergedPointCloud/sample.pcd"}, root),
                blob,
            )
            with self.assertRaises(FileNotFoundError):
                _lidar_path({"lidar_path": "missing.pcd"}, root)

    def test_camera_to_ego_composes_lidar_extrinsic(self) -> None:
        lidar_to_ego = np.eye(4, dtype=np.float64)
        lidar_to_ego[:3, 3] = [0.4, -0.2, 0.1]
        camera_record = {
            "sensor2lidar_rotation": np.eye(3).tolist(),
            "sensor2lidar_translation": [1.0, 2.0, 3.0],
        }

        result = _camera_to_ego({"lidar2ego": lidar_to_ego.tolist()}, camera_record)

        np.testing.assert_allclose(result[:3, 3], [1.4, 1.8, 3.1])


if __name__ == "__main__":
    unittest.main()
