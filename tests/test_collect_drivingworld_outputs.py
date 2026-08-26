import tempfile
import unittest
from pathlib import Path

from scripts.collect_drivingworld_outputs import collect


class CollectDrivingWorldOutputsTest(unittest.TestCase):
    def test_collects_numbered_future_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(15, 20):
                (root / f"{index}.png").write_bytes(b"png")
            rows = collect([{"branch_id": "b", "input_dir": str(root), "future_action_targets": 5}])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["generated_future_images"][-1], str((root / "19.png").resolve()))
            self.assertEqual(rows[0]["wam_generation_metadata"]["context_frames"], 15)

    def test_maps_external_sliding_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "outputs" / "exp" / "sliding_0"
            generated.mkdir(parents=True)
            for index in range(15, 20):
                (generated / f"{index}.png").write_bytes(b"png")
            rows = collect([{"branch_id": "b", "input_dir": str(root), "future_action_targets": 5}], generated_root=root / "outputs", experiment_name="exp")
            self.assertEqual(rows[0]["generated_future_images"][0], str((generated / "15.png").resolve()))


if __name__ == "__main__":
    unittest.main()
