import unittest

from iac_new.check_splits import split_keys


class SplitTest(unittest.TestCase):
    def test_split_keys_capture_scene_sample_and_frame(self) -> None:
        scenes, samples, frames = split_keys(
            [{"scene_id": "scene", "sample_id": "sample", "frame_paths": ["a.jpg"]}]
        )
        self.assertEqual(scenes, {"scene"})
        self.assertEqual(samples, {"sample"})
        self.assertEqual(frames, {"a.jpg"})

    def test_split_keys_capture_explicit_history_and_future(self) -> None:
        _, _, frames = split_keys(
            [{
                "scene_id": "scene",
                "sample_id": "sample",
                "history_frame_paths": ["h.jpg"],
                "future_frame_paths": ["f.jpg"],
            }]
        )
        self.assertEqual(frames, {"h.jpg", "f.jpg"})


if __name__ == "__main__":
    unittest.main()
