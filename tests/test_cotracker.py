import unittest

import numpy as np

from iac_new.cotracker import PointTrackObservation, point_track_curvature_features


class CoTrackerFeatureTest(unittest.TestCase):
    def test_point_tracks_report_lateral_contrast_and_future_intervals(self) -> None:
        points = np.asarray(
            [[x, y] for y in (160.0, 220.0) for x in (80.0, 180.0, 300.0, 420.0)],
            dtype=np.float32,
        )
        tracks = np.repeat(points[None, ...], 5, axis=0)
        for frame_index in range(1, len(tracks)):
            tracks[frame_index, :, 1] += 4.0 * frame_index
            for point_index, point in enumerate(points):
                if point[0] < 200.0:
                    tracks[frame_index, point_index, 0] -= 1.0 * frame_index
                elif point[0] > 300.0:
                    tracks[frame_index, point_index, 0] += 1.0 * frame_index
        visibility = np.ones((5, len(points)), dtype=np.float32)
        confidence = np.ones_like(visibility)
        observation = PointTrackObservation(
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            query_points=points,
            source_size=(512, 288),
            target_size=(512, 288),
        )
        result = point_track_curvature_features(observation, future_start_interval=2)
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0]["curvature_lateral_contrast_rad"], 0.1)
        self.assertEqual(result[0]["interval_index"], 2)


if __name__ == "__main__":
    unittest.main()
