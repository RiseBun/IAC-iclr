import unittest

import numpy as np

from iac_new.cotracker import (
    PointTrackObservation,
    actor_box_query_points,
    actor_pixel_tracks_from_observation,
    aggregate_actor_region_tracks,
    point_track_curvature_features,
    validate_query_frame_indices,
    validate_query_points,
)


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

    def test_custom_actor_queries_are_validated_without_loading_model(self) -> None:
        points = validate_query_points(
            np.asarray([[12.0, 20.0], [200.0, 100.0]]), height=288, width=512
        )
        self.assertEqual(points.shape, (2, 2))
        with self.assertRaises(ValueError):
            validate_query_points(np.asarray([[512.0, 20.0]]), height=288, width=512)

    def test_queries_can_start_when_an_actor_first_becomes_visible(self) -> None:
        frames = validate_query_frame_indices(
            np.asarray([0, 3]), num_queries=2, num_frames=8
        )
        np.testing.assert_array_equal(frames, [0, 3])
        with self.assertRaises(ValueError):
            validate_query_frame_indices(
                np.asarray([0.5]), num_queries=1, num_frames=8
            )
        with self.assertRaises(ValueError):
            validate_query_frame_indices(
                np.asarray([8]), num_queries=1, num_frames=8
            )

    def test_actor_region_consensus_moves_ground_anchor(self) -> None:
        query = actor_box_query_points(
            np.asarray([100.0, 80.0, 160.0, 200.0]), height=288, width=512,
            columns=2, rows=2,
        )
        tracks = np.repeat(query[None, :, :], 3, axis=0)
        tracks[1] += [5.0, -2.0]
        tracks[2] += [10.0, -4.0]
        visibility = np.ones((3, len(query)), dtype=np.float32)
        confidence = np.ones_like(visibility)
        observation = PointTrackObservation(
            tracks=tracks, visibility=visibility, confidence=confidence,
            query_points=query, source_size=(512, 288), target_size=(512, 288),
        )
        anchors, valid, scores = aggregate_actor_region_tracks(
            observation, np.asarray([130.0, 200.0])
        )
        np.testing.assert_allclose(anchors[2], [140.0, 196.0], atol=1e-4)
        self.assertTrue(valid.all())
        self.assertTrue((scores > 0.9).all())

    def test_actor_queries_keep_tracker_visibility_and_confidence(self) -> None:
        tracks = np.asarray([
            [[20.0, 30.0], [40.0, 50.0]],
            [[21.0, 30.5], [41.0, 50.5]],
            [[22.0, 31.0], [42.0, 51.0]],
        ], dtype=np.float32)
        visibility = np.asarray([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        confidence = np.asarray([[0.9, 0.8], [0.9, 0.1], [0.9, 0.7]], dtype=np.float32)
        observation = PointTrackObservation(
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            query_points=tracks[0],
            source_size=(512, 288),
            target_size=(512, 288),
        )
        result = actor_pixel_tracks_from_observation(
            observation,
            [{"actor_id": "vehicle-1", "class_label": "vehicle", "query_index": 1}],
            np.asarray([0.5, 1.0, 1.5]),
        )
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].pixels_uv[1], [41.0, 50.5])
        self.assertFalse(result[0].visibility[1])


if __name__ == "__main__":
    unittest.main()
