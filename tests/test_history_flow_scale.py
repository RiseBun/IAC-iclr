import numpy as np

from iac_new.temporal_geometry import estimate_history_flow_scale


def test_history_flow_scale_fails_closed_without_static_points():
    result = estimate_history_flow_scale(
        history_flows=np.zeros((3, 8, 8, 2), dtype=np.float64),
        history_ego_state=np.zeros((4, 5), dtype=np.float64),
        history_times_s=np.asarray([-1.5, -1.0, -0.5, 0.0]),
        camera_to_ego=np.eye(4),
        intrinsics=np.asarray([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]),
        roi_mask=np.ones((8, 8), dtype=bool),
    )
    assert result["available"] is False


def test_history_flow_scale_validates_history_shapes():
    try:
        estimate_history_flow_scale(
            history_flows=np.zeros((3, 8, 8, 2), dtype=np.float64),
            history_ego_state=np.zeros((3, 5), dtype=np.float64),
            history_times_s=np.asarray([-1.0, 0.0, 1.0]),
            camera_to_ego=np.eye(4),
            intrinsics=np.eye(3),
            roi_mask=np.ones((8, 8), dtype=bool),
        )
    except ValueError as error:
        assert "history_ego_state" in str(error)
    else:
        raise AssertionError("expected shape validation failure")


def test_history_flow_scale_rejects_invalid_shrinkage():
    try:
        estimate_history_flow_scale(
            history_flows=np.zeros((3, 8, 8, 2), dtype=np.float64),
            history_ego_state=np.zeros((4, 5), dtype=np.float64),
            history_times_s=np.asarray([-1.5, -1.0, -0.5, 0.0]),
            camera_to_ego=np.eye(4),
            intrinsics=np.eye(3),
            roi_mask=np.ones((8, 8), dtype=bool),
            correction_shrinkage=1.5,
        )
    except ValueError as error:
        assert "correction_shrinkage" in str(error)
    else:
        raise AssertionError("expected shrinkage validation failure")
