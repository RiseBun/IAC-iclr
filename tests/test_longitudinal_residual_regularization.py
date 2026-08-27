import numpy as np

from iac_new.trajectory_decode import _longitudinal_residual_penalty


def test_second_difference_regularizer_penalizes_drift_but_not_constant_residual():
    history = np.zeros(8, dtype=np.float64)
    constant = np.full(8, 0.5, dtype=np.float64)
    drifting = np.arange(8, dtype=np.float64) * 0.05
    constant_penalty = _longitudinal_residual_penalty(
        constant,
        history,
        maximum_residual_mps=3.0,
        residual_weight=0.0,
        residual_smoothness_weight=0.0,
        residual_curvature_weight=1.0,
    )
    drifting_penalty = _longitudinal_residual_penalty(
        drifting,
        history,
        maximum_residual_mps=3.0,
        residual_weight=0.0,
        residual_smoothness_weight=0.0,
        residual_curvature_weight=1.0,
    )
    assert constant_penalty == 0.0
    assert drifting_penalty == 0.0


def test_second_difference_regularizer_penalizes_curved_residual():
    history = np.zeros(8, dtype=np.float64)
    curved = np.arange(8, dtype=np.float64) ** 2 * 0.01
    penalty = _longitudinal_residual_penalty(
        curved,
        history,
        maximum_residual_mps=3.0,
        residual_weight=0.0,
        residual_smoothness_weight=0.0,
        residual_curvature_weight=1.0,
    )
    assert penalty > 0.0
