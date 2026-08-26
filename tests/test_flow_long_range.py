import unittest

import numpy as np

from iac_new.flow import long_range_flow_residual


class LongRangeFlowResidualTest(unittest.TestCase):
    def test_constant_rate_motion_has_zero_residual(self) -> None:
        direct = np.full((3, 4, 2), [2.0, -1.0], dtype=np.float32)
        two_step = np.full((3, 4, 2), [4.0, -2.0], dtype=np.float32)
        np.testing.assert_allclose(long_range_flow_residual(direct, two_step), 0.0)

    def test_reports_vector_discrepancy(self) -> None:
        direct = np.asarray([[[2.0, 0.0]]], dtype=np.float32)
        two_step = np.asarray([[[2.0, 0.0]]], dtype=np.float32)
        np.testing.assert_allclose(long_range_flow_residual(direct, two_step), 1.0)

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            long_range_flow_residual(
                np.zeros((2, 2, 2), dtype=np.float32),
                np.zeros((2, 3, 2), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
