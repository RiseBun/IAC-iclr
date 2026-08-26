import unittest

import numpy as np

from iac_new.road_relative import compare_action_to_support, road_relative_posterior
from iac_new.road_structure import (
    build_road_structure,
    extract_road_boundaries,
    estimate_focus_of_expansion,
)


class RoadRelativeTest(unittest.TestCase):
    def test_boundaries_and_structure(self) -> None:
        mask = np.zeros((40, 60), dtype=bool)
        for y in range(40):
            left = 20 - y // 8
            right = 40 + y // 8
            mask[y, left:right + 1] = True
        boundary = extract_road_boundaries(mask, row_step=2)
        self.assertTrue(boundary["valid"])
        structure = build_road_structure(
            np.stack([mask, mask]),
            np.zeros((1, 40, 60, 2), dtype=np.float64),
            np.ones((1, 40, 60), dtype=np.float64),
            mask,
        )
        self.assertIn("near_road", structure)
        self.assertIn("far_road", structure)

    def test_foe_on_radial_flow(self) -> None:
        height, width = 30, 40
        yy, xx = np.indices((height, width), dtype=np.float64)
        foe = np.asarray([20.0, 8.0])
        flow = np.stack([xx - foe[0], yy - foe[1]], axis=-1)[None, ...]
        result = estimate_focus_of_expansion(flow, np.ones((1, height, width)), np.ones((height, width), dtype=bool))
        self.assertTrue(result["valid"])
        np.testing.assert_allclose(result["foe_xy"], foe, atol=1e-5)

    def test_support_comparison_is_tolerant(self) -> None:
        times = np.asarray([1.0, 2.0])
        trajectory = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.05, 0.02]])
        posterior = road_relative_posterior(trajectory, times)
        score = compare_action_to_support(trajectory, posterior, times)
        self.assertGreater(score["joint_support_coverage"], 0.9)
