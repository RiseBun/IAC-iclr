import unittest

import numpy as np

from iac_new.sea_raft_flow import SeaRaftFlowExtractor


class SeaRaftFlowContractTest(unittest.TestCase):
    def test_observe_supports_main_evaluator_contract(self) -> None:
        extractor = SeaRaftFlowExtractor.__new__(SeaRaftFlowExtractor)
        extractor.use_forward_backward = True
        extractor.fb_abs_threshold_px = 1.5
        extractor.fb_relative_threshold = 0.05

        images = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(3)]
        intrinsics = np.eye(3, dtype=np.float64)
        extractor._read_images = lambda *args, **kwargs: (images, intrinsics, (12, 8))

        calls = []

        def infer(first, second, *, return_uncertainty=False, uncertainty_tail=8):
            calls.append((len(first), return_uncertainty, uncertainty_tail))
            flow = np.zeros((len(first), 4, 6, 2), dtype=np.float32)
            uncertainty = (
                np.full((len(first), 4, 6), 0.25, dtype=np.float32)
                if return_uncertainty
                else None
            )
            return flow, uncertainty

        extractor._infer_pairs = infer
        observation = extractor.observe(
            ["a.jpg", "b.jpg", "c.jpg"],
            intrinsics,
            np.asarray([], dtype=np.float64),
            (6, 4),
            return_uncertainty=True,
            uncertainty_tail=4,
            long_range_consistency=True,
        )

        self.assertEqual(observation.forward.shape, (2, 4, 6, 2))
        self.assertEqual(observation.consistency_masks.shape, (2, 4, 6))
        self.assertEqual(observation.refinement_uncertainty.shape, (2, 4, 6))
        self.assertEqual(observation.long_range_residual.shape, (2, 4, 6))
        self.assertTrue(np.isnan(observation.long_range_residual[-1]).all())
        self.assertEqual(calls, [(2, True, 4), (1, False, 8), (2, False, 8)])


if __name__ == "__main__":
    unittest.main()
