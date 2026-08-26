import unittest

import torch
from types import SimpleNamespace

from iac_new.drivewam_adapter import DriveWAMIntervention, trajectory_to_drivewam_chunk


class DriveWAMAdapterTest(unittest.TestCase):
    def test_intervention_variants_are_explicit(self):
        self.assertEqual(DriveWAMIntervention(0).condition_chunk, 0)
        self.assertEqual(DriveWAMIntervention(1).condition_chunk, 1)
        with self.assertRaises(ValueError):
            DriveWAMIntervention(2)

    def test_trajectory_is_encoded_as_incremental_action_chunk(self):
        evaluator = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float32,
            config=SimpleNamespace(
                action_chunk_steps=2,
                action_dim=30,
                used_action_channel_ids=[0, 1, 2],
                norm_stat={"q01": [0.0, 0.0, 0.0], "q99": [2.0, 2.0, 2.0]},
            ),
        )
        out = trajectory_to_drivewam_chunk(evaluator, torch.tensor([[1.0, 0.0, 0.0], [2.0, 1.0, 0.5]]))
        self.assertEqual(tuple(out.shape), (1, 30, 1, 2, 1))
        self.assertTrue(torch.allclose(out[0, 0, 0, :, 0], torch.tensor([0.0, 0.0]), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
