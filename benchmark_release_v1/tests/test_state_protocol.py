import unittest

import numpy as np

from iac_new.state_protocol import navsim_states, nuplan_states, waymo_states


class StateProtocolTest(unittest.TestCase):
    def test_nuplan_states_are_anchor_relative(self) -> None:
        rows = [
            {"timestamp": 0, "x": 0.0, "y": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "vx": 2.0, "vy": 0.0, "angular_rate_z": 0.0},
            {"timestamp": 500000, "x": 1.0, "y": 0.1, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "vx": 2.0, "vy": 0.0, "angular_rate_z": 0.0},
        ]
        states = nuplan_states(rows, anchor_index=1)
        self.assertTrue(np.allclose(states[-1, :3], 0.0))
        self.assertAlmostEqual(states[-1, 3], 2.0)

    def test_navsim_and_waymo_adapters(self) -> None:
        records = [
            {"timestamp": 0.0, "ego2global": np.eye(4).tolist(), "ego_dynamic_state": [1.0, 0.0, 0.0, 0.0]},
            {"timestamp": 0.5, "ego2global": [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "ego_dynamic_state": [1.0, 0.0, 0.0, 0.0]},
        ]
        self.assertEqual(navsim_states(records, anchor_index=0).shape, (2, 5))
        waymo = [{"timestamp": 0.0, "x": 0, "y": 0, "heading": 0, "vx": 1, "vy": 0}, {"timestamp": 1.0, "x": 1, "y": 0, "heading": 0, "vx": 1, "vy": 0}]
        self.assertEqual(waymo_states(waymo, anchor_index=0).shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
