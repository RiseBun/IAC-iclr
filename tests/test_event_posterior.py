import unittest

import numpy as np

from iac_new.event_posterior import build_event_posterior
from iac_new.maneuver import extract_maneuver


class EventPosteriorTest(unittest.TestCase):
    def test_turn_keeps_set_valued_support(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5])
        trajectory = np.asarray([
            [1.0, 0.0, 0.0],
            [2.0, 0.1, 0.04],
            [3.0, 0.3, 0.08],
        ])
        maneuver = extract_maneuver(trajectory, times)
        posterior = maneuver["event_posterior"]
        self.assertEqual(len(posterior), 3)
        self.assertEqual(posterior[1]["lateral_event"], "turn_left")
        self.assertIn("turn_left", posterior[1]["allowed_events"])
        self.assertTrue(abs(sum(posterior[1]["lateral_posterior"].values()) - 1.0) < 1e-9)

    def test_missing_topology_is_unknown(self) -> None:
        times = np.asarray([0.5])
        maneuver = extract_maneuver(np.asarray([[1.0, 0.0, 0.0]]), times)
        self.assertEqual(maneuver["event_posterior"][0]["road_event"], "unknown")

    def test_low_observability_marks_abstain(self) -> None:
        times = np.asarray([0.5])
        maneuver = extract_maneuver(np.asarray([[1.0, 0.0, 0.0]]), times)
        posterior = build_event_posterior(maneuver, times, observability=[0.1])
        self.assertTrue(posterior[0]["abstain"])


if __name__ == "__main__":
    unittest.main()
