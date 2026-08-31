import unittest

import numpy as np

from scripts.wam_onboarding_preflight import audit_rows


def row(branch: str, times=None):
    return {
        "source_key": "scene-1",
        "counterfactual_group_id": "scene-1",
        "branch_id": f"scene-1::{branch}",
        "history_images": ["h0.png", "h1.png"],
        "future_images": ["f0.png", "f1.png"],
        "future_times_s": times or [0.2, 0.4],
        "action_trajectory": [[1.0, 0.0, 0.0], [2.0, 0.1, 0.01]],
        "action_injection_verified": True,
    }


class WamOnboardingPreflightTest(unittest.TestCase):
    def test_native_grid_with_resampling_is_formal_level2_but_not_fcs(self):
        report = audit_rows(
            [row("logged"), row("left")],
            {
                "native_action_head": True,
                "external_trajectory_control": True,
                "time_alignment": "continuous_resample",
                "independent_rollout": False,
            },
            np.asarray([0.5, 1.0]),
        )
        self.assertTrue(report["image_probe_ready"])
        self.assertTrue(report["counterfactual_image_ready"])
        self.assertTrue(report["formal_level2_ready"])
        self.assertFalse(report["fcs_ready"])
        self.assertFalse(report["canonical_time_grid_exact"])

    def test_missing_control_fails_closed(self):
        report = audit_rows(
            [row("logged"), row("left")],
            {
                "native_action_head": True,
                "external_trajectory_control": False,
                "time_alignment": "exact",
                "independent_rollout": True,
            },
            np.asarray([0.2, 0.4]),
        )
        self.assertTrue(report["image_probe_ready"])
        self.assertFalse(report["counterfactual_image_ready"])
        self.assertFalse(report["formal_level2_ready"])


if __name__ == "__main__":
    unittest.main()
