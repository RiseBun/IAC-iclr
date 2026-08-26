import unittest

from scripts.build_wam_iac_diagnosis import diagnose


class WamIacDiagnosisTest(unittest.TestCase):
    def test_weak_wam_is_not_blamed_on_iac(self) -> None:
        report = diagnose(
            {"mean_heading_cosine": 0.995, "mean_lateral_abs_m": 0.20},
            {"action_image_distance_correlation": 0.05, "mean_response_ratio": 0.08},
        )
        self.assertEqual(report["diagnosis"], "wam_future_response_weak")

    def test_weak_iac_is_not_blamed_on_wam(self) -> None:
        report = diagnose(
            {"mean_heading_cosine": 0.90, "mean_lateral_abs_m": 0.80},
            {"action_image_distance_correlation": 0.60, "mean_response_ratio": 0.70},
        )
        self.assertEqual(report["diagnosis"], "iac_recovery_failure")


if __name__ == "__main__":
    unittest.main()
