import unittest

import numpy as np

from iac_new.continuous_motion import (
    compare_future_control,
    compare_history_baseline,
    compare_longitudinal_behavior,
    compare_counterfactual_motion_deltas,
    compare_counterfactual_se2_consistency,
    compare_motion_profiles,
    compare_distance_profiles,
    compare_pose_profiles,
    compare_pose_posteriors,
    apply_pose_interval_calibration,
    foresight_gain,
    history_anchored_residual_motion_profile,
    history_only_motion_profile,
    image_motion_profile,
    longitudinal_residual_features,
    reanchor_longitudinal_control_profile,
    trajectory_to_motion_profile,
)
from scripts.calibrate_longitudinal_residual import fit_longitudinal_gain, split_scene_groups
from scripts.calibrate_pose_posterior import fit_pose_interval_calibration
from iac_new.trajectory_decode import integrate_piecewise_controls
from scripts.evaluate_counterfactual_continuous_alignment import audit_pair
from scripts.evaluate_continuous_motion_alignment import _level1_input_audit


def decoder(speeds: list[float], *, curvature: float = 0.0) -> dict:
    times = np.asarray([0.5, 1.0, 1.5, 2.0])
    trajectory = integrate_piecewise_controls(
        times,
        speeds_mps=np.asarray(speeds),
        curvatures_1pm=np.full(4, curvature),
    )
    return {
        "protocol": "candidate-blind-continuous-trajectory-v1",
        "trajectory": trajectory.tolist(),
        "profile_support": [
            {
                "x_m": {"q05": float(point[0] - 0.2), "q50": float(point[0]), "q95": float(point[0] + 0.2)},
                "y_m": {"q05": float(point[1] - 0.1), "q50": float(point[1]), "q95": float(point[1] + 0.1)},
                "yaw_rad": {"q05": float(point[2] - 0.02), "q50": float(point[2]), "q95": float(point[2] + 0.02)},
            }
            for point in trajectory
        ],
        "speed_support": [
            {"q05": speed - 0.4, "q50": speed, "q95": speed + 0.4, "observability": 0.8, "status": "usable"}
            for speed in speeds
        ],
    }


class ContinuousMotionTest(unittest.TestCase):
    def test_waypoints_become_interval_motion(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        trajectory = integrate_piecewise_controls(
            times, speeds_mps=np.full(4, 4.0), curvatures_1pm=np.zeros(4)
        )
        profile = trajectory_to_motion_profile(trajectory, times, initial_speed_mps=4.0)
        np.testing.assert_allclose([row["speed_mps"] for row in profile["rows"]], 4.0)
        np.testing.assert_allclose([row["acceleration_mps2"] for row in profile["rows"]], 0.0)
        np.testing.assert_allclose([row["lateral_speed_mps"] for row in profile["rows"]], 0.0)

    def test_action_never_enters_image_profile(self) -> None:
        profile = image_motion_profile(decoder([4.0] * 4), [0.5, 1.0, 1.5, 2.0], initial_speed_mps=4.0)
        self.assertEqual(profile["source"], "image_only_candidate_blind_decoder")
        self.assertFalse(profile["candidate_bank_used"])
        self.assertNotIn("action_trajectory", profile)

    def test_continuous_comparison_scores_magnitude(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        imagined = image_motion_profile(decoder([4.0] * 4), times, initial_speed_mps=4.0)
        action_trajectory = integrate_piecewise_controls(
            np.asarray(times), speeds_mps=np.full(4, 5.0), curvatures_1pm=np.zeros(4)
        )
        action = trajectory_to_motion_profile(action_trajectory, times, initial_speed_mps=4.0)
        result = compare_motion_profiles(imagined, action)
        self.assertAlmostEqual(result["metrics"]["speed_mps"]["mae"], 1.0)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["speed_posterior"]["empirical_coverage"], 0.0)
        self.assertGreater(result["speed_posterior"]["mean_wis_90"], 0.0)
        self.assertFalse(result["leakage_audit"]["action_waypoint_visible_to_image_decoder"])

    def test_distance_alignment_has_metric_scale_free_and_relative_modes(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        imagined = image_motion_profile(decoder([4.0, 4.0, 4.0, 4.0]), times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(
            integrate_piecewise_controls(np.asarray(times), speeds_mps=np.full(4, 5.0), curvatures_1pm=np.zeros(4)),
            times,
            initial_speed_mps=4.0,
        )
        metric = compare_distance_profiles(imagined, action, scale_mode="metric")
        shape = compare_distance_profiles(imagined, action, scale_mode="scale_free")
        relative = compare_distance_profiles(imagined, action, scale_mode="relative")
        assert metric["metrics"]["forward_displacement_profile"]["mae"] > 0.0
        assert shape["metrics"]["forward_displacement_profile"]["mae"] == 0.0
        assert relative["metrics"]["forward_displacement_profile"]["mae"] == 0.0
        assert relative["metrics"]["forward_displacement_profile"]["endpoint_abs_error"] == 0.0
        assert shape["leakage_audit"]["action_used_for_image_scale"] is False

    def test_relative_distance_can_use_pose_when_speed_is_uncertain(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        value = decoder([4.0, 4.0, 4.0, 4.0])
        for row in value["speed_support"]:
            row["status"] = "uncertain"
        imagined = image_motion_profile(value, times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(value["trajectory"], times, initial_speed_mps=4.0)
        strict = compare_distance_profiles(imagined, action, scale_mode="relative")
        pose_only = compare_distance_profiles(
            imagined, action, scale_mode="relative", include_uncertain=True
        )
        assert strict["status"] == "abstain"
        assert pose_only["status"] == "ok"
        assert pose_only["metrics"]["forward_displacement_profile"]["mae"] == 0.0

    def test_pose_alignment_compares_forward_lateral_and_heading(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        imagined = image_motion_profile(decoder([4.0] * 4, curvature=0.02), times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(
            integrate_piecewise_controls(
                np.asarray(times), speeds_mps=np.full(4, 5.0), curvatures_1pm=np.full(4, 0.02)
            ),
            times,
            initial_speed_mps=4.0,
        )
        metric = compare_pose_profiles(imagined, action, scale_mode="metric")
        shape = compare_pose_profiles(imagined, action, scale_mode="scale_free")
        relative = compare_pose_profiles(imagined, action, scale_mode="relative")
        assert metric["metrics"]["se2_pose"]["forward_mae"] > 0.0
        assert shape["metrics"]["se2_pose"]["path_cosine"] > 0.99
        assert relative["metrics"]["se2_pose"]["path_cosine"] > 0.99
        assert shape["leakage_audit"]["action_used_for_image_scale"] is False

    def test_pose_posterior_reports_component_and_joint_coverage(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        value = decoder([4.0] * 4)
        imagined = image_motion_profile(value, times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(value["trajectory"], times, initial_speed_mps=4.0)
        result = compare_pose_posteriors(imagined, action)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metrics"]["x_m"]["empirical_coverage"], 1.0)
        self.assertEqual(result["metrics"]["y_m"]["empirical_coverage"], 1.0)
        self.assertEqual(result["metrics"]["heading_rad"]["empirical_coverage"], 1.0)
        self.assertEqual(result["joint_pose"]["empirical_coverage"], 1.0)
        self.assertFalse(result["leakage_audit"]["action_used_for_pose_interval_calibration"])

    def test_pose_posterior_excludes_abstained_intervals_from_coverage(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        value = decoder([4.0] * 4)
        value["speed_support"][0]["status"] = "abstain"
        imagined = image_motion_profile(value, times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(value["trajectory"], times, initial_speed_mps=4.0)
        result = compare_pose_posteriors(imagined, action)
        self.assertEqual(result["evaluable_intervals"], 3)
        self.assertEqual(result["coverage"], 0.75)

    def test_pose_interval_calibration_is_additive_and_action_blind_at_apply_time(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        value = decoder([4.0] * 4)
        imagined = image_motion_profile(value, times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(value["trajectory"], times, initial_speed_mps=4.0)
        for row in action["rows"]:
            row["progress_m"] += 1.0
        artifact = fit_pose_interval_calibration([
            {"decoder": value, "image_profile": imagined, "reference_profile": action}
        ])
        self.assertGreater(artifact["parameters"]["conformal_radius"]["x_m"], 0.0)
        self.assertFalse(artifact["action_waypoint_used_for_interval_fit"])
        self.assertTrue(artifact["independent_reference_used_for_interval_fit"])
        calibrated = apply_pose_interval_calibration(imagined, {
            **artifact,
            "protocol": "continuous-se2-pose-calibration-v1",
        })
        result = compare_pose_posteriors(calibrated, action)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metrics"]["x_m"]["empirical_coverage"], 1.0)
        self.assertFalse(calibrated["pose_interval_calibration"]["action_waypoint_used"])

    def test_history_null_uses_last_speed_and_yaw_rate(self) -> None:
        profile = history_only_motion_profile(
            [[-2.0, 0.0, 0.0, 4.0, 0.2], [0.0, 0.0, 0.0, 4.0, 0.2]],
            [0.5, 1.0],
        )
        np.testing.assert_allclose([row["speed_mps"] for row in profile["rows"]], 4.0)
        np.testing.assert_allclose([row["yaw_rate_radps"] for row in profile["rows"]], 0.2)
        self.assertEqual(profile["source"], "history_only_constant_speed_yaw_rate")

    def test_strong_history_null_uses_only_past_acceleration(self) -> None:
        profile = history_only_motion_profile(
            [[-3.0, 0.0, 0.0, 2.0, 0.0], [-1.5, 0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0, 0.0]],
            [0.5, 1.0],
            history_times_s=[-1.0, -0.5, 0.0],
            model="constant_acceleration_yaw_rate",
        )
        self.assertAlmostEqual(profile["history_anchor"]["acceleration_mps2"], 2.0)
        np.testing.assert_allclose([row["speed_mps"] for row in profile["rows"]], [4.5, 5.5])
        self.assertEqual(profile["source"], "history_only_constant_acceleration_yaw_rate")

    def test_foresight_gain_uses_identical_image_eligibility(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        imagined = image_motion_profile(decoder([4.0] * 4), times, initial_speed_mps=3.0)
        action = trajectory_to_motion_profile(decoder([4.0] * 4)["trajectory"], times, initial_speed_mps=3.0)
        history = history_only_motion_profile([[0.0, 0.0, 0.0, 3.0, 0.0]], times)
        image_result = compare_motion_profiles(imagined, action)
        history_result = compare_history_baseline(history, action, imagined)
        gain = foresight_gain(image_result, history_result)
        self.assertEqual(image_result["coverage"], history_result["coverage"])
        self.assertGreater(gain["metrics"]["speed_mps"]["absolute_gain"], 0.0)

    def test_longitudinal_residual_discards_multiplicative_image_scale(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        history = history_only_motion_profile([[0.0, 0.0, 0.0, 4.0, 0.0]], times)
        first = longitudinal_residual_features(decoder([5.0, 6.0, 7.0, 8.0]), times, history)
        shifted = longitudinal_residual_features(decoder([10.0, 12.0, 14.0, 16.0]), times, history)
        np.testing.assert_allclose(
            [row["image_speed_delta_mps"] for row in first["rows"]],
            [row["image_speed_delta_mps"] for row in shifted["rows"]],
        )
        self.assertFalse(first["absolute_image_speed_used"])

    def test_zero_longitudinal_gain_is_exact_history_null(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        history = history_only_motion_profile([[0.0, 0.0, 0.0, 4.0, 0.0]], times)
        profile = history_anchored_residual_motion_profile(
            decoder([5.0, 6.0, 7.0, 8.0]),
            times,
            history,
            longitudinal_gain=0.0,
            speed_interval_radius_mps=0.5,
        )
        np.testing.assert_allclose(
            [row["speed_mps"] for row in profile["rows"]],
            [row["speed_mps"] for row in history["rows"]],
        )
        self.assertFalse(profile["longitudinal_model"]["action_waypoint_visible_to_predictor"])

    def test_longitudinal_control_reanchors_to_recipient_history(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        donor_history = history_only_motion_profile([[0.0, 0.0, 0.0, 4.0, 0.0]], times)
        target_history = history_only_motion_profile([[0.0, 0.0, 0.0, 10.0, 0.0]], times)
        donor = history_anchored_residual_motion_profile(
            decoder([5.0, 5.5, 6.0, 6.5]),
            times,
            donor_history,
            longitudinal_gain=1.0,
            speed_interval_radius_mps=0.5,
        )
        transferred = reanchor_longitudinal_control_profile(donor, target_history, times)
        donor_delta = [
            row["longitudinal_residual_feature"]["image_speed_delta_mps"]
            for row in donor["rows"]
        ]
        np.testing.assert_allclose(
            [row["speed_mps"] for row in transferred["rows"]],
            np.asarray(donor_delta) + 10.0,
        )
        self.assertTrue(transferred["longitudinal_model"]["control_reanchored"])

    def test_scene_split_is_deterministic_and_disjoint(self) -> None:
        rows = [
            {"sample_id": f"sample-{index}", "scene_id": f"scene-{index // 2}"}
            for index in range(12)
        ]
        first = split_scene_groups(rows, seed=17, fit_fraction=0.4, calibration_fraction=0.3)
        second = split_scene_groups(rows, seed=17, fit_fraction=0.4, calibration_fraction=0.3)
        self.assertEqual(first, second)
        self.assertFalse(set(first["fit"]) & set(first["calibration"]))
        self.assertFalse(set(first["fit"]) & set(first["evaluation"]))
        self.assertFalse(set(first["calibration"]) & set(first["evaluation"]))

    def test_longitudinal_gain_fit_recovers_synthetic_residual_scale(self) -> None:
        record = {
            "decoder": {
                "speed_support": [
                    {"status": "usable", "observability": 1.0},
                    {"status": "usable", "observability": 1.0},
                ]
            },
            "features": {"rows": [{"innovation_mps": 1.0}, {"innovation_mps": 2.0}]},
            "history_profile": {"rows": [{"speed_mps": 4.0}, {"speed_mps": 4.0}]},
            "reference_profile": {"rows": [{"speed_mps": 4.5}, {"speed_mps": 5.0}]},
        }
        fit = fit_longitudinal_gain([record])
        self.assertAlmostEqual(fit["longitudinal_gain"], 0.5)
        self.assertAlmostEqual(fit["weighted_residual_mae_mps"], 0.0)

    def test_longitudinal_behavior_scores_change_direction(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        predicted = image_motion_profile(decoder([3.9, 3.5, 3.0, 2.6]), times, initial_speed_mps=4.0)
        action = trajectory_to_motion_profile(
            decoder([3.8, 3.4, 3.1, 2.5])["trajectory"], times, initial_speed_mps=4.0
        )
        result = compare_longitudinal_behavior(predicted, action, predicted)
        self.assertGreaterEqual(result["change_direction_accuracy"], 0.75)
        self.assertGreater(result["significant_change_direction_accuracy"], 0.5)

    def test_future_control_uses_target_observability_mask(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        target = decoder([4.0] * 4)
        target["speed_support"][0]["status"] = "abstain"
        target_profile = image_motion_profile(target, times)
        control_profile = image_motion_profile(decoder([2.0] * 4), times)
        action = trajectory_to_motion_profile(decoder([4.0] * 4)["trajectory"], times)
        result = compare_future_control(control_profile, action, target_profile)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(result["eligibility_mask_source"], "target_image_probe_observability")

    def test_abstained_intervals_do_not_count_as_failures(self) -> None:
        value = decoder([4.0] * 4)
        value["speed_support"][0]["status"] = "abstain"
        imagined = image_motion_profile(value, [0.5, 1.0, 1.5, 2.0])
        action = trajectory_to_motion_profile(value["trajectory"], [0.5, 1.0, 1.5, 2.0])
        result = compare_motion_profiles(imagined, action)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(result["metrics"]["speed_mps"]["count"], 3)

    def test_counterfactual_delta_checks_response_not_labels(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        clear_image = image_motion_profile(decoder([5.0] * 4), times, initial_speed_mps=5.0)
        risk_image = image_motion_profile(decoder([4.5, 4.0, 3.5, 3.0]), times, initial_speed_mps=5.0)
        clear_action = trajectory_to_motion_profile(decoder([5.0] * 4)["trajectory"], times, initial_speed_mps=5.0)
        risk_action = trajectory_to_motion_profile(decoder([4.5, 4.0, 3.5, 3.0])["trajectory"], times, initial_speed_mps=5.0)
        result = compare_counterfactual_motion_deltas(clear_image, risk_image, clear_action, risk_action)
        self.assertEqual(result["metrics"]["speed_mps"]["sign_agreement"], 1.0)
        self.assertAlmostEqual(result["metrics"]["speed_mps"]["delta_mae"], 0.0)

    def test_continuous_cfc_is_one_when_future_and_action_responses_match(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        clear_decoder = decoder([5.0] * 4, curvature=0.01)
        risk_decoder = decoder([4.5, 4.0, 3.5, 3.0], curvature=0.03)
        clear_image = image_motion_profile(clear_decoder, times, initial_speed_mps=5.0)
        risk_image = image_motion_profile(risk_decoder, times, initial_speed_mps=5.0)
        clear_action = trajectory_to_motion_profile(clear_decoder["trajectory"], times, initial_speed_mps=5.0)
        risk_action = trajectory_to_motion_profile(risk_decoder["trajectory"], times, initial_speed_mps=5.0)
        result = compare_counterfactual_se2_consistency(clear_image, risk_image, clear_action, risk_action)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["score"], 1.0, places=6)
        self.assertAlmostEqual(result["subscores"]["response_direction"], 1.0, places=6)

    def test_continuous_cfc_penalizes_opposite_counterfactual_response(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        clear_image = image_motion_profile(decoder([5.0] * 4), times, initial_speed_mps=5.0)
        risk_image = image_motion_profile(decoder([5.5] * 4), times, initial_speed_mps=5.0)
        clear_action = trajectory_to_motion_profile(decoder([5.0] * 4)["trajectory"], times, initial_speed_mps=5.0)
        risk_action = trajectory_to_motion_profile(decoder([4.5] * 4)["trajectory"], times, initial_speed_mps=5.0)
        result = compare_counterfactual_se2_consistency(clear_image, risk_image, clear_action, risk_action)
        self.assertEqual(result["status"], "ok")
        self.assertLess(result["subscores"]["response_direction"], 0.5)
        self.assertLess(result["score"], 0.5)

    def test_continuous_cfc_abstains_without_material_action_response(self) -> None:
        times = [0.5, 1.0, 1.5, 2.0]
        clear = decoder([5.0] * 4)
        image = image_motion_profile(clear, times, initial_speed_mps=5.0)
        action = trajectory_to_motion_profile(clear["trajectory"], times, initial_speed_mps=5.0)
        result = compare_counterfactual_se2_consistency(image, image, action, action)
        self.assertEqual(result["status"], "abstain")
        self.assertIsNone(result["score"])

    def test_counterfactual_readiness_is_fail_closed(self) -> None:
        base = {
            "history_fingerprint": "history-1",
            "wam_model_id": "wam-1",
            "nuisance_seed": 7,
            "future_images_source": "wam_generated",
            "action_trajectory_source": "native_action_head",
            "candidate_bank_used_by_decoder": False,
            "future_times_s": [0.5, 1.0],
        }
        clear = dict(base, action_trajectory=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        risk = dict(base, action_trajectory=[[0.8, 0.0, 0.0], [1.5, 0.0, 0.0]])
        self.assertEqual(audit_pair("group-1", {"clear": clear, "risk": risk}), [])
        del risk["nuisance_seed"]
        self.assertIn("missing_nuisance_seed", audit_pair("group-1", {"clear": clear, "risk": risk}))

    def test_level1_input_audit_requires_native_wam_outputs(self) -> None:
        row = {
            "future_images_source": "wam_generated",
            "action_trajectory_source": "native_action_head",
            "wam_model_id": "wam-1",
        }
        self.assertTrue(_level1_input_audit(row, "action")["ready"])
        self.assertFalse(_level1_input_audit(row, "logged_gt")["ready"])
        row["action_trajectory_source"] = "logged_candidate_proxy"
        self.assertFalse(_level1_input_audit(row, "action")["ready"])


if __name__ == "__main__":
    unittest.main()
