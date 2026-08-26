import unittest

from scripts.build_native_wam_branch_manifest import build_rows


class NativeWamBranchManifestTest(unittest.TestCase):
    def _row(self):
        return {
            "source_key": "navsim:scene:anchor",
            "history_images": ["h0", "h1", "h2", "h3"],
            "history_ego_state": [[0, 0, 0, 1, 0]] * 4,
            "realized_future_ego_state": [[1, 0, 0, 1, 0]] * 4,
            "trajectory": [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            "future_times_s": [0.5, 1.0, 1.5, 2.0],
            "task_success": True,
            "state_reference_source": "navsim_logged_ego_state",
        }

    def test_lineage_is_shared_but_realized_state_only_on_logged_branch(self):
        rows = build_rows([self._row()], lateral_offset_m=0.75, yaw_offset_rad=0.12, modes=("logged", "left", "right"))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["source_key"] for row in rows}, {"navsim:scene:anchor"})
        self.assertEqual({row["counterfactual_group_id"] for row in rows}, {"navsim:scene:anchor"})
        logged = next(row for row in rows if row["branch_mode"] == "logged")
        counterfactual = next(row for row in rows if row["branch_mode"] == "left")
        self.assertIsNotNone(logged["realized_future_ego_state"])
        self.assertIsNone(counterfactual["realized_future_ego_state"])
        self.assertEqual(counterfactual["future_images_source"], "wam_pending")

    def test_rejects_missing_lineage_state(self):
        row = self._row()
        del row["source_key"]
        with self.assertRaises(ValueError):
            build_rows([row], lateral_offset_m=0.75, yaw_offset_rad=0.12, modes=("logged",))

    def test_records_model_specific_window_contract(self):
        row = self._row()
        row["history_images"] = [f"h{i}" for i in range(15)]
        row["history_ego_state"] = [[0, 0, 0, 1, 0]] * 15
        rows = build_rows(
            [row],
            lateral_offset_m=0.75,
            yaw_offset_rad=0.12,
            modes=("logged",),
            model_id="drivingworld",
            expected_history_count=15,
            expected_future_count=4,
        )
        self.assertEqual(rows[0]["wam_model_id"], "drivingworld")
        self.assertEqual(rows[0]["window_spec"]["history_frames"], 15)
        self.assertEqual(rows[0]["window_spec"]["future_frames"], 4)


if __name__ == "__main__":
    unittest.main()
