import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from iac_new.wam_adapters import WAMCapability, inspect_known_wams


class WamAdapterTest(unittest.TestCase):
    def test_only_external_control_is_image_cc_ready(self):
        base = WAMCapability(
            "x", "x", "/tmp/x", True, False, True, "x", "x", "x", "x", "x", "x"
        )
        self.assertFalse(base.suitable_for_counterfactual_image_cc)
        controlled = WAMCapability(
            "x", "x", "/tmp/x", True, True, True, "x", "x", "x", "x", "x", "x"
        )
        self.assertTrue(controlled.suitable_for_counterfactual_image_cc)

    def test_drivewam_is_primary_but_waits_for_base(self):
        with TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "DriveWAM").mkdir()
            rows = inspect_known_wams(home)
            drivewam = next(row for row in rows if row.model_id == "drivewam_navsim")
            self.assertEqual(drivewam.status, "checkpoint_available_base_pending")
            self.assertTrue(drivewam.external_trajectory_control)
            self.assertIn("LingBot-VA Base", drivewam.reason)

    def test_local_repositories_are_reported(self):
        rows = inspect_known_wams(Path("/tmp"))
        self.assertIsInstance(rows, list)


if __name__ == "__main__":
    unittest.main()
