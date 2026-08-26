import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.mine_nuplan_causal_candidates import mine_candidates


class MineNuplanCausalCandidatesTest(unittest.TestCase):
    def test_mines_candidate_without_promoting_tag_to_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "sample.db"
            sensor_root = root / "sensor_blobs"
            sensor_root.mkdir()
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE scenario_tag (lidar_pc_token BLOB, type TEXT);
                CREATE TABLE lidar_pc (
                    token BLOB, timestamp INTEGER, scene_token BLOB, lidar_token BLOB
                );
                CREATE TABLE lidar (token BLOB, log_token BLOB);
                CREATE TABLE log (token BLOB, logfile TEXT, location TEXT);
                CREATE TABLE camera (token BLOB, channel TEXT);
                CREATE TABLE image (
                    camera_token BLOB, timestamp INTEGER, filename_jpg TEXT
                );
                """
            )
            connection.execute(
                "INSERT INTO scenario_tag VALUES (?, ?)",
                (b"pc", "waiting_for_pedestrian_to_cross"),
            )
            connection.execute(
                "INSERT INTO lidar_pc VALUES (?, ?, ?, ?)",
                (b"pc", 10_000_000, b"scene", b"lidar"),
            )
            connection.execute("INSERT INTO lidar VALUES (?, ?)", (b"lidar", b"log"))
            connection.execute(
                "INSERT INTO log VALUES (?, ?, ?)",
                (b"log", "sample-log", "test-city"),
            )
            connection.execute("INSERT INTO camera VALUES (?, ?)", (b"cam", "CAM_F0"))
            for offset in (-2, -1, 0, 1, 2, 3):
                relative = Path("sample-log") / "CAM_F0" / f"{offset}.jpg"
                path = sensor_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"jpg")
                connection.execute(
                    "INSERT INTO image VALUES (?, ?, ?)",
                    (b"cam", 10_000_000 + offset * 1_000_000, str(relative)),
                )
            connection.commit()
            connection.close()

            records, summary = mine_candidates(
                [db_path],
                sensor_root=sensor_root,
                max_per_chain=2,
                future_offsets_s=(1.0, 2.0, 3.0),
            )
            self.assertEqual(summary["num_records"], 1)
            self.assertTrue(summary["candidate_only"])
            self.assertEqual(records[0]["chain_type"], "pedestrian_crossing")
            self.assertEqual(
                records[0]["trigger_label_status"],
                "candidate_only_requires_blind_confirmation",
            )
            self.assertEqual(records[0]["counterfactual_pair_status"], "not_constructed")
            self.assertEqual(len(records[0]["history_images"]), 3)
            self.assertEqual(len(records[0]["future_images"]), 3)


if __name__ == "__main__":
    unittest.main()
