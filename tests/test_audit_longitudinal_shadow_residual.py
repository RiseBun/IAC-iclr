import json
from pathlib import Path

from scripts.audit_longitudinal_shadow_residual import audit


def test_audit_reports_artifact_overlap_and_paired_gain(tmp_path: Path) -> None:
    manifest_row = {
        "sample_id": "s1",
        "scene_id": "scene-a",
        "future_frame_paths": ["f0", "f1", "f2"],
        "future_times_s": [0.5, 1.0, 1.5],
        "history_times_s": [-1.0, -0.5, 0.0],
        "history_ego_state": [[-2, 0, 0, 4.0, 0], [-1, 0, 0, 4.5, 0], [0, 0, 0, 5.0, 0]],
        "gt_candidate_id": "logged",
        "candidates": [{"candidate_id": "logged", "trajectory": [[2.5, 0, 0], [5.0, 0, 0], [7.5, 0, 0]]}],
    }
    probe_row = {"sample_id": "s1", "predicted_progress_m": [2.0, 0.3393201223], "gt_progress_m": [2.5, 2.5]}
    manifest = tmp_path / "manifest.jsonl"
    probe = tmp_path / "probe.json"
    output = tmp_path / "audit.json"
    manifest.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
    probe.write_text(json.dumps({"num_evaluable": 1, "results": [probe_row]}), encoding="utf-8")
    report = audit(manifest, probe, seed=1)
    assert report["artifact_0p3393201223"]["occurrences"] == 1
    assert report["overlap"]["unique_frame_pairs"] == 2
    assert report["all_window_summary"]["strong_null_mae"]["n"] == 2
