import json
from pathlib import Path

from scripts.evaluate_pure_speed_reciprocal import evaluate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_reciprocal_controls_are_twin_level_and_candidate_blind(tmp_path: Path) -> None:
    manifest = []
    scores = []
    times = [0.5, 1.0, 1.5, 2.0]
    cases = {
        "clean": ("fast", "slow"),
        "time_reversed": ("slow", "fast"),
        "wrong_identity": ("slow", "fast"),
    }
    row_id = 0
    for control, (fast_source, slow_source) in cases.items():
        for condition, source in (("fast", fast_source), ("slow", slow_source)):
            row_id += 1
            key = f"video-{row_id}"
            manifest.append({
                "video_id": key,
                "twin_id": f"twin-{control}",
                "condition": condition,
                "control_type": control,
                "future_times_s": times,
            })
            curve = [1.0, 2.0, 3.0, 4.0] if source == "fast" else [0.5, 1.0, 1.5, 2.0]
            scores.append({"video_id": key, "predicted_progress_curve": curve})

    manifest_path = tmp_path / "manifest.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(scores_path, scores)
    report = evaluate(manifest_path, scores_path, seed=17)

    assert report["clean"]["condition_accuracy"]["mean"] == 1.0
    assert report["controls"]["time_reversed"]["condition_accuracy"]["mean"] == 0.0
    assert report["controls"]["wrong_identity"]["condition_accuracy"]["mean"] == 0.0
    assert report["wrong_identity_rejection"] == 1.0
    assert report["time_reversal_drop"] == 1.0


def test_nested_motion_speed_is_supported(tmp_path: Path) -> None:
    manifest = [
        {"video_id": "fast", "twin_id": "t1", "condition": "fast", "future_times_s": [1.0]},
        {"video_id": "slow", "twin_id": "t1", "condition": "slow", "future_times_s": [1.0]},
    ]
    scores = [
        {"video_id": "fast", "predicted_motion": {"forward_rate_mps": 8.0}},
        {"video_id": "slow", "predicted_motion": {"forward_rate_mps": 4.0}},
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(scores_path, scores)
    report = evaluate(manifest_path, scores_path)
    assert report["clean"]["condition_accuracy"]["mean"] == 1.0
    assert report["results"][0]["margin_fast_minus_slow"] == 4.0
