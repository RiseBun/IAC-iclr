import json
from pathlib import Path

from scripts.prepare_navsim_level1_v5_manifest import _select_nonoverlap


def _row(scene: str, start: int, sample: str) -> dict:
    return {
        "scene_id": scene,
        "sample_id": sample,
        "history_frame_paths": ["h"] * 4,
        "future_frame_paths": ["f"] * 8,
        "history_times_s": [-1.5, -1.0, -0.5, 0.0],
        "future_times_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "candidates": [{"candidate_id": "logged"}],
        "metadata": {"native_window_frame_indices": list(range(start, start + 12))},
    }


def test_select_nonoverlap_is_scene_aware():
    rows = [_row("a", 0, "a0"), _row("a", 1, "a1"), _row("a", 12, "a12"), _row("b", 0, "b0")]
    assert _select_nonoverlap(rows) == {"a0", "a12", "b0"}
