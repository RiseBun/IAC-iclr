#!/usr/bin/env python3
"""Render representative and failure-case CoTracker actor overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sample_error(sample: dict[str, Any]) -> float:
    values = sample.get("pixel_errors_px") or []
    return float(np.median(values)) if values else float("inf")


def _select(samples: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    output = {}
    for chain in sorted({sample["chain_type"] for sample in samples}):
        group = [sample for sample in samples if sample["chain_type"] == chain]
        finite = [sample for sample in group if np.isfinite(_sample_error(sample))]
        median = float(np.median([_sample_error(sample) for sample in finite]))
        output[chain] = {
            "representative": min(finite, key=lambda sample: abs(_sample_error(sample) - median)),
            "worst": max(finite, key=_sample_error),
        }
    return output


def _point(values: list[float | None]) -> tuple[int, int] | None:
    if len(values) != 2 or any(value is None for value in values):
        return None
    return tuple(round(float(value)) for value in values)


def _render(
    sample: dict[str, Any],
    manifest_row: dict[str, Any],
    output: Path,
    *,
    target_size: tuple[int, int],
    label: str,
) -> None:
    tracks = sample["pixel_tracks"]
    predicted = tracks["predicted_distorted_uv"]
    reference = tracks["reference_distorted_uv"]
    reference_visibility = tracks["reference_visibility"]
    tracker_visibility = tracks["tracker_visibility"]
    scored = tracks["scored"]
    query_index = int(sample["query_frame_index"])
    width, height = target_size
    panels = []
    for index, frame_path in enumerate(manifest_row["future_frame_paths"]):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        ref = _point(reference[index]) if reference_visibility[index] else None
        pred = _point(predicted[index]) if tracker_visibility[index] else None
        if ref is not None:
            cv2.circle(frame, ref, 7, (70, 220, 70), 2, cv2.LINE_AA)
        if pred is not None:
            cv2.drawMarker(frame, pred, (220, 80, 220), cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
        if ref is not None and pred is not None and scored[index]:
            cv2.line(frame, ref, pred, (60, 210, 240), 1, cv2.LINE_AA)
        if index == query_index and ref is not None:
            cv2.circle(frame, ref, 11, (40, 220, 240), 2, cv2.LINE_AA)
        header = np.full((34, width, 3), 24, dtype=np.uint8)
        time_s = manifest_row["future_times_s"][index]
        status = "query" if index == query_index else "score" if scored[index] else "not scored"
        error = ""
        if ref is not None and pred is not None:
            error = f"  EPE={np.linalg.norm(np.asarray(ref) - np.asarray(pred)):.1f}px"
        cv2.putText(
            header, f"t={time_s:.1f}s  {status}{error}", (9, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.53, (235, 235, 235), 1, cv2.LINE_AA,
        )
        panels.append(np.vstack([header, frame]))
    sheet = np.vstack([np.hstack(panels[:4]), np.hstack(panels[4:])])
    title_height = 54
    title = np.full((title_height, sheet.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(
        title,
        f"{sample['chain_type']} | {label} | green=LiDAR projection, magenta=CoTracker",
        (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.vstack([title, sheet])):
        raise RuntimeError(f"failed to write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    args = parser.parse_args()
    manifest = {row["sample_id"]: row for row in _read_jsonl(args.manifest)}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    selected = _select(report["samples"])
    index = []
    for chain, variants in selected.items():
        for label, sample in variants.items():
            filename = f"{chain}_{label}.jpg"
            _render(
                sample, manifest[sample["sample_id"]], args.output_dir / filename,
                target_size=(args.width, args.height), label=label,
            )
            index.append({
                "chain_type": chain,
                "selection": label,
                "sample_id": sample["sample_id"],
                "median_pixel_epe_px": _sample_error(sample),
                "path": str((args.output_dir / filename).resolve()),
            })
    (args.output_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
