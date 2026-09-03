#!/usr/bin/env python3
"""Build deterministic blur/compression/flicker controls from logged frames.

This creates a diagnostic B group.  It never changes timestamps, states or
labels, and writes new image files instead of modifying the logged source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _images(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None:
        value = row.get("history_frame_paths" if key == "history_images" else "future_frame_paths")
    return [str(item) for item in value or []]


def _factor(sample_id: str, index: int, amplitude: float) -> float:
    digest = hashlib.sha256(f"{sample_id}:{index}".encode()).digest()
    sign = 1.0 if digest[0] % 2 else -1.0
    return 1.0 + sign * amplitude


def _degrade(image: np.ndarray, *, sample_id: str, index: int, sigma: float, jpeg_quality: int, flicker: float) -> np.ndarray:
    result = image
    if sigma > 0.0:
        k = max(3, int(round(sigma * 6)) | 1)
        result = cv2.GaussianBlur(result, (k, k), sigmaX=sigma, sigmaY=sigma)
    if flicker > 0.0:
        result = np.clip(result.astype(np.float32) * _factor(sample_id, index, flicker), 0.0, 255.0).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--name", default="medium")
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--flicker", type=float, default=0.10)
    args = parser.parse_args()
    if args.sigma < 0 or not 1 <= args.jpeg_quality <= 100 or args.flicker < 0 or args.flicker >= 1:
        raise SystemExit("invalid degradation parameters")

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("source_key") or len(output_rows))
        safe_id = hashlib.sha256(sample_id.encode()).hexdigest()[:16]
        out = dict(row)
        out["sample_id"] = sample_id
        for key in ("history_images", "future_images"):
            paths = _images(row, key)
            converted: list[str] = []
            for index, source in enumerate(paths):
                image = cv2.imread(source, cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(source)
                target = args.output_root / args.name / safe_id / f"{key}_{index:02d}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), _degrade(image, sample_id=sample_id, index=index, sigma=args.sigma, jpeg_quality=args.jpeg_quality, flicker=args.flicker))
                converted.append(str(target))
            out[key] = converted
            # The frozen protocol normalizes from the explicit frame-path
            # fields when they are present. Keep both aliases in sync so the
            # decoder cannot silently fall back to the undegraded source.
            path_key = "history_frame_paths" if key == "history_images" else "future_frame_paths"
            out[path_key] = converted
        out["frame_paths"] = list(out.get("history_frame_paths", [])) + list(out.get("future_frame_paths", []))
        out["degradation_control"] = {
            "name": args.name,
            "sigma": args.sigma,
            "jpeg_quality": args.jpeg_quality,
            "flicker_amplitude": args.flicker,
            "source_manifest": str(args.manifest),
        }
        output_rows.append(out)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(output_rows), "output_manifest": str(args.output_manifest), "name": args.name}))


if __name__ == "__main__":
    main()
