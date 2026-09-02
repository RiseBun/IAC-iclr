#!/usr/bin/env python3
"""Download missing Waymo Perception v2 camera-image shards.

Calibration and vehicle-pose shards are used as the authoritative inventory;
only FRONT-camera image shards are downloaded.  The script is resumable and
never overwrites a complete parquet file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BUCKET = "gs://waymo_open_dataset_v_2_0_1/validation/camera_image"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="raw/perception_v2/validation root")
    parser.add_argument("--limit", type=int, default=0, help="optional maximum number of shards")
    parser.add_argument("--workers", type=int, default=4, help="parallel gcloud transfers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    calibration = args.root / "camera_calibration"
    camera_image = args.root / "camera_image"
    camera_image.mkdir(parents=True, exist_ok=True)
    segments = sorted(p.stem for p in calibration.glob("*.parquet"))
    missing = [s for s in segments if not (camera_image / f"{s}.parquet").is_file()]
    if args.limit:
        missing = missing[: args.limit]
    print(json.dumps({"calibration_segments": len(segments), "missing_before": len(missing), "bucket": BUCKET}, indent=2))
    status = []

    def download(segment: str) -> dict[str, object]:
        destination = camera_image / f"{segment}.parquet"
        command = ["gcloud", "storage", "cp", f"{BUCKET}/{segment}.parquet", str(destination)]
        item = {"segment_id": segment, "destination": str(destination), "command": " ".join(command)}
        if args.dry_run:
            item["status"] = "planned"
            return item
        try:
            subprocess.run(command, check=True)
            item["status"] = "ok"
        except subprocess.CalledProcessError as exc:
            item["status"] = "failed"
            item["returncode"] = exc.returncode
        return item

    if args.dry_run or args.workers <= 1:
        status = [download(segment) for segment in missing]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download, segment): segment for segment in missing}
            for future in as_completed(futures):
                status.append(future.result())
    status.sort(key=lambda item: str(item["segment_id"]))
    for i, item in enumerate(status, 1):
        print(json.dumps({"progress": f"{i}/{len(status)}", **item}))
    (camera_image / "download_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"attempted": len(status), "ok": sum(x["status"] == "ok" for x in status), "failed": sum(x["status"] == "failed" for x in status)}, indent=2))


if __name__ == "__main__":
    main()
