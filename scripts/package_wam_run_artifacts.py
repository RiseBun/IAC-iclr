#!/usr/bin/env python3
"""Package auditable per-sample WAM/Level-1/CCFC artifacts without GT images."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--ccfc", type=Path, action="append", default=[])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = {str(r["sample_id"]): r for r in (json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip())}
    alignment = load_json(args.alignment)
    ccfc_rows: dict[str, list[dict[str, Any]]] = {}
    ccfc_reports: list[dict[str, Any]] = []
    for path in args.ccfc:
        payload = load_json(path)
        for report in payload.get("reports", []):
            gid = str(report.get("counterfactual_group_id"))
            enriched = {"source_file": str(path), **report}
            ccfc_reports.append(enriched)
            ccfc_rows.setdefault(gid, []).append(enriched)
    records = []
    for rec in alignment.get("records", []):
        sid = str(rec["sample_id"])
        m = manifest.get(sid, {})
        records.append({
            "sample_id": sid,
            "scene_id": rec.get("scene_id"),
            "image_timestamps_s": {"history": m.get("history_times_s"), "future": m.get("future_times_s")},
            "action_timestamps_s": rec.get("future_times_s"),
            "image_motion_profile": rec.get("image_motion_profile"),
            "raw_image_motion_profile": rec.get("raw_image_motion_profile"),
            "interval_observability": [x.get("observability") for x in rec.get("comparison", {}).get("per_interval", [])],
            "interval_status": [x.get("status") for x in rec.get("comparison", {}).get("per_interval", [])],
            "comparison": rec.get("comparison"),
            "distance_alignment": rec.get("distance_alignment_relative_observable"),
            "pose_alignment": rec.get("pose_alignment_arc_relative"),
            "ccfc_groups": ccfc_rows.get(sid, []),
        })
    out = {
        "protocol": "wam-run-artifact-bundle-v1",
        "alignment_source": str(args.alignment),
        "manifest_source": str(args.manifest),
        "ccfc_sources": [str(x) for x in args.ccfc],
        "timestamp_note": "Level-1 image/action timestamps are manifest future_times_s; CCFC reports retain native report fields and are not joined to Level-1 sample IDs unless IDs match.",
        "records": records,
        "ccfc_reports": ccfc_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output)}))
if __name__ == "__main__": main()
