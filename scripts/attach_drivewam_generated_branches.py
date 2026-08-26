#!/usr/bin/env python3
"""Attach DriveWAM generated-frame manifests to exact NAVSIM branches."""
import argparse
import json
import pickle
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_rows(project_root, manifest_path):
    rows = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    result = {}
    for item in rows:
        sample_path = Path(item["source_sample"])
        if not sample_path.is_absolute():
            sample_path = project_root / sample_path
        with open(sample_path, "rb") as f:
            sample = pickle.load(f)
        metadata = sample.get("metadata", {})
        key = metadata.get("source_key")
        if not key:
            raise ValueError(f"{sample_path}: metadata.source_key is required")
        paths = []
        for path in item.get("future_images", []):
            p = Path(path)
            paths.append(str(p if p.is_absolute() else project_root / p))
        if not paths:
            raise ValueError(f"{manifest_path}: sample {item.get('sample_index')} has no future images")
        result[(key, metadata.get("branch_mode"))] = {**item, "future_images": paths, "metadata": metadata}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--branches", required=True)
    ap.add_argument("--logged", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    generated = {}
    for path in (args.logged, args.left, args.right):
        generated.update(manifest_rows(root, path))
    output = []
    missing = []
    for row in read_jsonl(args.branches):
        key = (row.get("source_key"), row.get("branch_mode"))
        item = generated.get(key)
        if item is None:
            missing.append(row.get("branch_id"))
            continue
        merged = dict(row)
        merged["future_images"] = item["future_images"]
        # DriveWAM's NavSim config consumes future_image_indices=[1,3,5,7],
        # therefore its four decoded frames correspond to native 1/2/3/4 s
        # targets (native future points 2/4/6/8). Keep the original clock for
        # realized-state scoring while exposing the image clock to IAC.
        native_times = list(row.get("future_times_s") or [])
        if len(item["future_images"]) == 4 and len(native_times) == 8:
            merged["native_future_times_s"] = native_times
            merged["future_times_s"] = [native_times[i] for i in (1, 3, 5, 7)]
            native_action = list(row.get("action_trajectory") or [])
            if len(native_action) == 8:
                merged["native_action_trajectory"] = native_action
                merged["action_trajectory"] = [native_action[i] for i in (1, 3, 5, 7)]
        merged["future_images_source"] = "drivewam_generated"
        merged["wam_generation_status"] = "generated"
        merged["action_injection_verified"] = bool(item.get("action_injection_verified", False))
        merged["intervention_variant"] = item.get("intervention_variant")
        merged["predicted_action_trajectory"] = item.get("predicted_action_trajectory")
        output.append(merged)
    if missing:
        raise ValueError(f"missing generated manifest rows: {len(missing)}; first={missing[:3]}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": len(output), "missing": len(missing), "generated_source": "DriveWAM"}, indent=2))


if __name__ == "__main__":
    main()
