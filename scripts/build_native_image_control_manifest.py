#!/usr/bin/env python3
"""Build a candidate-blind IAC control manifest from native future images."""
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    native = {row["source_key"]: row for row in read_jsonl(args.records)}
    output = []
    for row in read_jsonl(args.branches):
        source = native[row["source_key"]]
        merged = dict(row)
        merged["history_images"] = source["history_images"]
        merged["future_images"] = source["future_images"]
        merged["future_images_source"] = "navsim_native_realized"
        merged["future_times_s"] = source["future_times_s"]
        merged["action_trajectory"] = row["action_trajectory"]
        output.append(merged)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": len(output), "future_frames": len(output[0]["future_images"]) if output else 0}, indent=2))


if __name__ == "__main__":
    main()
