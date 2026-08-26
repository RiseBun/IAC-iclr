#!/usr/bin/env python3
"""Streaming image evaluator for long NAVSIM batches.

The regular CLI writes only after all records finish.  This runner flushes
every scored row so a preemption or SSH disconnect cannot erase a long run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iac_new.evaluate import evaluate_record, load_json
from iac_new.flow import RaftFlowExtractor, cuda_peak_memory_mb
from iac_new.perception import build_perception
from iac_new.protocol import read_jsonl, validate_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    raw_rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        raw_rows = raw_rows[: args.max_samples]
    records = [validate_record(row, manifest_root=args.manifest.parent) for row in raw_rows]
    flow_cfg = config["flow"]
    extractor = RaftFlowExtractor(
        model_size=str(flow_cfg["model"]), device=args.device,
        updates=int(flow_cfg["updates"]), batch_size=int(flow_cfg["batch_size"]),
        forward_backward=bool(flow_cfg["forward_backward"]),
        fb_abs_threshold_px=float(flow_cfg["fb_abs_threshold_px"]),
        fb_relative_threshold=float(flow_cfg["fb_relative_threshold"]),
    )
    perception = build_perception(config, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored = []
    invalid = []
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            try:
                result = evaluate_record(record, extractor, config, None, None, perception)
                scored.append(result)
                handle.write(json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n")
                handle.flush()
            except Exception as error:
                if args.fail_fast:
                    raise
                invalid.append({"sample_id": record["sample_id"], "error": str(error)})
            print(json.dumps({"completed": index, "total": len(records)}, separators=(",", ":")), flush=True)
    elapsed = time.perf_counter() - started
    valid_rows = [row for row in scored if row.get("valid")]
    internal_abstain = [row for row in scored if not row.get("valid")]
    gt_rows = [row for row in valid_rows if row.get("gt_candidate_id") is not None]
    ranks = []
    for row in gt_rows:
        ordered = sorted(row["candidate_scores"], key=lambda item: item["energy"])
        ranks.append(next(i + 1 for i, item in enumerate(ordered) if item["candidate_id"] == row["gt_candidate_id"]))
    internal_reason_counts: dict[str, int] = {}
    for row in internal_abstain:
        for reason in row.get("abstain_reasons", []):
            key = str(reason).split(":", 1)[-1]
            internal_reason_counts[key] = internal_reason_counts.get(key, 0) + 1
    exception_reason_counts: dict[str, int] = {}
    for item in invalid:
        message = str(item.get("error", ""))
        key = "low_effective_pixels" if "effective pixels" in message else "other_exception"
        exception_reason_counts[key] = exception_reason_counts.get(key, 0) + 1
    summary = {
        "protocol": "iac-new-image-v1-streaming",
        "config": str(args.config.resolve()), "manifest": str(args.manifest.resolve()),
        "num_input": len(records), "num_returned": len(scored),
        "num_valid": len(valid_rows), "num_internal_abstain": len(internal_abstain),
        "num_exception": len(invalid),
        "num_total_abstain": len(internal_abstain) + len(invalid),
        "abstain_fraction": (len(internal_abstain) + len(invalid)) / len(records) if records else None,
        "top1_accuracy": float(np.mean([rank == 1 for rank in ranks])) if ranks else None,
        "mean_native_rank": float(np.mean(ranks)) if ranks else None,
        "median_native_rank": float(np.median(ranks)) if ranks else None,
        "coverage": float(np.mean([row["gt_in_prediction_set"] for row in gt_rows])) if gt_rows else None,
        "mean_prediction_set_size": float(np.mean([row["prediction_set_size"] for row in gt_rows])) if gt_rows else None,
        "mean_observability_effective_static_fraction": float(np.mean([
            item["effective_static_pixel_fraction"] for row in scored for item in row["observability"]
        ])) if scored else None,
        "elapsed_s": elapsed,
        "samples_per_s": len(records) / elapsed if elapsed > 0 else None,
        "peak_cuda_memory_mb": cuda_peak_memory_mb(extractor.torch),
        "invalid_records": invalid,
        "internal_abstain_reason_counts": internal_reason_counts,
        "exception_reason_counts": exception_reason_counts,
        "rank_histogram": {str(rank): ranks.count(rank) for rank in sorted(set(ranks))},
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
