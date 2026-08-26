#!/usr/bin/env python3
"""Build a versioned, auditable metric card from benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from iac_new import __version__


def _load(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def _artifact(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def _first(report: dict[str, Any] | None, *keys: str) -> Any:
    if report is None:
        return None
    for key in keys:
        if report.get(key) is not None:
            return report[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-audit", type=Path, required=True)
    parser.add_argument("--image-probe-report", type=Path)
    parser.add_argument("--wam-report", type=Path)
    parser.add_argument("--split", choices=("calibration", "validation", "holdout", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = _load(args.manifest_audit) or {}
    image = _load(args.image_probe_report)
    wam = _load(args.wam_report)
    image_compatibility = _first(image, "mean_realized_state_compatibility", "mean_soft_compatibility")
    image_scored = _first(image, "num_scored", "rows")
    image_input = _first(image, "num_input", "rows")
    coverage = float(image_scored / image_input) if image_scored is not None and image_input else None
    action_cc = _first(wam, "mean_action_image_matrix_cc_margin", "mean_counterfactual_consistency")
    realized_cc = _first(wam, "mean_realized_state_counterfactual_consistency")
    fcs_payload = wam.get("foresight_conditioned_success") if wam else None
    fcs = fcs_payload.get("foresight_conditioned_success") if isinstance(fcs_payload, dict) else None
    gates = {
        "manifest_valid": audit.get("status") == "ok",
        "image_probe_ready": bool(audit.get("image_probe_ready")),
        "counterfactual_pairs_ready": bool(audit.get("action_response_ready")),
        "realized_state_ready": bool(audit.get("realized_state_ready")),
        "fcs_ready": bool(audit.get("fcs_ready")),
    }
    metrics = {
        "image_trajectory_compatibility": {
            "status": "available" if image_compatibility is not None else "unavailable",
            "value": image_compatibility,
            "coverage": coverage,
            "claim": "IAC image-to-trajectory capability; not causal WAM CC",
        },
        "action_image_counterfactual_consistency": {
            "status": "available" if action_cc is not None and gates["counterfactual_pairs_ready"] else "unavailable",
            "value": action_cc if gates["counterfactual_pairs_ready"] else None,
            "claim": "same-history action intervention response",
        },
        "realized_state_counterfactual_consistency": {
            "status": "available" if realized_cc is not None and gates["realized_state_ready"] else "unavailable",
            "value": realized_cc if gates["realized_state_ready"] else None,
            "claim": "generated-image response against independent realized future ego state",
        },
        "foresight_conditioned_success": {
            "status": "available" if fcs is not None and gates["fcs_ready"] else "unavailable",
            "value": fcs if gates["fcs_ready"] else None,
            "claim": "task success conditioned on frozen future-state compatibility threshold",
        },
    }
    result = {
        "protocol": "iac-wam-metric-card-v1",
        "iac_version": __version__,
        "split": args.split,
        "gates": gates,
        "metrics": metrics,
        "artifacts": {
            "manifest_audit": _artifact(args.manifest_audit),
            "image_probe_report": _artifact(args.image_probe_report),
            "wam_report": _artifact(args.wam_report),
        },
        "valid_claims": [name for name, payload in metrics.items() if payload["status"] == "available"],
        "invalid_claims": [name for name, payload in metrics.items() if payload["status"] != "available"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
