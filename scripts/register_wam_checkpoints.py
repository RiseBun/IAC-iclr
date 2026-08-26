#!/usr/bin/env python3
"""Register available WAM checkpoints in one stable, non-destructive tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_ROOT = Path(os.environ.get("IAC_MODEL_ROOT", Path.cwd() / "model_checkpoints"))
WAM_ROOT = Path(os.environ.get("IAC_WAM_ROOT", Path.cwd() / "third_party"))
DEFAULT_ENTRIES = [
    ("worlddrive", "worldtraj_stage1_1024_tadwm.pkl", str(MODEL_ROOT / "worlddrive" / "worldtraj_stage1_1024_tadwm.pkl"), "action-conditioned NAVSIM world model"),
    ("epona", "epona_nuplan.pkl", str(MODEL_ROOT / "epona" / "epona_nuplan.pkl"), "action-conditioned NAVSIM world model"),
    ("drivingworld", "world_model.pth", str(MODEL_ROOT / "drivingworld" / "world_model.pth"), "action-conditioned NAVSIM world model"),
    ("vista", "vista.safetensors", str(MODEL_ROOT / "vista" / "vista.safetensors"), "video world model; action conditioning requires adapter"),
    ("cami2v", "256_cami2v.pt", str(MODEL_ROOT / "cami2v" / "256_cami2v.pt"), "camera-conditioned video model; not a native action WAM"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def register(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for model_id, filename, source, role in DEFAULT_ENTRIES:
        source_path = Path(source)
        target_dir = output_root / model_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        item: dict[str, Any] = {
            "model_id": model_id,
            "role": role,
            "source_path": str(source_path),
            "registry_path": str(target),
            "status": "missing",
        }
        if source_path.exists() and source_path.is_file() and source_path.stat().st_size > 0:
            if not target.exists():
                os.link(source_path, target)
            elif target.stat().st_size != source_path.stat().st_size:
                raise ValueError(f"registry target exists with a different size: {target}")
            item.update({
                "status": "available",
                "size_bytes": int(source_path.stat().st_size),
                "sha256": _sha256(source_path),
            })
        entries.append(item)
    drivewam_root = output_root / "drivewam"
    drivewam_root.mkdir(parents=True, exist_ok=True)
    entries.append({
        "model_id": "drivewam",
        "role": "native action-conditioned WAM; official checkpoint required",
        "source_path": str(WAM_ROOT / "DriveWAM"),
        "registry_path": str(drivewam_root),
        "status": "code_available_weights_missing",
        "checkpoint_source": "https://huggingface.co/chenchenshi/DriveWAM",
    })
    # DriveWAM publishes separate dataset-specific checkpoints.  The files are
    # downloaded directly into the registry, so registration must not relink a
    # file onto itself; it only records availability and integrity metadata.
    for variant, relative_path, role in (
        ("drivewam_navsim", "drivewam/navsim/diffusion_pytorch_model.safetensors", "native action-conditioned DriveWAM for NAVSIM"),
        ("drivewam_physicalai", "drivewam/physicalai/diffusion_pytorch_model.safetensors", "native action-conditioned DriveWAM for PhysicalAI"),
    ):
        checkpoint = output_root / relative_path
        item = {
            "model_id": variant,
            "role": role,
            "source_path": "https://huggingface.co/chenchenshi/DriveWAM",
            "registry_path": str(checkpoint),
            "status": "missing",
        }
        if checkpoint.exists() and checkpoint.is_file() and checkpoint.stat().st_size > 0:
            item.update({
                "status": "available",
                "size_bytes": int(checkpoint.stat().st_size),
                "sha256": _sha256(checkpoint),
            })
        entries.append(item)
    manifest = {
        "protocol": "wam-checkpoint-registry-v1",
        "registry_root": str(output_root),
        "entries": entries,
        "non_destructive": True,
        "note": "Registry entries are hard links to existing files; original source paths are retained.",
    }
    (output_root / "registry.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("IAC_MODEL_REGISTRY", Path.cwd() / "model_registry" / "wam")),
    )
    args = parser.parse_args()
    print(json.dumps(register(args.output_root), indent=2))


if __name__ == "__main__":
    main()
