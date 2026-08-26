"""Blind annotation-pack construction for four-chain risk-seed confirmation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .causal_chains import CAUSAL_CHAIN_TEMPLATES


PROTOCOL = "iac-causal-seed-annotation-v1"
PUBLIC_CHAIN_LABELS = (*sorted(CAUSAL_CHAIN_TEMPLATES), "none_of_four", "uncertain")
PUBLIC_RESPONSE_EVENTS = (
    "maintain_speed",
    "accelerate",
    "decelerate",
    "emergency_brake",
    "yield",
    "stop",
    "restart",
    "keep_lane",
    "lane_change_left",
    "lane_change_right",
    "avoid_left",
    "avoid_right",
    "gap_accept",
    "turn_left",
    "turn_right",
    "merge_left",
    "merge_right",
    "other",
    "uncertain",
)
PUBLIC_RESOLUTION_STATES = (
    "safe_progress",
    "safe_stop_unresolved",
    "unsafe",
    "not_observed",
    "uncertain",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_frame(source: Path, target: Path, size: tuple[int, int]) -> None:
    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {source}")
    target_width, target_height = size
    source_height, source_width = image.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = source_width / source_height
    if source_aspect > target_aspect:
        crop_width = max(1, round(source_height * target_aspect))
        left = (source_width - crop_width) // 2
        image = image[:, left : left + crop_width]
    elif source_aspect < target_aspect:
        crop_height = max(1, round(source_width / target_aspect))
        top = (source_height - crop_height) // 2
        image = image[top : top + crop_height]
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"failed to write normalized image: {target}")


def _validate_candidate(record: Mapping[str, Any], index: int) -> None:
    if record.get("protocol") != "iac-causal-candidate-v1":
        raise ValueError(f"candidate {index}: invalid protocol")
    if not record.get("candidate_id"):
        raise ValueError(f"candidate {index}: missing candidate_id")
    if record.get("chain_type") not in CAUSAL_CHAIN_TEMPLATES:
        raise ValueError(f"candidate {index}: invalid chain_type")
    if record.get("trigger_label_status") != "candidate_only_requires_blind_confirmation":
        raise ValueError(f"candidate {index}: trigger label was not marked candidate-only")
    history = list(record.get("history_images") or [])
    future = list(record.get("future_images") or [])
    history_offsets = list(record.get("history_offsets_s") or [])
    future_offsets = list(record.get("future_offsets_s") or [])
    if not history or not future:
        raise ValueError(f"candidate {index}: history and future images are required")
    if len(history) != len(history_offsets) or len(future) != len(future_offsets):
        raise ValueError(f"candidate {index}: frame/offset length mismatch")
    if any(float(offset) > 0 for offset in history_offsets):
        raise ValueError(f"candidate {index}: history offsets must be non-positive")
    if any(float(offset) <= 0 for offset in future_offsets):
        raise ValueError(f"candidate {index}: future offsets must be positive")
    if 0.0 not in [float(value) for value in history_offsets]:
        raise ValueError(f"candidate {index}: history must include the anchor at 0 s")
    missing = [str(path) for path in history + future if not Path(str(path)).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])


def build_blind_causal_seed_pack(
    candidates: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    seed: str = "iac-causal-seed-blind-v1",
    normalize_media: bool = True,
    media_size: tuple[int, int] = (960, 540),
) -> dict[str, Any]:
    """Create an opaque public task pack and a separate provenance key."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if not candidates:
        raise ValueError("at least one causal candidate is required")
    if media_size[0] < 1 or media_size[1] < 1:
        raise ValueError("media_size dimensions must be positive")

    validated = [dict(record) for record in candidates]
    for index, record in enumerate(validated):
        _validate_candidate(record, index)
    candidate_ids = [str(record["candidate_id"]) for record in validated]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be globally unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    public_root = output_dir / "public"
    private_root = output_dir / "private"
    media_root = public_root / "media"
    public_root.mkdir()
    private_root.mkdir(mode=0o700)
    media_root.mkdir()

    ordered = sorted(
        validated,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['candidate_id']}".encode("utf-8")
        ).hexdigest(),
    )
    tasks: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for index, record in enumerate(ordered):
        item_id = f"chain-{index:05d}"
        item_dir = media_root / item_id
        item_dir.mkdir()
        history = [Path(str(path)) for path in record["history_images"]]
        future = [Path(str(path)) for path in record["future_images"]]
        offsets = [
            *(float(value) for value in record["history_offsets_s"]),
            *(float(value) for value in record["future_offsets_s"]),
        ]
        copied: list[str] = []
        for frame_index, source in enumerate(history + future):
            role = "history" if frame_index < len(history) else "future"
            role_index = frame_index if role == "history" else frame_index - len(history)
            suffix = ".png" if normalize_media else (source.suffix.lower() or ".png")
            target = item_dir / f"{role}_{role_index:02d}{suffix}"
            if normalize_media:
                _normalize_frame(source, target, media_size)
            else:
                shutil.copyfile(source, target)
            copied.append(target.relative_to(public_root).as_posix())

        tasks.append(
            {
                "protocol": PROTOCOL,
                "item_id": item_id,
                "frame_paths": copied,
                "frame_offsets_s": offsets,
                "anchor_frame_index": len(history) - 1,
                "history_frame_count": len(history),
                "future_frame_count": len(future),
                "allowed_chain_types": list(PUBLIC_CHAIN_LABELS),
                "allowed_response_events": list(PUBLIC_RESPONSE_EVENTS),
                "allowed_resolution_states": list(PUBLIC_RESOLUTION_STATES),
            }
        )
        templates.append(
            {
                "protocol": PROTOCOL,
                "item_id": item_id,
                "annotator_id": "REPLACE_WITH_ANNOTATOR_ID",
                "clip_observable": None,
                "chain_type": None,
                "trigger_present": None,
                "trigger_onset_offset_s": None,
                "conflict_present": None,
                "conflict_onset_offset_s": None,
                "ego_response_events": [],
                "response_onset_offset_s": None,
                "resolution_state": None,
                "resolution_offset_s": None,
                "stage_observable": {
                    "trigger": None,
                    "conflict": None,
                    "response": None,
                    "resolution": None,
                },
                "confidence_1_to_5": None,
                "notes": "",
            }
        )
        private_rows.append(
            {
                **record,
                "protocol": f"{PROTOCOL}-private-key",
                "item_id": item_id,
            }
        )

    protocol_text = """# IAC Causal Seed Annotation V1

Label only what is visible in the ordered frames. Do not infer a trigger from
the ego response alone. Use `none_of_four` when the clip is observable but no
registered chain is present; use `uncertain` only when evidence is ambiguous.
Mark each stage independently. Onset and resolution values must be chosen from
the provided `frame_offsets_s`. This pack confirms real risk seeds; it does not
score a WAM or create a counterfactual pair.
"""
    _write_jsonl(public_root / "tasks.jsonl", tasks)
    _write_jsonl(public_root / "annotations_template.jsonl", templates)
    _write_jsonl(private_root / "private_key.jsonl", private_rows)
    (public_root / "PROTOCOL.md").write_text(protocol_text, encoding="utf-8")
    (private_root / "private_key.jsonl").chmod(0o600)

    public_paths = [
        public_root / "tasks.jsonl",
        public_root / "annotations_template.jsonl",
        public_root / "PROTOCOL.md",
        *sorted(path for path in media_root.rglob("*") if path.is_file()),
    ]
    public_checksums = [
        f"{_sha256(path)}  {path.relative_to(public_root).as_posix()}"
        for path in public_paths
    ]
    (public_root / "SHA256SUMS.txt").write_text(
        "\n".join(public_checksums) + "\n", encoding="ascii"
    )
    private_key = private_root / "private_key.jsonl"
    (private_root / "SHA256SUMS.txt").write_text(
        f"{_sha256(private_key)}  private_key.jsonl\n", encoding="ascii"
    )
    summary = {
        "protocol": PROTOCOL,
        "seed": seed,
        "num_items": len(tasks),
        "frames_per_item": sorted({len(task["frame_paths"]) for task in tasks}),
        "media_normalization": {
            "enabled": bool(normalize_media),
            "center_crop": bool(normalize_media),
            "width": int(media_size[0]),
            "height": int(media_size[1]),
            "format": "png" if normalize_media else "source",
        },
        "minimum_independent_annotators": 3,
        "distribute_directory": "public/",
        "withhold_directory": "private/",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary
