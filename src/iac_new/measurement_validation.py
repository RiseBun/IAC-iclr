"""Blind human validation for image-derived maneuver event posteriors."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .event_metrics import event_counterfactual_matrix


LATERAL_LABELS = ("keep_lane", "turn_left", "turn_right")
_LABEL_ALIASES = {"straight": "keep_lane", "left": "turn_left", "right": "turn_right"}


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL or a JSON list/report without guessing field semantics."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        for key in ("rows", "records", "groups"):
            if isinstance(value.get(key), list):
                return list(value[key])
    raise ValueError(f"{path} does not contain a record list")


def _branch_map(groups: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for group in groups:
        for branch in group.get("branches", []):
            branch_id = str(branch.get("branch_id", ""))
            if not branch_id or branch_id in output:
                raise ValueError("event groups require globally unique branch_id values")
            output[branch_id] = dict(branch)
    return output


def _media_paths(record: Mapping[str, Any], key: str) -> list[Path]:
    alternatives = {
        "history": ("history_images", "history_frame_paths"),
        "future": ("future_images", "future_frame_paths", "generated_frame_paths"),
    }[key]
    for field in alternatives:
        values = record.get(field)
        if values:
            return [Path(str(value)) for value in values]
    return []


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


def build_blind_annotation_pack(
    sources: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    seed: str = "measurement-validity-v1",
    history_frame_count: int = 2,
    normalize_media: bool = True,
    media_size: tuple[int, int] = (960, 540),
) -> dict[str, Any]:
    """Materialize opaque media tasks and a separate private decoding key."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if history_frame_count < 1:
        raise ValueError("history_frame_count must be positive")
    if media_size[0] < 1 or media_size[1] < 1:
        raise ValueError("media_size dimensions must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    public_root = output_dir / "public"
    private_root = output_dir / "private"
    public_root.mkdir()
    private_root.mkdir(mode=0o700)
    media_root = public_root / "media"
    media_root.mkdir()

    candidates = []
    for source in sources:
        source_id = str(source["source_id"])
        manifest = read_records(Path(source["manifest_path"]))
        branches = _branch_map(read_records(Path(source["event_groups_path"])))
        for record in manifest:
            branch_id = str(record.get("branch_id", ""))
            if branch_id not in branches:
                raise ValueError(f"missing event posterior for branch: {branch_id}")
            branch = branches[branch_id]
            posterior = list(branch.get("imagined_event_posterior", []))
            if not posterior:
                raise ValueError(f"branch has no imagined_event_posterior: {branch_id}")
            history = _media_paths(record, "history")
            future = _media_paths(record, "future")
            if not history or len(future) != len(posterior):
                raise ValueError(
                    f"branch {branch_id} requires history and one future frame per interval"
                )
            missing = [str(path) for path in history + future if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            token = hashlib.sha256(f"{seed}:{source_id}:{branch_id}".encode()).hexdigest()
            candidates.append((token, source_id, dict(record), branch))
    if not candidates:
        raise ValueError("at least one annotation item is required")

    tasks = []
    private_rows = []
    templates = []
    for index, (_, source_id, record, branch) in enumerate(sorted(candidates)):
        item_id = f"evt-{index:05d}"
        item_dir = media_root / item_id
        item_dir.mkdir()
        history = _media_paths(record, "history")[-history_frame_count:]
        future = _media_paths(record, "future")
        copied = []
        for frame_index, source_path in enumerate(history + future):
            role = "history" if frame_index < len(history) else "future"
            role_index = frame_index if role == "history" else frame_index - len(history)
            suffix = ".png" if normalize_media else (source_path.suffix.lower() or ".png")
            target = item_dir / f"{role}_{role_index:02d}{suffix}"
            if normalize_media:
                _normalize_frame(source_path, target, media_size)
            else:
                target.write_bytes(source_path.read_bytes())
            copied.append(target.relative_to(public_root).as_posix())

        posterior = list(branch["imagined_event_posterior"])
        task = {
            "item_id": item_id,
            "frame_paths": copied,
            "history_frame_count": len(history),
            "future_frame_count": len(future),
            "num_intervals": len(posterior),
            "allowed_labels": list(LATERAL_LABELS),
        }
        tasks.append(task)
        private_rows.append({
            "item_id": item_id,
            "source_id": source_id,
            "scene_id": record.get("scene_name", record.get("scene_id")),
            "counterfactual_group_id": record.get("counterfactual_group_id"),
            "branch_id": record.get("branch_id"),
            "condition_action_id": branch.get(
                "condition_action_id", record.get("branch_mode")
            ),
            "future_times_s": list(
                record.get("future_times_s", range(1, len(future) + 1))
            ),
            "probe_event_posterior": posterior,
            "action_event_target": branch.get("action_event_target"),
        })
        templates.append({
            "item_id": item_id,
            "annotator_id": "REPLACE_WITH_ANNOTATOR_ID",
            "interval_labels": [None] * len(posterior),
            "interval_observable": [None] * len(posterior),
            "clip_quality": None,
            "notes": "",
        })

    _write_jsonl(public_root / "tasks.jsonl", tasks)
    _write_jsonl(private_root / "private_key.jsonl", private_rows)
    _write_jsonl(public_root / "annotations_template.jsonl", templates)
    (private_root / "private_key.jsonl").chmod(0o600)
    public_paths = [
        public_root / "tasks.jsonl",
        public_root / "annotations_template.jsonl",
        *sorted(path for path in media_root.rglob("*") if path.is_file()),
    ]
    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(public_root).as_posix()}"
        for path in public_paths
    ]
    (public_root / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    (private_root / "SHA256SUMS.txt").write_text(
        _sha256(private_root / "private_key.jsonl") + "  private_key.jsonl\n",
        encoding="ascii",
    )
    summary = {
        "protocol": "blind-wam-event-annotation-pack-v1",
        "seed": seed,
        "num_items": len(tasks),
        "num_intervals": sum(row["num_intervals"] for row in tasks),
        "num_sources": len({row["source_id"] for row in private_rows}),
        "media_normalization": {
            "enabled": bool(normalize_media),
            "center_crop": bool(normalize_media),
            "width": int(media_size[0]),
            "height": int(media_size[1]),
            "format": "png" if normalize_media else "source",
        },
        "distribute_directory": "public/",
        "withhold_directory": "private/",
        "public_files": [
            "public/tasks.jsonl",
            "public/annotations_template.jsonl",
            "public/media/",
            "public/SHA256SUMS.txt",
        ],
        "private_files": ["private/private_key.jsonl", "private/SHA256SUMS.txt"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return summary


def _canonical_label(label: Any) -> str | None:
    if label is None:
        return None
    canonical = _LABEL_ALIASES.get(str(label), str(label))
    return canonical if canonical in LATERAL_LABELS else None


def _posterior(row: Mapping[str, Any]) -> dict[str, float] | None:
    values = row.get("lateral_posterior")
    if values is None:
        label = _canonical_label(row.get("lateral_event"))
        return None if label is None else {item: float(item == label) for item in LATERAL_LABELS}
    output = {
        _LABEL_ALIASES.get(str(key), str(key)): float(value)
        for key, value in values.items()
        if _LABEL_ALIASES.get(str(key), str(key)) in LATERAL_LABELS
    }
    total = sum(output.values())
    if total <= 0.0 or not all(np.isfinite(value) and value >= 0.0 for value in output.values()):
        raise ValueError("invalid lateral posterior")
    return {label: output.get(label, 0.0) / total for label in LATERAL_LABELS}


def _classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_intervals": 0, "accuracy": None, "macro_f1": None, "nll": None, "brier": None}
    truth = [str(row["human_label"]) for row in rows]
    predicted = [str(row["probe_label"]) for row in rows]
    accuracy = float(np.mean([left == right for left, right in zip(truth, predicted)]))
    f1_values = []
    for label in LATERAL_LABELS:
        tp = sum(left == label and right == label for left, right in zip(truth, predicted))
        fp = sum(left != label and right == label for left, right in zip(truth, predicted))
        fn = sum(left == label and right != label for left, right in zip(truth, predicted))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {
        "num_intervals": len(rows),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "nll": float(np.mean([-math.log(max(float(row["true_probability"]), 1e-12)) for row in rows])),
        "brier": float(np.mean([float(row["brier"]) for row in rows])),
    }


def _agreement(vote_sets: Sequence[Sequence[str]]) -> dict[str, Any]:
    observed = []
    counts = Counter()
    for votes in vote_sets:
        counts.update(votes)
        if len(votes) >= 2:
            matches = sum(left == right for i, left in enumerate(votes) for right in votes[i + 1 :])
            pairs = len(votes) * (len(votes) - 1) / 2
            observed.append(matches / pairs)
    total = sum(counts.values())
    expected = None if not total else sum((counts[label] / total) ** 2 for label in LATERAL_LABELS)
    observed_mean = None if not observed else float(np.mean(observed))
    kappa = None
    if observed_mean is not None and expected is not None and expected < 1.0:
        kappa = float((observed_mean - expected) / (1.0 - expected))
    return {
        "pairwise_agreement": observed_mean,
        "chance_agreement": expected,
        "generalized_kappa": kappa,
        "label_counts": dict(sorted(counts.items())),
    }


def _ece(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> float | None:
    if not rows:
        return None
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = [
            row for row in rows
            if float(row["confidence"]) >= lower
            and (float(row["confidence"]) < upper or np.isclose(upper, 1.0))
        ]
        if selected:
            accuracy = np.mean([row["probe_label"] == row["human_label"] for row in selected])
            confidence = np.mean([float(row["confidence"]) for row in selected])
            result += len(selected) / len(rows) * abs(float(accuracy) - float(confidence))
    return float(result)


def _risk_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
    if not ordered:
        return {"aurc": None, "curve": []}
    errors = np.asarray(
        [row["probe_label"] != row["human_label"] for row in ordered], dtype=np.float64
    )
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return {
        "aurc": float(np.mean(risk)),
        "curve": [
            {
                "coverage": float((index + 1) / len(ordered)),
                "risk": float(risk[index]),
                "minimum_confidence": float(ordered[index]["confidence"]),
            }
            for index in range(len(ordered))
        ],
    }


def _bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0 or not rows:
        return {"samples": samples, "cluster": "scene_id", "confidence_intervals": {}}
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(row)
    scenes = sorted(by_scene)
    if len(scenes) < 2:
        return {"samples": samples, "cluster": "scene_id", "confidence_intervals": {}}
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(samples):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        sample_rows = [row for scene in selected for row in by_scene[str(scene)]]
        metrics = _classification_metrics(sample_rows)
        for key in ("accuracy", "macro_f1", "nll", "brier"):
            values[key].append(metrics[key])
    return {
        "samples": samples,
        "cluster": "scene_id",
        "num_clusters": len(scenes),
        "confidence_intervals": {
            key: {
                "lower_95": float(np.percentile(items, 2.5)),
                "upper_95": float(np.percentile(items, 97.5)),
            }
            for key, items in values.items()
        },
    }


def _mean_available(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _human_probe_event_cc(
    private_rows: Sequence[Mapping[str, Any]],
    human_sequences: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    minimum_probe_observability: float,
) -> dict[str, Any]:
    grouped = defaultdict(list)
    for private in private_rows:
        group_id = private.get("counterfactual_group_id")
        action_id = private.get("condition_action_id")
        action_target = private.get("action_event_target")
        item_id = str(private["item_id"])
        posterior = list(private.get("probe_event_posterior", []))
        consensus = human_sequences.get(item_id, {})
        if group_id is None or action_id is None or not action_target:
            continue
        if len(consensus) != len(posterior):
            continue
        grouped[(str(private.get("source_id")), str(group_id))].append({
            "item_id": item_id,
            "condition_action_id": str(action_id),
            "action_event_target": action_target,
            "probe_event_posterior": posterior,
            "human_event_posterior": [consensus[index] for index in range(len(posterior))],
        })

    rows = []
    for (source_id, group_id), branches in sorted(grouped.items()):
        if len(branches) < 2:
            continue
        action_ids = [branch["condition_action_id"] for branch in branches]
        if len(set(action_ids)) != len(action_ids):
            continue
        common = [
            {
                "branch_id": branch["item_id"],
                "condition_action_id": branch["condition_action_id"],
                "action_event_target": branch["action_event_target"],
            }
            for branch in branches
        ]
        human_branches = [
            {**base, "imagined_event_posterior": branch["human_event_posterior"]}
            for base, branch in zip(common, branches)
        ]
        probe_branches = [
            {**base, "imagined_event_posterior": branch["probe_event_posterior"]}
            for base, branch in zip(common, branches)
        ]
        human = event_counterfactual_matrix(
            human_branches,
            minimum_observability=0.0,
            minimum_interval_coverage=1.0,
        )
        probe = event_counterfactual_matrix(
            probe_branches,
            minimum_observability=minimum_probe_observability,
            minimum_interval_coverage=0.5,
        )
        rows.append({
            "source_id": source_id,
            "counterfactual_group_id": group_id,
            "num_branches": len(branches),
            "human_diagonal_top1_accuracy": human["diagonal_top1_accuracy"],
            "probe_diagonal_top1_accuracy": probe["diagonal_top1_accuracy"],
            "human_mean_cc_margin": human["mean_cc_margin"],
            "probe_mean_cc_margin": probe["mean_cc_margin"],
            "probe_num_evaluable": probe["num_evaluable"],
        })
    paired_margins = [
        (row["human_mean_cc_margin"], row["probe_mean_cc_margin"])
        for row in rows
        if row["human_mean_cc_margin"] is not None and row["probe_mean_cc_margin"] is not None
    ]
    correlation = None
    margin_mae = None
    if paired_margins:
        human_values = np.asarray([item[0] for item in paired_margins], dtype=np.float64)
        probe_values = np.asarray([item[1] for item in paired_margins], dtype=np.float64)
        margin_mae = float(np.mean(np.abs(human_values - probe_values)))
        if len(paired_margins) >= 2 and np.std(human_values) > 0 and np.std(probe_values) > 0:
            correlation = float(np.corrcoef(human_values, probe_values)[0, 1])
    return {
        "num_groups": len(rows),
        "human_diagonal_top1_accuracy": _mean_available(
            rows, "human_diagonal_top1_accuracy"
        ),
        "probe_diagonal_top1_accuracy": _mean_available(
            rows, "probe_diagonal_top1_accuracy"
        ),
        "human_mean_cc_margin": _mean_available(rows, "human_mean_cc_margin"),
        "probe_mean_cc_margin": _mean_available(rows, "probe_mean_cc_margin"),
        "human_probe_cc_margin_correlation": correlation,
        "human_probe_cc_margin_mae": margin_mae,
        "rows": rows,
    }


def score_measurement_validity(
    private_rows: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    minimum_annotators: int = 3,
    consensus_fraction: float = 2.0 / 3.0,
    minimum_probe_observability: float = 0.25,
    bootstrap_samples: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Score a frozen probe against blinded multi-annotator consensus."""
    if minimum_annotators < 2:
        raise ValueError("minimum_annotators must be at least two")
    if not 0.5 < consensus_fraction <= 1.0:
        raise ValueError("consensus_fraction must be in (0.5,1]")
    private_by_id = {str(row["item_id"]): row for row in private_rows}
    if len(private_by_id) != len(private_rows):
        raise ValueError("private key item_id values must be unique")
    annotations_by_id = defaultdict(list)
    seen = set()
    for row in annotations:
        item_id = str(row.get("item_id", ""))
        annotator_id = str(row.get("annotator_id", ""))
        if item_id not in private_by_id or not annotator_id:
            raise ValueError("annotation has unknown item_id or missing annotator_id")
        identity = (item_id, annotator_id)
        if identity in seen:
            raise ValueError("one annotator may submit only one row per item")
        seen.add(identity)
        annotations_by_id[item_id].append(row)

    scored = []
    human_sequences = defaultdict(dict)
    vote_sets = []
    exclusion_counts = Counter()
    for item_id, private in private_by_id.items():
        posterior_rows = list(private.get("probe_event_posterior", []))
        item_annotations = annotations_by_id.get(item_id, [])
        for annotation in item_annotations:
            if len(annotation.get("interval_labels", [])) != len(posterior_rows):
                raise ValueError(f"annotation interval length mismatch: {item_id}")
            if len(annotation.get("interval_observable", [])) != len(posterior_rows):
                raise ValueError(f"annotation observability length mismatch: {item_id}")
        for interval_index, probe_row in enumerate(posterior_rows):
            votes = []
            for annotation in item_annotations:
                if annotation["interval_observable"][interval_index] is True:
                    label = _canonical_label(annotation["interval_labels"][interval_index])
                    if label is None:
                        raise ValueError("observable annotation requires a supported label")
                    votes.append(label)
            vote_sets.append(votes)
            if len(votes) < minimum_annotators:
                exclusion_counts["insufficient_annotators"] += 1
                continue
            counts = Counter(votes)
            top_count = max(counts.values())
            winners = [label for label, count in counts.items() if count == top_count]
            if len(winners) != 1 or top_count / len(votes) < consensus_fraction:
                exclusion_counts["no_human_consensus"] += 1
                continue
            human_label = winners[0]
            human_sequences[item_id][interval_index] = {
                "lateral_event": human_label,
                "lateral_posterior": {
                    label: counts.get(label, 0) / len(votes) for label in LATERAL_LABELS
                },
                "observability": top_count / len(votes),
                "abstain": False,
            }
            if bool(probe_row.get("abstain", False)):
                exclusion_counts["probe_abstain"] += 1
                continue
            observability = float(probe_row.get("observability", 1.0))
            if observability < minimum_probe_observability:
                exclusion_counts["low_probe_observability"] += 1
                continue
            probabilities = _posterior(probe_row)
            if probabilities is None:
                exclusion_counts["missing_probe_posterior"] += 1
                continue
            probe_label = max(LATERAL_LABELS, key=lambda label: probabilities[label])
            confidence = float(probabilities[probe_label])
            brier = sum(
                (probabilities[label] - float(label == human_label)) ** 2
                for label in LATERAL_LABELS
            )
            scored.append({
                "item_id": item_id,
                "interval_index": interval_index,
                "source_id": private.get("source_id"),
                "scene_id": private.get("scene_id", private.get("counterfactual_group_id")),
                "human_label": human_label,
                "human_agreement": top_count / len(votes),
                "probe_label": probe_label,
                "confidence": confidence,
                "true_probability": probabilities[human_label],
                "brier": float(brier),
                "probe_observability": observability,
            })

    total_intervals = sum(len(row.get("probe_event_posterior", [])) for row in private_rows)
    consensus_intervals = len(scored) + sum(
        exclusion_counts[key]
        for key in ("probe_abstain", "low_probe_observability", "missing_probe_posterior")
    )
    metrics = _classification_metrics(scored)
    metrics.update({
        "ece_10_bin": _ece(scored),
        "coverage_over_human_consensus": (
            0.0 if not consensus_intervals else len(scored) / consensus_intervals
        ),
        "coverage_over_all_intervals": 0.0 if not total_intervals else len(scored) / total_intervals,
    })
    per_source = {
        source: _classification_metrics([row for row in scored if row["source_id"] == source])
        for source in sorted({str(row["source_id"]) for row in scored})
    }
    return {
        "protocol": "wam-event-measurement-validity-v1",
        "num_items": len(private_rows),
        "num_annotations": len(annotations),
        "num_total_intervals": total_intervals,
        "num_scored_intervals": len(scored),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "human_agreement": _agreement(vote_sets),
        "probe_metrics": metrics,
        "risk_coverage": _risk_coverage(scored),
        "scene_bootstrap": _bootstrap(scored, samples=bootstrap_samples, seed=seed),
        "human_probe_event_cc": _human_probe_event_cc(
            private_rows,
            human_sequences,
            minimum_probe_observability=minimum_probe_observability,
        ),
        "per_source": per_source,
        "rows": scored,
    }
