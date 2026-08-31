#!/usr/bin/env python3
"""Build candidate-blind Level-1 records for WorldDrive native action/future pairs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_sample(root: Path, index: int) -> dict:
    path = root / "drivewam_samples_logged" / f"sample_{index:06d}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def _materialize_history(sample: dict, output_dir: Path) -> tuple[list[str], list[list[float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = np.asarray(sample["images"][:4])
    paths = []
    for index, image in enumerate(images):
        cropped = image[28:-28]
        resized = cv2.resize(cropped, (1024, 512), interpolation=cv2.INTER_AREA)
        path = output_dir / f"frame_{index:02d}.png"
        if not cv2.imwrite(str(path), resized[..., ::-1]):
            raise RuntimeError(f"failed to write {path}")
        paths.append(str(path.resolve()))

    intrinsic = np.asarray(sample["metadata"]["camera_intrinsic"], dtype=np.float64).copy()
    source_height, source_width = images.shape[1:3]
    scale_x = 1024.0 / source_width
    scale_y = 512.0 / (source_height - 56)
    intrinsic[0, 0] *= scale_x
    intrinsic[0, 2] *= scale_x
    intrinsic[1, 1] *= scale_y
    intrinsic[1, 2] = (intrinsic[1, 2] - 28.0) * scale_y
    return paths, intrinsic.tolist()


def _history_state(sample: dict) -> list[list[float]]:
    poses = np.asarray(sample["history_poses"], dtype=np.float64)
    speeds = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) / 0.5
    latest = float(np.linalg.norm(np.asarray(sample["ego_status"]["velocity"], dtype=np.float64)))
    speeds = np.concatenate([[speeds[0] if len(speeds) else latest], speeds])
    speeds[-1] = latest
    return [pose.tolist() + [float(speed), 0.0] for pose, speed in zip(poses, speeds)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--history-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generated = _read_jsonl(args.generated_manifest)
    output = []
    for ordinal, generated_row in enumerate(generated):
        if not generated_row.get("native_action_head_recorded"):
            raise ValueError(f"row {ordinal}: native action head is not recorded")
        index = int(generated_row.get("sample_index", ordinal))
        sample = _load_sample(args.sample_root, index)
        action = np.asarray(generated_row["action_trajectory"], dtype=np.float64)
        if action.shape != (8, 3) or not np.isfinite(action).all():
            raise ValueError(f"row {ordinal}: invalid native action {action.shape}")
        future = list(generated_row["future_images"])
        if len(future) != 8:
            raise ValueError(f"row {ordinal}: expected 8 generated future frames")
        history_paths, intrinsic = _materialize_history(
            sample, args.history_output_root / f"sample_{index:06d}"
        )
        candidate = {
            "candidate_id": "worlddrive_native_action",
            "trajectory": action.tolist(),
            "prior": 1.0,
            "support_label": "independent_native_action_head",
        }
        placeholder = dict(candidate)
        placeholder["candidate_id"] = "schema_placeholder_duplicate"
        output.append({
            "sample_id": generated_row["sample_id"],
            "scene_id": str(sample["metadata"].get("scene_token", generated_row["sample_id"])),
            "source_key": generated_row["source_key"],
            "history_frame_paths": history_paths,
            "future_frame_paths": future,
            "history_times_s": [-1.5, -1.0, -0.5, 0.0],
            "future_times_s": generated_row["future_timestamps"],
            "intrinsics": intrinsic,
            "distortion": [],
            "camera_to_ego": sample["metadata"]["camera_to_ego"],
            "candidates": [candidate, placeholder],
            "gt_candidate_id": "worlddrive_native_action",
            "action_trajectory": action.tolist(),
            "action_trajectory_source": "wam_action_head",
            "future_images_source": "worlddrive_generated",
            "wam_model_id": generated_row["model_id"],
            "metadata": {
                "protocol": "worlddrive-native-action-future-level1-v1",
                "history_ego_state": _history_state(sample),
                "candidate_blind_image_branch": True,
                "candidate_bank_used_by_decoder": False,
                "action_waypoint_used_by_image_branch": False,
                "native_action_head_recorded": True,
                "action_origin": generated_row["action_origin"],
                "planner_checkpoint_sha256": generated_row["planner_checkpoint_sha256"],
                "world_model_checkpoint_sha256": generated_row["world_model_checkpoint_sha256"],
                "evidence_tier": "native_action_future_pair",
                "causal_claim_eligible": False,
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "rows": len(output),
        "native_action_head_recorded": True,
        "candidate_bank_used_by_decoder": False,
        "causal_claim_eligible": False,
    }, indent=2))


if __name__ == "__main__":
    main()
