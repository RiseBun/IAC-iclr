#!/usr/bin/env python3
"""Build a Level-1 diagnostic manifest for WorldDrive action interventions.

The resulting rows are intentionally labelled ``external_intervention``.
They test whether the candidate-blind image decoder recovers the injected
trajectory ordering; they are not native-action-head CCFC records.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np


BRANCHES = ("left", "logged", "right")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample_index(group_id: str) -> int:
    match = re.fullmatch(r"worlddrive-(\d{6})", group_id)
    if match is None:
        raise ValueError(f"unsupported WorldDrive group id: {group_id}")
    return int(match.group(1))


def _load_sample(root: Path, branch: str, index: int) -> dict:
    path = root / f"drivewam_samples_{branch}" / f"sample_{index:06d}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def _trajectory(sample: dict) -> list[list[float]]:
    value = np.asarray([row["pose"] for row in sample["future_trajectory"]], dtype=np.float64)
    if value.shape != (8, 3):
        raise ValueError(f"expected 8x3 trajectory, got {value.shape}")
    return value.tolist()


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
    by_group: dict[str, dict[str, dict]] = {}
    for row in generated:
        by_group.setdefault(str(row["counterfactual_group_id"]), {})[str(row["branch_mode"])] = row

    for group_id, generated_branches in sorted(by_group.items()):
        if set(generated_branches) != set(BRANCHES):
            raise ValueError(f"{group_id}: expected branches {BRANCHES}")
        index = _sample_index(group_id)
        samples = {branch: _load_sample(args.sample_root, branch, index) for branch in BRANCHES}
        histories = {np.ascontiguousarray(sample["images"][:4]).tobytes() for sample in samples.values()}
        if len(histories) != 1:
            raise RuntimeError(f"{group_id}: histories differ")
        history_paths, intrinsic = _materialize_history(
            samples["logged"], args.history_output_root / f"sample_{index:06d}"
        )
        candidates = [
            {
                "candidate_id": f"{branch}_intervention",
                "trajectory": _trajectory(samples[branch]),
                "prior": 1.0,
                "support_label": "external_action_response_probe",
            }
            for branch in BRANCHES
        ]
        for branch in BRANCHES:
            generated_row = generated_branches[branch]
            action = _trajectory(samples[branch])
            output.append({
                "sample_id": f"{group_id}::{branch}",
                "scene_id": str(samples[branch]["metadata"].get("scene_token", group_id)),
                "source_key": str(samples[branch]["metadata"].get("source_key", group_id)),
                "counterfactual_group_id": group_id,
                "branch_mode": branch,
                "history_frame_paths": history_paths,
                "future_frame_paths": generated_row["future_images"],
                "history_times_s": [-1.5, -1.0, -0.5, 0.0],
                "future_times_s": [0.5 * value for value in range(1, 9)],
                "intrinsics": intrinsic,
                "distortion": [],
                "camera_to_ego": samples[branch]["metadata"]["camera_to_ego"],
                "candidates": candidates,
                "gt_candidate_id": f"{branch}_intervention",
                "action_trajectory": action,
                "action_trajectory_source": "external_intervention",
                "future_images_source": "worlddrive_generated",
                "wam_model_id": "worlddrive_tadwm_1024",
                "metadata": {
                    "protocol": "worlddrive-action-response-probe-history4-future8-v1",
                    "history_ego_state": _history_state(samples[branch]),
                    "candidate_blind_image_branch": True,
                    "candidate_bank_used_by_decoder": False,
                    "action_waypoint_used_by_image_branch": False,
                    "native_action_head_recorded": False,
                    "evidence_tier": "action_response_probe",
                },
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(output), "groups": len(by_group)}, indent=2))


if __name__ == "__main__":
    main()
