#!/usr/bin/env python3
"""Attach DriveWAM decoded frames to native NAVSIM state records."""
import argparse, json, pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivewam-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--future-count", type=int, default=1)
    ap.add_argument("--real-future", action="store_true", help="use native NAVSIM future images as IAC control")
    args = ap.parse_args()
    rows = []
    for item in json.loads(Path(args.drivewam_manifest).read_text()):
        source_sample = Path(item["source_sample"]).resolve()
        sample = pickle.load(open(source_sample, "rb"))
        meta = sample["metadata"]
        h = int(len(sample["history_poses"]))
        ft = sample["future_trajectory"]
        future_states = []
        for i, step in enumerate(ft[: args.future_count]):
            pose = np.asarray(step["pose"], dtype=float)
            speed = float(step.get("velocity", 0.0))
            future_states.append([float(pose[0]), float(pose[1]), float(pose[2]), speed, 0.0])
        future_times = [0.5 * (i + 1) for i in range(len(future_states))]
        native_future = [str(Path(p).resolve()) for p in meta["image_paths"][h : h + args.future_count]]
        future_paths = native_future if args.real_future else [str(Path(p).resolve()) for p in item["future_images"][: args.future_count]]
        rows.append({
            "protocol": "native-realized-state-v1",
            "record_type": "drivewam_generated_future",
            "dataset": "navsim",
            "source_key": f"drivewam:{meta.get('scene_token','')}:{meta.get('frame_idx',0)}:{item['sample_index']}",
            "scene_name": str(meta.get("scene_token", "")),
            "history_images": [str(p) for p in meta["image_paths"][:h]],
            "future_images": future_paths,
            "future_images_source": "native_dataset" if args.real_future else "wam_generated",
            "history_ego_state": [[*np.asarray(p, dtype=float).tolist(), 0.0, 0.0] for p in sample["history_poses"]],
            "realized_future_ego_state": future_states,
            "future_times_s": future_times,
            "trajectory": [s[:3] for s in future_states],
            "trajectory_source": "native_navsim_ego_state",
            "camera_intrinsic": [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]],
            "camera_distortion": [],
            "camera_to_ego": [[-0.0030311323941387964, -0.019786295321436612, 0.9997996373043262, 1.6240250126233996],
                               [-0.9999953180155606, -0.00035968662895047944, -0.003038843939251048, -0.0055507164874228345],
                               [0.00041974202478565527, -0.9998041673962867, -0.019785112424517044, 1.5331206139432636],
                               [0.0, 0.0, 0.0, 1.0]],
            "task_success": None,
            "task_success_source": None,
            "wam_model_id": "drivewam_navsim",
            "wam_generation_status": "complete",
        })
    out = Path(args.output)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(json.dumps({"output": str(out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
