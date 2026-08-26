#!/usr/bin/env python3
"""Convert Epona relative pose/degree-yaw controls to cumulative SE(2) radians."""
import argparse, json
from pathlib import Path
import numpy as np

def compose(relative):
    pose = np.zeros(3, dtype=np.float64)
    out = []
    for item in relative:
        dx, dy, yaw_deg = np.asarray(item, dtype=np.float64)
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pose[:2] += np.asarray([c * dx - s * dy, s * dx + c * dy])
        pose[2] += np.deg2rad(yaw_deg)
        out.append(pose.copy().tolist())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--image-manifest', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    raw = {r['branch_id']: r for r in json.loads(args.raw.read_text())}
    images = {r['branch_id']: r for r in json.loads(args.image_manifest.read_text())}
    refs = {}
    for line in args.reference.read_text().splitlines():
        if line.strip():
            r = json.loads(line); refs[r['branch_id']] = r
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w') as f:
        for branch_id, image in images.items():
            if branch_id not in raw or branch_id not in refs:
                raise KeyError(branch_id)
            r = dict(refs[branch_id])
            r['history_images'] = image['history_images']
            r['future_images'] = image['future_images']
            r['future_images_source'] = 'epona_generated_common_random'
            r['action_trajectory'] = compose(raw[branch_id]['action_trajectory'])
            r['trajectory'] = r['action_trajectory']
            # Four generated frames are at Epona's native 5 Hz clock.
            r['future_times_s'] = [0.2 * (i + 1) for i in range(len(r['action_trajectory']))]
            f.write(json.dumps(r) + '\n'); count += 1
    print(json.dumps({'rows': count, 'output': str(args.output)}))

if __name__ == '__main__':
    main()
