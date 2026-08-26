#!/usr/bin/env python3
"""Compare SegFormer road boundaries with the independent LiDAR oracle."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from iac_new.perception import build_perception
from iac_new.road_structure import extract_road_boundaries

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    perception = build_perception(cfg, device=args.device)
    rows = []
    for item in [json.loads(x) for x in args.oracle.read_text().splitlines() if x.strip()]:
        image = cv2.imread(item['anchor_image'], cv2.IMREAD_COLOR)
        if image is None: continue
        h, w = int(cfg['image']['height']), int(cfg['image']['width'])
        obs = perception.observe(
            [item['anchor_image']], target_size=(w, h),
            intrinsics=np.asarray(item.get('camera_intrinsic'), dtype=np.float64),
            distortion=np.asarray(item.get('distortion', []), dtype=np.float64),
        )
        seg = extract_road_boundaries(np.asarray(obs.traversable_masks[0], dtype=bool), row_step=4)
        oracle = item.get('boundary_descriptor', {})
        if not seg.get('valid') or not oracle.get('rows'):
            rows.append({'source_key': item.get('source_key'), 'valid': False})
            continue
        oy = np.asarray(oracle['rows'], dtype=float)
        ol = np.asarray(oracle['left_x'], dtype=float)
        oright = np.asarray(oracle['right_x'], dtype=float)
        sy = np.asarray(seg['rows'], dtype=float)
        sl = np.interp(oy, sy, np.asarray(seg['left_x'], dtype=float))
        sr = np.interp(oy, sy, np.asarray(seg['right_x'], dtype=float))
        center_err = 0.5 * (sl + sr - ol - oright)
        width_err = (sr - sl) - (oright - ol)
        far = oy < 0.62 * h
        mid = (oy >= 0.62 * h) & (oy < 0.72 * h)
        near = oy >= 0.72 * h
        def stats(mask):
            vals = np.abs(center_err[mask])
            return {'n': int(mask.sum()), 'median_center_abs_px': float(np.median(vals)) if vals.size else None, 'p90_center_abs_px': float(np.quantile(vals, .9)) if vals.size else None, 'median_width_abs_px': float(np.median(np.abs(width_err[mask]))) if vals.size else None}
        rows.append({'source_key': item.get('source_key'), 'valid': True, 'segformer_confidence': float(seg.get('confidence', 0.0)), 'oracle_confidence': float(item.get('boundary_confidence', 0.0)), 'all': stats(np.ones_like(oy, dtype=bool)), 'far': stats(far), 'mid': stats(mid), 'near': stats(near)})
    valid = [r for r in rows if r.get('valid')]
    def aggregate(name):
        vals = [r[name]['median_center_abs_px'] for r in valid if r[name]['median_center_abs_px'] is not None]
        p90 = [r[name]['p90_center_abs_px'] for r in valid if r[name]['p90_center_abs_px'] is not None]
        return {'median_of_medians_px': float(np.median(vals)) if vals else None, 'median_p90_px': float(np.median(p90)) if p90 else None}
    result = {'protocol': 'segformer-lidar-boundary-audit-v1', 'num_input': len(rows), 'num_valid': len(valid), 'valid_fraction': float(len(valid)/max(len(rows),1)), 'aggregate': {k: aggregate(k) for k in ('all','far','mid','near')}, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('protocol','num_input','num_valid','valid_fraction','aggregate')}, indent=2))
if __name__ == '__main__': main()
