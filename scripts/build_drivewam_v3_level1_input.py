#!/usr/bin/env python3
"""Join v3 WAM generated frames with private geometry for Level-1 probe."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def lines(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def resize_intrinsics(k, source_size, target_size):
    """Map pinhole intrinsics through the same resize used for history frames."""
    sx = float(target_size[0]) / float(source_size[0])
    sy = float(target_size[1]) / float(source_size[1])
    out = np.asarray(k, dtype=float).reshape(3, 3).copy()
    out[0, :] *= sx
    out[1, :] *= sy
    out[2, :] = [0.0, 0.0, 1.0]
    return out.tolist()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v3',type=Path,required=True); ap.add_argument('--generated',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--history-root',type=Path,required=True); ap.add_argument('--intrinsics-source-width',type=int,default=1920); ap.add_argument('--intrinsics-source-height',type=int,default=1080); a=ap.parse_args()
    base={r['source_key']:r for r in lines(a.v3)}; gen={r['source_key']:r for r in lines(a.generated)}; out=[]
    for r in sorted(base.values(), key=lambda x:int(str(x['benchmark_id']).rsplit('-',1)[-1])):
        g=gen[r['source_key']]; fut=list(g['future_images']);
        if len(fut)!=4: raise ValueError(f"{r['source_key']}: expected 4 generated frames")
        action = list(g.get('action_trajectory') or [])
        action4 = [action[j] for j in (1, 3, 5, 7)]
        history=[]; hdir=a.history_root / str(r.get('benchmark_id',len(out))); hdir.mkdir(parents=True,exist_ok=True)
        # K is supplied in the source camera coordinate system.  The v3
        # files may already be resized, so their dimensions cannot identify
        # that coordinate system.
        source_size = [int(a.intrinsics_source_width), int(a.intrinsics_source_height)]
        target_size = [448, 256]
        calibrated_k = resize_intrinsics(r['intrinsics'], source_size, target_size)
        for i,p in enumerate(r['history_images']):
            hp=hdir/f'history_{i:02d}.png'
            if not hp.exists():
                with Image.open(p) as im: im.convert('RGB').resize((448,256),Image.Resampling.BILINEAR).save(hp)
            history.append(str(hp))
        row={
            'sample_id':r.get('sample_id',r['source_key']), 'source_key':r['source_key'],
            'history_frame_paths':history, 'future_frame_paths':fut,
            'history_times_s':list(r.get('history_times_s',[-1.5,-1.0,-0.5,0.0])),
            'future_times_s':list(g['future_times_s']), 'intrinsics':calibrated_k,
            'distortion':r.get('distortion',[]), 'camera_to_ego':r['camera_to_ego'],
            'metadata':{'history_ego_state':r.get('history_ego_state'),'stratum':r.get('stratum'),'benchmark_id':r.get('benchmark_id'),'wam_model_id':g.get('wam_model_id'), 'intrinsics_source_size':source_size, 'intrinsics_image_size':target_size, 'intrinsics_transform':'resize_only'},
            'action_trajectory':action4, 'action_trajectory_source':g.get('action_trajectory_source'),
            'future_images_source':'wam_generated', 'wam_model_id':g.get('wam_model_id'),
            'gt_candidate_id':'wam_action_head',
            'candidates':[{'candidate_id':'wam_action_head','prior':1.0,'trajectory':action4},{'candidate_id':'zero_null','prior':0.1,'trajectory':[[0.0,0.0,0.0] for _ in action4]}],
        }
        out.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out),encoding='utf-8'); print(json.dumps({'rows':len(out),'output':str(a.output)}))
if __name__=='__main__': main()
