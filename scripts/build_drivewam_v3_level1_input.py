#!/usr/bin/env python3
"""Join v3 WAM generated frames with private geometry for Level-1 probe."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def lines(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v3',type=Path,required=True); ap.add_argument('--generated',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    base={r['source_key']:r for r in lines(a.v3)}; gen={r['source_key']:r for r in lines(a.generated)}; out=[]
    for r in sorted(base.values(), key=lambda x:int(str(x['benchmark_id']).rsplit('-',1)[-1])):
        g=gen[r['source_key']]; fut=list(g['future_images']);
        if len(fut)!=4: raise ValueError(f"{r['source_key']}: expected 4 generated frames")
        row={
            'sample_id':r.get('sample_id',r['source_key']), 'source_key':r['source_key'],
            'frame_paths':list(r['history_images'])+fut, 'history_count':4,
            'history_times_s':list(r.get('history_times_s',[-1.5,-1.0,-0.5,0.0])),
            'future_times_s':list(g['future_times_s']), 'intrinsics':r['intrinsics'],
            'distortion':r.get('distortion',[]), 'camera_to_ego':r['camera_to_ego'],
            'metadata':{'history_ego_state':r.get('history_ego_state'),'stratum':r.get('stratum'),'benchmark_id':r.get('benchmark_id'),'wam_model_id':g.get('wam_model_id')},
            'action_trajectory':g.get('action_trajectory'), 'action_trajectory_source':g.get('action_trajectory_source'),
            'future_images_source':'wam_generated', 'wam_model_id':g.get('wam_model_id'),
        }
        out.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out),encoding='utf-8'); print(json.dumps({'rows':len(out),'output':str(a.output)}))
if __name__=='__main__': main()
