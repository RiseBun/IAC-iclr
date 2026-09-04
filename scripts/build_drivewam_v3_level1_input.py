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
        action = list(g.get('action_trajectory') or [])
        action4 = [action[j] for j in (1, 3, 5, 7)]
        row={
            'sample_id':r.get('sample_id',r['source_key']), 'source_key':r['source_key'],
            'history_frame_paths':list(r['history_images']), 'future_frame_paths':fut,
            'history_times_s':list(r.get('history_times_s',[-1.5,-1.0,-0.5,0.0])),
            'future_times_s':list(g['future_times_s']), 'intrinsics':r['intrinsics'],
            'distortion':r.get('distortion',[]), 'camera_to_ego':r['camera_to_ego'],
            'metadata':{'history_ego_state':r.get('history_ego_state'),'stratum':r.get('stratum'),'benchmark_id':r.get('benchmark_id'),'wam_model_id':g.get('wam_model_id')},
            'action_trajectory':action4, 'action_trajectory_source':g.get('action_trajectory_source'),
            'future_images_source':'wam_generated', 'wam_model_id':g.get('wam_model_id'),
            'gt_candidate_id':'wam_action_head',
            'candidates':[{'candidate_id':'wam_action_head','prior':1.0,'trajectory':action4},{'candidate_id':'zero_null','prior':0.1,'trajectory':[[0.0,0.0,0.0] for _ in action4]}],
        }
        out.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in out),encoding='utf-8'); print(json.dumps({'rows':len(out),'output':str(a.output)}))
if __name__=='__main__': main()
