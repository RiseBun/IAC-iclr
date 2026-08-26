#!/usr/bin/env python3
"""Summarize raw/fused ego-frame road-boundary stability."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

QUERIES = np.asarray([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

def sample(item):
    if not item.get('valid'): return None, None
    sides=[]
    for side in ('left_xy','right_xy'):
        p=np.asarray(item.get(side,[]),dtype=float)
        if p.ndim!=2 or len(p)<2: return None,None
        order=np.argsort(p[:,0]);p=p[order]
        value=np.full(len(QUERIES),np.nan)
        valid=(QUERIES>=p[:,0].min())&(QUERIES<=p[:,0].max())
        value[valid]=np.interp(QUERIES[valid],p[:,0],p[:,1])
        sides.append(value)
    return .5*(sides[0]+sides[1]), sides[1]-sides[0]

def metric(seq):
    centers=[];widths=[]
    for item in seq:
        c,w=sample(item)
        if c is not None: centers.append(c);widths.append(w)
    if len(centers)<2:return {'valid':False}
    c=np.asarray(centers);w=np.asarray(widths)
    dc=np.abs(np.diff(c,axis=0));dw=np.abs(np.diff(w,axis=0))
    return {'valid':True,'center_jitter_m':float(np.nanmedian(dc)),'far_center_jitter_m':float(np.nanmedian(dc[:,QUERIES>=20])),'width_jitter_m':float(np.nanmedian(dw)),'median_width_m':float(np.nanmedian(np.abs(w))),'coverage':float(np.mean(np.isfinite(c)))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    rows=[]
    for line in args.input.read_text().splitlines():
        item=json.loads(line);e=item.get('ego_frame_boundary_fusion') or {}
        raw=metric(e.get('raw_ego_boundaries',[]));fused=metric(e.get('ego_boundaries',[]))
        rows.append({'sample_id':item.get('sample_id'),'raw':raw,'fused':fused,'clipped_fraction':(e.get('diagnostics') or {}).get('clipped_fraction')})
    def agg(key,field):
        v=[r[key].get(field) for r in rows if r[key].get('valid') and r[key].get(field) is not None]
        return float(np.median(v)) if v else None
    fields=['center_jitter_m','far_center_jitter_m','width_jitter_m','median_width_m','coverage']
    result={'protocol':'ego-frame-boundary-stability-v1','num_samples':len(rows),'aggregate':{k:{f:agg(k,f) for f in fields} for k in ('raw','fused')},'rows':rows}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('protocol','num_samples','aggregate')},indent=2))
if __name__=='__main__':main()
