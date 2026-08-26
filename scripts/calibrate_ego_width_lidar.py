#!/usr/bin/env python3
"""Calibrate ego-frame boundary width against matched NAVSIM LiDAR oracle."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from iac_new.geometry import scale_intrinsics
from iac_new.road_structure import boundary_pixels_to_ego

def widths(item):
    if not item.get('valid'): return np.empty(0)
    left=np.asarray(item.get('left_xy',[]),float); right=np.asarray(item.get('right_xy',[]),float)
    if len(left)<2 or len(right)<2:return np.empty(0)
    n=min(len(left),len(right)); return np.abs(right[:n,1]-left[:n,1])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--oracle',type=Path,required=True);ap.add_argument('--state-manifest',type=Path,required=True);ap.add_argument('--fusion',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--applied-width-shrink',type=float,default=1.0);args=ap.parse_args()
    states={}
    for line in args.state_manifest.read_text().splitlines():
        x=json.loads(line);states[x.get('metadata',{}).get('source_key')]=x
    fused={}
    for line in args.fusion.read_text().splitlines():
        x=json.loads(line);fused[x.get('sample_id')]=x.get('ego_frame_boundary_fusion') or {}
    rows=[]
    for line in args.oracle.read_text().splitlines():
        o=json.loads(line); key=o.get('source_key'); s=states.get(key)
        if s is None:continue
        anchor=s['history_frame_paths'][-1]; image=cv2.imread(anchor,cv2.IMREAD_COLOR)
        if image is None:continue
        K=scale_intrinsics(np.asarray(s['intrinsics'],float),(image.shape[1],image.shape[0]),(512,288))
        b=boundary_pixels_to_ego({'valid':True,'rows':o['boundary_descriptor']['rows'],'left_x':o['boundary_descriptor']['left_x'],'right_x':o['boundary_descriptor']['right_x'],'image_height':288},K,np.asarray(s['camera_to_ego'],float))
        ow=widths(b); f=fused.get(key,{})
        rw=widths({'valid':True,'left_xy':f.get('raw_ego_boundaries',[{}])[0].get('left_xy',[]) if f.get('raw_ego_boundaries') else [],'right_xy':f.get('raw_ego_boundaries',[{}])[0].get('right_xy',[]) if f.get('raw_ego_boundaries') else []})
        fw=[]
        for z in f.get('ego_boundaries',[]):fw.extend((widths(z)/max(float(args.applied_width_shrink),1e-6)).tolist())
        if len(ow):rows.append({'source_key':key,'oracle_width_median':float(np.median(ow)),'fused_width_median':float(np.median(fw)) if fw else None,'raw_width_median':float(np.median(rw)) if len(rw) else None,'ratio_oracle_fused':float(np.median(ow)/np.median(fw)) if fw and np.median(fw)>1e-6 else None})
    ratios=[r['ratio_oracle_fused'] for r in rows if r['ratio_oracle_fused'] is not None]
    result={'protocol':'ego-width-lidar-calibration-v1','num_oracle_state_matches':len(rows),'num_fused_matches':len(ratios),'median_oracle_width_m':float(np.median([r['oracle_width_median'] for r in rows])) if rows else None,'median_fused_width_m':float(np.median([r['fused_width_median'] for r in rows if r['fused_width_median'] is not None])) if ratios else None,'recommended_width_shrink':float(np.clip(np.median(ratios),0.6,1.0)) if len(ratios)>=5 else None,'calibration_status':'provisional' if len(ratios)<20 else 'usable','rows':rows}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('protocol','num_oracle_state_matches','num_fused_matches','median_oracle_width_m','median_fused_width_m','recommended_width_shrink','calibration_status')},indent=2))
if __name__=='__main__':main()
