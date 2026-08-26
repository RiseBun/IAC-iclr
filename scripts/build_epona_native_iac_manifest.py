#!/usr/bin/env python3
"""Build generated/control IAC manifests for the Epona NAVSIM batch."""
import argparse, json, pickle
from pathlib import Path
import numpy as np

K = [[1545.0,0.0,960.0],[0.0,1545.0,560.0],[0.0,0.0,1.0]]
T = [[-0.0030311323941387964,-0.019786295321436612,0.9997996373043262,1.6240250126233996],[-0.9999953180155606,-0.00035968662895047944,-0.003038843939251048,-0.0055507164874228345],[0.00041974202478565527,-0.9998041673962867,-0.019785112424517044,1.5331206139432636],[0.0,0.0,0.0,1.0]]

def rel_pose(f, a):
    Ra=np.asarray(a["ego2global"],float); Rf=np.asarray(f["ego2global"],float)
    d=np.asarray(f["ego2global_translation"],float)[:2]-np.asarray(a["ego2global_translation"],float)[:2]
    xy=Ra[:2,:2].T@d; yaw=np.arctan2((Ra[:2,:2].T@Rf[:2,:2])[1,0],(Ra[:2,:2].T@Rf[:2,:2])[0,0])
    return [float(xy[0]),float(xy[1]),float(yaw)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pkl',required=True); ap.add_argument('--sensor-root',required=True); ap.add_argument('--epona-manifest',required=True); ap.add_argument('--output',required=True); ap.add_argument('--real-future',action='store_true'); args=ap.parse_args()
    frames=pickle.load(open(args.pkl,'rb')); gen=json.loads(Path(args.epona_manifest).read_text()); rows=[]
    for i,item in enumerate(gen):
        start=i*5; w=frames[start:start+15]; a=w[9];
        history=[]
        for f in w[6:10]: history.append(str((Path(args.sensor_root)/f['cams']['CAM_F0']['data_path']).resolve()))
        f=w[10]; native_future=str((Path(args.sensor_root)/f['cams']['CAM_F0']['data_path']).resolve())
        pose=rel_pose(f,a); speed=float(f.get('ego_dynamic_state',[0])[0])
        future=native_future if args.real_future else str(Path(item['future_image']).resolve())
        rows.append({'protocol':'native-realized-state-v1','record_type':'epona_generated_future','dataset':'navsim','source_key':f'epona:{a.get("scene_token","")}:{a.get("frame_idx",0)}:{i}','scene_name':str(a.get('scene_token','')),'history_images':history,'future_images':[future],'future_images_source':'native_dataset' if args.real_future else 'wam_generated','history_ego_state':[[0,0,0,0,0]]*4,'realized_future_ego_state':[[pose[0],pose[1],pose[2],speed,0.0]],'future_times_s':[0.5],'trajectory':[pose],'trajectory_source':'native_navsim_ego_state','camera_intrinsic':K,'camera_distortion':[],'camera_to_ego':T,'task_success':None,'task_success_source':None,'wam_model_id':'epona_nuplan','wam_generation_status':'complete'})
    Path(args.output).write_text('\n'.join(json.dumps(r) for r in rows)+'\n'); print(json.dumps({'output':args.output,'rows':len(rows)}))
if __name__=='__main__': main()
