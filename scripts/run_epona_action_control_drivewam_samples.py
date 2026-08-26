#!/usr/bin/env python3
"""Epona action-control probe on DriveWAM's self-contained native samples.

The DriveWAM sample pickle stores RGB arrays, so this avoids depending on a
separate NAVSIM sensor mount while preserving the same native history/action
contract. Epona receives explicit next-frame pose/yaw controls before video
generation for logged/left/right branches.
"""
import argparse, json, os, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch
from einops import rearrange

EPONA_ROOT = os.environ.get("EPONA_ROOT", str(Path.cwd() / "third_party" / "Epona"))
sys.path.insert(0, EPONA_ROOT)
CAMERA_INTRINSIC = [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]]
CAMERA_TO_EGO = [[-0.0030311323941387964,-0.019786295321436612,0.9997996373043262,1.6240250126233996],[-0.9999953180155606,-0.00035968662895047944,-0.003038843939251048,-0.0055507164874228345],[0.00041974202478565527,-0.9998041673962867,-0.019785112424517044,1.5331206139432636],[0.0,0.0,0.0,1.0]]

def cfg_from(args):
    from types import SimpleNamespace
    ns={}; exec(Path(f"{EPONA_ROOT}/configs/dit_config_dcae_nuplan.py").read_text(), ns)
    c=SimpleNamespace(**{k:v for k,v in ns.items() if not k.startswith('__')})
    c.batch_size=1; c.vae_ckpt=args.vae; c.resume_path=args.checkpoint
    c.num_sampling_steps=args.sampling_steps; c.condition_frames=10
    c.image_size=(512,1024); c.temporal_patch_size=6; c.test_video_frames=4
    c.device=args.device
    return c

def pose_mats(points):
    points=np.asarray(points,dtype=np.float32); out=np.tile(np.eye(4,dtype=np.float32),(len(points),1,1))
    out[:,:2,3]=points[:,:2]; c=np.cos(points[:,2]); s=np.sin(points[:,2])
    out[:,0,0]=c; out[:,0,1]=-s; out[:,1,0]=s; out[:,1,1]=c
    return out

def controls(native, mode):
    x=np.asarray(native,dtype=np.float32).copy()
    if mode=='left': x[:,1]-=0.30; x[:,2]+=np.deg2rad(3.0)
    elif mode=='right': x[:,1]+=0.30; x[:,2]-=np.deg2rad(3.0)
    elif mode!='logged': raise ValueError(mode)
    return x

def cumulative_trajectory(relative):
    """Convert Epona's relative [dx,dy,yaw_deg] controls to ego-frame [x,y,yaw_rad]."""
    pose = np.zeros(3, dtype=np.float64); result = []
    for dx, dy, yaw_deg in np.asarray(relative, dtype=np.float64):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pose[:2] += [c * dx - s * dy, s * dx + c * dy]
        pose[2] += np.deg2rad(yaw_deg)
        result.append(pose.copy().tolist())
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--vae',required=True); ap.add_argument('--output',required=True); ap.add_argument('--num-samples',type=int,default=3); ap.add_argument('--future-steps',type=int,default=4); ap.add_argument('--sampling-steps',type=int,default=4); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--common-random',action='store_true'); args=ap.parse_args()
    from models.model import TrainTransformersDiT
    from models.modules.tokenizer import VAETokenizer
    from utils.preprocess import get_rel_pose
    c=cfg_from(args); torch.cuda.set_device(torch.device(args.device))
    model=TrainTransformersDiT(c,load_path=args.checkpoint,local_rank=0,condition_frames=10).eval(); tok=VAETokenizer(c,0)
    paths=sorted(Path(args.data).glob('sample_*.pkl'))[:args.num_samples]; out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True); rows=[]
    for si,p in enumerate(paths):
        s=pickle.load(open(p,'rb')); imgs=np.asarray(s['images'])
        if imgs.shape[0]<10: raise ValueError(f'{p}: need >=10 images')
        # Four history poses plus six interpolated/extrapolated poses form the
        # ten-frame Epona context; the next four are explicit interventions.
        hist=np.asarray(s['history_poses'],dtype=np.float32)
        fut=np.asarray([[f['pose'][0],f['pose'][1],f['pose'][2]] for f in s['future_trajectory']],dtype=np.float32)
        base=np.concatenate([hist,fut],axis=0)
        # DriveWAM's compact pickle stores four history poses and eight future
        # poses, while Epona's native context is ten frames. Extrapolate only
        # the missing metadata points from the last native delta; RGB history
        # remains exactly the stored frames and this is recorded as a probe.
        needed = 10 + args.future_steps
        while len(base) < needed:
            delta = base[-1] - base[-2]
            base = np.concatenate([base, (base[-1] + delta)[None]], axis=0)
        base=base[:needed]
        if len(base)<10+args.future_steps: raise ValueError(f'{p}: insufficient pose points')
        mats=torch.from_numpy(pose_mats(base)).unsqueeze(0).to(args.device)
        rel_pose,rel_yaw=get_rel_pose(mats); hist_pose=rel_pose[:,:10]; hist_yaw=rel_yaw[:,:10]
        history_paths=[]
        history_dir=out/f'sample_{si:06d}'/'history'; history_dir.mkdir(parents=True,exist_ok=True)
        for hi,raw in enumerate(imgs[:10]):
            hp=history_dir/f'frame_{hi:02d}.png'; cv2.imwrite(str(hp),cv2.cvtColor(raw,cv2.COLOR_RGB2BGR)); history_paths.append(str(hp))
        rgb=torch.from_numpy(imgs[:10]).permute(0,3,1,2).float()/255.0; rgb=torch.nn.functional.interpolate(rgb,size=(512,1024),mode='bilinear',align_corners=False); rgb=((rgb-.5)*2).unsqueeze(0).to(args.device)
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            hlat=tok.encode_to_z(rgb)
            native_pose = rel_pose[0,10:10+args.future_steps].detach().cpu().numpy()
            native_yaw = rel_yaw[0,10:10+args.future_steps].detach().cpu().numpy()
            native = np.concatenate([native_pose, native_yaw], axis=1)
        for mode in ('logged','left','right'):
                # Use common random numbers across counterfactual branches so
                # image differences are attributable to controls, not DiT noise.
                if args.common_random:
                    torch.manual_seed(100000 + si)
                d=out/f'sample_{si:06d}'/mode; d.mkdir(parents=True,exist_ok=True); ctl=controls(native,mode); poses=hist_pose.clone(); yaws=hist_yaw.clone(); lat=hlat.clone(); fs=[]
                for t in range(args.future_steps):
                    poses=torch.cat([poses,torch.from_numpy(ctl[t,:2]).to(args.device).view(1,1,2)],1); yaws=torch.cat([yaws,torch.from_numpy(np.asarray([ctl[t,2]],np.float32)).to(args.device).view(1,1,1)],1)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        z=model.generate_gt_pose_gt_yaw(lat,poses[:,-11:],yaws[:,-11:])
                    z=rearrange(z,'(b F) h w c -> b F h w c',F=1); im=tok.z_to_image(z[:,0]).float().cpu()[0]; arr=(im.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8); fp=d/f'future_{t+1:02d}.png'; cv2.imwrite(str(fp),cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)); fs.append(str(fp)); zseq=rearrange(z,'b 1 h w c -> b 1 (h w) c'); lat=torch.cat([lat[:,1:],zseq],1)
                source_key=s.get('metadata',{}).get('source_key',f'epona_drivewam:{si}')
                rows.append({'sample_index':si,'source_sample':str(p),'source_key':source_key,'counterfactual_group_id':source_key,'branch_id':f'{source_key}::branch={mode}','branch_mode':mode,'history_images':history_paths,'future_images':fs,'future_images_source':'epona_generated','action_injection_verified':True,'intervention_variant':'epona_generate_gt_pose_gt_yaw','native_control_deltas':ctl.tolist(),'action_trajectory':cumulative_trajectory(ctl),'action_trajectory_representation':'cumulative_ego_se2_radians','future_times_s':[0.2*(i+1) for i in range(args.future_steps)],'camera_intrinsic':CAMERA_INTRINSIC,'camera_distortion':[],'camera_to_ego':CAMERA_TO_EGO})
                del lat; torch.cuda.empty_cache()
        print(json.dumps(rows[-1])[:1000],flush=True)
    (out/'manifest.json').write_text(json.dumps(rows,indent=2)); print(json.dumps({'output':str(out),'rows':len(rows),'groups':len(rows)//3},indent=2))
if __name__=='__main__': main()
