#!/usr/bin/env python3
"""Make Epona history and generated frames share the rendered camera geometry."""
import argparse, json
from pathlib import Path
import cv2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    root=args.manifest.parent.resolve(); out=root/'history_render_1024x512'; out.mkdir(exist_ok=True)
    # Original NAVSIM calibration is defined on 1920x1080; Epona renders 1024x512.
    sx,sy=1024.0/1920.0,512.0/1080.0
    K=[[1545.0*sx,0.0,960.0*sx],[0.0,1545.0*sy,560.0*sy],[0.0,0.0,1.0]]
    with args.output.open('w') as f:
        for row in rows:
            history=[]
            for i,path in enumerate(row['history_images']):
                target=out/f"{row['sample_index']:06d}_{i:02d}.png"
                image=cv2.imread(str(Path(path)))
                if image is None: raise FileNotFoundError(path)
                image=cv2.resize(image,(1024,512),interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(target),image); history.append(str(target))
            row['history_images']=history
            row['camera_intrinsic']=K
            row['render_geometry']='epona_1024x512_from_navsim_1920x1080'
            f.write(json.dumps(row)+'\n')
    print(json.dumps({'rows':len(rows),'output':str(args.output),'intrinsic':K}))
if __name__=='__main__': main()
