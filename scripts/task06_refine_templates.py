import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import task06_engineering as E

parser=argparse.ArgumentParser();parser.add_argument('--data',default='/prep/data');parser.add_argument('--starter',default='/prep/starter');parser.add_argument('--rounds',type=int,default=2)
a=parser.parse_args();root=Path(a.starter)
frames=[r for r in E.load_frames(a.data,'train',0) if r[0]['kind']=='static']
for iteration in range(a.rounds):
 model=E.Estimator(root/'templates.npz');modelpath=root/'templates.npz';data=np.load(modelpath);ids=data['ids'];groups={int(s):[] for s in ids}
 for metadata,obs,gt in frames:
  model.reset();pose=np.asarray(model.update(**obs));sid=int(pose[3]);groups[sid].append((obs,gt,pose))
 result=[];offsets=[];corrections={}
 for sid in ids:
  rows=groups[int(sid)];shift=[];angular=[]
  for obs,gt,pose in rows:
   c,s=np.cos(pose[2]),np.sin(pose[2]);R=np.array([[c,-s],[s,c]])
   shift.append((gt[:2]-pose[:2])@R);angular.append(float(E.wrap(gt[2]-pose[2])))
  offset=np.mean(shift,axis=0);yaw=float(np.mean(angular));corrections[int(sid)]={'xy_mm':(offset*1000).tolist(),'yaw_deg':float(np.rad2deg(yaw)),'frames':len(rows)}
  masks=[]
  for obs,gt,pose in rows:
   c,s=np.cos(pose[2]),np.sin(pose[2]);R=np.array([[c,-s],[s,c]]);xy=pose[:2]+R@offset;theta=pose[2]+yaw
   c,s=np.cos(theta),np.sin(theta);R=np.array([[c,-s],[s,c]])
   points=E.plane_points(obs);q=(points-xy)@R;masks.append(E.raster_points(q,np.zeros(2),.001,480))
  mask=(np.mean(masks,axis=0)>.5).astype(np.uint8)
  mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));yy,xx=np.nonzero(mask)
  anchor=(np.array([(xx.min()+xx.max())/2,(yy.min()+yy.max())/2])-240)*.001
  transform=np.array([[1.,0.,150-240-anchor[0]/.001],[0.,1.,150-240-anchor[1]/.001]])
  mask=cv2.warpAffine(mask.astype(np.float32),transform,(300,300),flags=cv2.INTER_LINEAR)
  result.append(cv2.GaussianBlur(mask,(3,3),.6));offsets.append(-anchor)
 np.savez_compressed(root/'templates.npz',templates=np.asarray(result),ids=ids,offsets=np.asarray(offsets))
 (root.parent/f'template-refinement-{iteration}.json').write_text(json.dumps(corrections,indent=2)+'\n');print(iteration,corrections,flush=True)
