import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from harness import episodes
import task06_engineering as E

parser=argparse.ArgumentParser();parser.add_argument('--output',default='/prep/data');parser.add_argument('--count',type=int,default=72)
args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
manifest={'source':'Original public episodes RPC; noisy pose labels, no shape IDs or meshes.','frames':[]};start=time.monotonic()
seeds=[11,23,37]+[11+1000*i for i in range(1,args.count-2)]
for i,seed in enumerate(seeds):
 frame=episodes.single_frame(seed);name=f'static-{seed}.npz'
 np.savez_compressed(root/name,**frame.obs,gt=np.asarray(frame.gt))
 manifest['frames'].append({'file':name,'seed':seed,'frame':0,'episode':f'static-{seed}','kind':'static','split':'validation' if i%4==3 else 'train'})
 if i%12==0:print('static',i,round(time.monotonic()-start,1),flush=True)
for seed in (23,37,2011,3011):
 ep=episodes.push_episode(seed,max_frames=24)
 for i,frame in enumerate(ep.frames):
  name=f'push-{seed}-{i:03d}.npz';np.savez_compressed(root/name,**frame.obs,gt=np.asarray(frame.gt))
  manifest['frames'].append({'file':name,'seed':seed,'frame':i,'episode':f'push-{seed}','kind':'push','split':'validation' if seed in (37,3011) else 'train'})
 print('push',seed,round(time.monotonic()-start,1),flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('complete',len(manifest['frames']),round(time.monotonic()-start,1),flush=True)
