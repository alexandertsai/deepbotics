import json
from pathlib import Path
import time
import numpy as np
from harness import episodes
root=Path('/prep/data');manifest=json.loads((root/'manifest.json').read_text());start=time.monotonic()
for i in range(180):
 seed=100011+1000*i;frame=episodes.single_frame(seed);name=f'static-{seed}.npz';np.savez_compressed(root/name,**frame.obs,gt=np.asarray(frame.gt))
 manifest['frames'].append({'file':name,'seed':seed,'frame':0,'episode':f'static-{seed}','kind':'static','split':'train'})
 if i%30==0:print('extra',i,round(time.monotonic()-start,1),flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('done',len(manifest['frames']),round(time.monotonic()-start,1),flush=True)
