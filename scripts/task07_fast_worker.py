"""Fast interface checks using synthetic observations; never a task score."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0,'/workspace')
import numpy as np
from harness import sandbox,spec


def main():
    reports = []
    for bundle in sorted(Path('/cases').iterdir()):
        if not (bundle/'policy.py').is_file():
            continue
        started = time.monotonic()
        row = {'case':bundle.name,'kind':'synthetic_interface_check_not_task_score'}
        try:
            with sandbox.PolicyProcess(str(bundle),act_timeout_s=5,startup_timeout_s=10) as policy:
                policy.check_loaded()
                policy.reset(spec.cell_spec(),0)
                failures,durations = 0,[]
                for index,t in enumerate((0.,.05,.1,.5,1.,2.,3.,5.)):
                    obs = {'qpos':np.array(spec.ARM_HOME),'qvel':np.zeros(7),'tau':np.zeros(7),
                        'ft':np.array([0.,0.,6.5727,0.,0.,0.]),'mag_on':0.,'t':t}
                    if index%2 == 0:
                        obs.update(overhead_rgb=np.full((480,640,3),120,dtype=np.uint8),
                            overhead_depth=np.full((480,640),.75,dtype=np.float32),
                            wrist_rgb=np.full((240,320,3),120,dtype=np.uint8),
                            wrist_depth=np.full((240,320),.5,dtype=np.float32),wrist_T_world_cam=np.eye(4))
                    step = time.monotonic()
                    ctrl = policy.act(obs)
                    durations.append(time.monotonic()-step)
                    if ctrl is None or np.shape(ctrl)!=(8,) or not np.isfinite(ctrl).all():
                        failures += 1
                row.update(passed=not failures and not policy.dead,failures=failures,
                    sample_calls=len(durations),max_act_wall_seconds=max(durations),
                    total_policy_wall_seconds=policy.policy_wall_s,stderr=policy.stderr_tail(3000))
        except Exception as error:
            row.update(passed=False,error=f'{type(error).__name__}: {error}')
        row['wall_seconds'] = time.monotonic()-started
        reports.append(row)
    Path('/logs/contracts.json').write_text(json.dumps({'kind':'synthetic_interface_checks_not_benchmark',
        'no_private_data':True,'reports':reports},indent=2)+'\n')


if __name__ == '__main__':main()
