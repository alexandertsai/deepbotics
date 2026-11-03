"""Bounded public preparation checks with no model calls and no private verifier."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import task07_adapter as Q
import task08_pilot as P


def check(cases,output,timeout=30):
    cases,output = cases.resolve(),output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    name = 'task07-contract-'+hashlib.sha256(str(output).encode()).hexdigest()[:12]
    command = ['docker','run','--rm','--name',name,'--network','none','--read-only','--cpus','2',
        '--memory','1g','--pids-limit','128','--tmpfs','/tmp:rw,nosuid,size=512m',
        '--security-opt','no-new-privileges','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1',
        '-e','MUJOCO_GL=disable','-e','PYTHONDONTWRITEBYTECODE=1','-v',f'{cases}:/cases:ro',
        '-v',f'{output}:/logs','-v',f'{P.ROOT}/scripts/task07_fast_worker.py:/pilot/check.py:ro',
        Q.AGENT_IMAGE,'python3','/pilot/check.py']
    started = time.monotonic()
    try:
        completed = P.checked(command,timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=5)
        raise TimeoutError('Fast contract check exceeded its deadline; owned container removed') from None
    (output/'container.log').write_text(completed.stdout+completed.stderr)
    result = json.loads((output/'contracts.json').read_text())
    result['total_wall_seconds'] = time.monotonic()-started
    P.write_json(output/'summary.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=30)
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error('Preparation checks must stop within 60 seconds')
    print(json.dumps(check(args.cases,args.output,args.timeout),indent=2))


if __name__ == '__main__':main()
