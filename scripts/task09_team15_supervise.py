"""Enforce the run's absolute deadline and clean only its owned processes and containers."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def cleanup(root, process):
    rows = subprocess.run(['ps','-axo','pid=,ppid='],capture_output=True,text=True,timeout=3).stdout.splitlines()
    parents = {int(row.split()[0]):int(row.split()[1]) for row in rows if len(row.split()) == 2}
    owned = {process.pid}
    while True:
        children = {pid for pid,parent in parents.items() if parent in owned}
        if children <= owned:
            break
        owned.update(children)
    for pid in owned-{process.pid}:
        try:
            os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=2)
    for pid in owned-{process.pid}:
        try:
            os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
    ids = subprocess.run(['docker','ps','-q'],capture_output=True,text=True,timeout=3).stdout.split()
    removed = []
    if ids:
        rows = json.loads(subprocess.run(['docker','inspect',*ids],capture_output=True,text=True,check=True,timeout=3).stdout)
        for row in rows:
            if any(Path(m['Source']).is_relative_to(root) for m in row.get('Mounts',[]) if m.get('Source')):
                subprocess.run(['docker','rm','-f',row['Id']],capture_output=True,timeout=3)
                removed.append(row['Name'])
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--script',type=Path,default=Path(__file__).with_name('task09_team15.py'))
    args = parser.parse_args()
    root = args.output.resolve()
    log = root.parent/(root.name+'.log')
    with log.open('w') as out:
        process = subprocess.Popen([str(Path('.venv/bin/python').absolute()),'-B',
            str(args.script.resolve()),'--output',str(root)],
            stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.time()+930
        while process.poll() is None:
            if (root/'process.json').is_file():
                deadline = json.loads((root/'process.json').read_text())['total_deadline_at']-4
            if time.time() >= deadline:
                removed = cleanup(root,process)
                root.mkdir(parents=True,exist_ok=True)
                (root/'supervisor.json').write_text(json.dumps({'status':'deadline_enforced','containers_removed':removed,'at':time.time()},indent=2)+'\n')
                return
            time.sleep(.25)
    if process.returncode:
        removed = cleanup(root,process)
    else:
        removed = []
    if root.exists():
        (root/'supervisor.json').write_text(json.dumps({'status':'exited','returncode':process.returncode,'containers_removed':removed,'at':time.time()},indent=2)+'\n')


if __name__ == '__main__':main()
