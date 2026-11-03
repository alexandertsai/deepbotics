"""Deadline cleanup restricted to explicitly registered shortened pilot processes."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def processes():
    result=subprocess.run(['ps','-axo','pid=,ppid=,command='],capture_output=True,text=True,check=True)
    rows={}
    for line in result.stdout.splitlines():
        fields=line.strip().split(None,2)
        if len(fields)==3:rows[int(fields[0])]=(int(fields[1]),fields[2])
    return rows


def cleanup(setup):
    table=processes()
    roots=[]
    outputs=[]
    for path in setup.glob('*-process.json'):
        value=json.loads(path.read_text())
        pid=value.get('pid')
        command=value.get('command',[])
        if '--output' not in command:continue
        output=Path(command[command.index('--output')+1]).absolute()
        outputs.append(output)
        if pid in table and str(output) in table[pid][1] or pid in table and str(output.relative_to(Path.cwd())) in table[pid][1]:
            roots.append(pid)
    owned=set(roots)
    while True:
        children={pid for pid,(parent,_) in table.items() if parent in owned}
        if children<=owned:break
        owned.update(children)
    owned.discard(os.getpid())
    for pid in sorted(owned,reverse=True):
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    time.sleep(5)
    current=processes()
    for pid in owned:
        if pid in current and current[pid][1]==table[pid][1]:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
    ids=subprocess.run(['docker','ps','-q'],capture_output=True,text=True,timeout=10).stdout.split()
    removed=[]
    if ids:
        inspected=subprocess.run(['docker','inspect',*ids],capture_output=True,text=True,timeout=15)
        if inspected.returncode==0:
            for item in json.loads(inspected.stdout):
                sources=[Path(m['Source']) for m in item.get('Mounts',[]) if m.get('Source')]
                if any(source==root or root in source.parents for source in sources for root in outputs):
                    subprocess.run(['docker','rm','-f',item['Id']],capture_output=True,timeout=15)
                    removed.append(item['Name'])
    return {'registered_roots_stopped':roots,'processes_signaled':len(owned),'containers_removed':removed}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup',type=Path,required=True)
    parser.add_argument('--stop-at',default='2026-11-02T15:00:00-08:00')
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.stop_at).timestamp()-30
    while time.time()<deadline:time.sleep(min(30,deadline-time.time()))
    report=cleanup(args.setup)
    report.update(at=time.time(),deadline=args.stop_at)
    (args.setup/'deadline-cleanup.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
