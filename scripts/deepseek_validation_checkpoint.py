"""Queue edited branch snapshots at periodic development checkpoints."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
import sys
import time


def fingerprint(path):
    return {p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.suffix in ('.py','.xml','.npz')}


def main():
    root=Path(sys.argv[1]).resolve()
    process=json.loads((root/'process.json').read_text())
    submitted={}
    log=root/'harness-checkpoints.jsonl'
    while time.time()<process['development_deadline_at']-75:
        jobs=[]
        for task in (6,8):
            baseline=fingerprint(root/f'starting-candidates/task{task:02d}')
            for branch in (root/f'task{task:02d}/deepseek').iterdir():
                candidate=branch/'candidate';config=branch/'agent-home/mcp.json'
                if not candidate.exists() or not config.exists() or (branch/'frozen').exists():continue
                current=fingerprint(candidate)
                if current==baseline or submitted.get(str(branch))==current:continue
                analyses=list((branch/'analysis').glob('*')) if (branch/'analysis').exists() else []
                if any(not (p/'result.json').exists() for p in analyses):continue
                jobs.append((branch,config,current))
        def queue(job):
            branch,config,current=job
            record={'at':time.time(),'actor':'automatic_harness_checkpoint','branch':branch.name}
            try:
                data=json.loads(config.read_text());server=next(iter(data['mcpServers'].values()))
                request={'jsonrpc':'2.0','id':'harness-checkpoint','method':'tools/call','params':{
                    'name':'test_candidate','arguments':{'hypothesis':'Automatic harness checkpoint: validate current edited exports without interrupting local experiments.'}}}
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(12);client.connect(server['args'][-1]);client.sendall(json.dumps(request).encode()+b'\n')
                    line=client.makefile('rb').readline(2*1024**2)
                reply=json.loads(line)['result'];record['reply']=reply
                if not reply.get('isError'):
                    value=json.loads(reply['content'][0]['text'])
                    if value.get('queued') or value.get('cached'):submitted[str(branch)]=current
            except Exception as error:record['error']=f'{type(error).__name__}: {error}'
            return record
        with ThreadPoolExecutor(max_workers=4) as executor:
            for record in executor.map(queue,jobs):
                with log.open('a') as output:output.write(json.dumps(record)+'\n')
        time.sleep(25)


if __name__=='__main__':main()
