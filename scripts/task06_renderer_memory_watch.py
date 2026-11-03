import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time

CODE = r'''
import json,os,signal,subprocess,time
services=[];busy=False
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 try:
  status=open('/proc/'+pid+'/status').read()
  cmd=open('/proc/'+pid+'/cmdline','rb').read().replace(b'\0',b' ')
  uid=int(next(x for x in status.splitlines() if x.startswith('Uid:')).split()[1])
  if uid==1000 and b'python' in cmd:busy=True
  if b'harness.training_service' in cmd and b'python -P -m' in cmd:
   rss=int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1])
   services.append((int(pid),rss))
 except (OSError,StopIteration):pass
marker='/run/rlebench/memory-idle'
should_reclaim=not busy and services and max(x[1] for x in services)>250000
restarted=False
if should_reclaim:
 if os.path.exists(marker) and time.time()-os.stat(marker).st_mtime>3:
  for pid,rss in services:os.kill(pid,signal.SIGTERM)
  services=[]
 elif not os.path.exists(marker):open(marker,'w').close()
else:
 try:os.unlink(marker)
 except FileNotFoundError:pass
if not services:
 env=os.environ.copy();env['PYTHONPATH']='/opt/private'
 out=open('/tmp/training-service-recovery.log','a')
 subprocess.Popen(['python','-P','-m','harness.training_service','--socket','/run/rlebench/task06-training.sock'],env=env,stdout=out,stderr=out,start_new_session=True)
 restarted=True
 try:os.unlink(marker)
 except FileNotFoundError:pass
print(json.dumps({'busy':busy,'service_rss_kib':[r for _,r in services],'restarted':restarted}))
'''


def main():
    output=Path('outputs/task06-task08-parallel-15m-20261102-01/renderer-memory-recovery.jsonl')
    deadline=time.time()+1000
    while time.time()<deadline:
        names=[n for n in subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
            if n.startswith('paired-b6-') or n=='paired-repaired-astra-6']
        if not names:break
        def check(name):
            r=subprocess.run(['docker','exec','-i',name,'python3','-'],input=CODE,text=True,capture_output=True,timeout=8)
            if r.returncode:return
            value=json.loads(r.stdout)
            if value['restarted']:
                with output.open('a') as stream:stream.write(json.dumps({'at':time.time(),'container':name,**value})+'\n')
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(check,names))
        time.sleep(5)


if __name__=='__main__':main()
