"""Run the approved shortened pilots sequentially and record results as they land."""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).absolute().parents[1]
SETUP=ROOT/'outputs/short-pilots-20261102'
PYTHON=str(ROOT/'.venv/bin/python')
STOP=datetime.fromisoformat('2026-11-02T15:00:00-08:00').timestamp()-30
CUTOFF=datetime.fromisoformat('2026-11-02T14:00:00-08:00').timestamp()


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def message(value):
    subprocess.run(['agent-message','claude','investigator',value],capture_output=True,text=True,timeout=20)


def output(record):
    args=record['command']
    return ROOT/args[args.index('--output')+1]


def running(record):
    result=subprocess.run(['ps','-p',str(record['pid']),'-o','stat=,args='],capture_output=True,text=True)
    value=result.stdout.strip()
    return bool(value and not value.startswith('Z') and
                (str(output(record)) in value or str(output(record).relative_to(ROOT)) in value))


def launch(name,command):
    with (SETUP/f'{name}.log').open('x') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                           env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    record={'pid':p.pid,'command':command,'started_at':time.time()}
    write(SETUP/f'{name}-process.json',record)
    return record


def wait_phase(name,records,barrier=None):
    reported=set()
    while time.time()<STOP:
        live=[running(r) for r in records]
        if barrier and not barrier.exists():
            if all((output(r)/'selection-frozen.json').exists() or not active for r,active in zip(records,live)):
                write(barrier,{'released_at':time.time(),'all_conditions_frozen_or_stopped':True})
        for index,(r,active) in enumerate(zip(records,live)):
            if index in reported:continue
            summary=read(output(r)/'summary.json')
            if summary or not active:
                reported.add(index)
                row={'task':name,'output':str(output(r)),'summary':summary,'process_running':active}
                completed.append(row)
                write(SETUP/'results.json',completed)
                if summary:
                    api=summary.get('api',{})
                    value=f"{name} result {output(r).name}: status={summary.get('status')}, score={summary.get('score')}, API estimate=${api.get('estimated_cost_usd',0):.4f}, elapsed={summary.get('total_seconds',summary.get('total_seconds_including_starter_probe',0)):.1f}s. Shortened/exploratory."
                else:value=f'{name} FAILED before summary: {output(r).name}; see operator log in outputs/short-pilots-20261102/. No fabricated score.'
                message(value)
        write(SETUP/'sequence-progress.json',{'phase':name,'running_conditions':sum(live),
            'completed_conditions':len(reported),'at':time.time(),'total_estimated_spend_usd':spend(),
            'combined_ceiling_usd':26})
        if not any(live):return
        time.sleep(5)


def spend():
    total=0
    seen=set()
    for path in SETUP.glob('*-process.json'):
        record=read(path)
        if not record or '--output' not in record.get('command',[]):continue
        root=output(record)
        if root in seen:continue
        seen.add(root)
        lanes=list(root.glob('**/shared-api/account-*/api-summary.json'))
        if lanes:total+=sum((read(p) or {}).get('estimated_cost_usd',0) for p in lanes)
        else:total+=(read(root/'summary.json') or {}).get('api',{}).get('estimated_cost_usd',0)
    return total


completed=[]


def main():
    first=[read(SETUP/'task08-swarm-02-process.json'),read(SETUP/'task08-astra-process.json')]
    if not all(first):raise ValueError('Task08 launch records missing')
    wait_phase('Task08',first)
    if time.time()>=STOP:return
    barrier=SETUP/'task09-final-release.json'
    common=['--hard-stop-at','2026-11-02T15:00:00-08:00','--final-barrier',str(barrier)]
    ds=launch('task09-swarm',[PYTHON,'-B','scripts/task09_swarm_short.py','--output',
        'outputs/task09-swarm-06-short-01','--tests-per-worker','3','--max-cost','10',*common])
    astra=launch('task09-astra',[PYTHON,'-B','scripts/short_pilot_astra.py','--task','9','--output',
        'outputs/task09-astra-short-01','--tests','3','--simulation-lock','/tmp/kimirax-short-task09-sim.lock',*common])
    message('Task09 shortened pilots STARTED: six DeepSeek workers pooled across two authorized accounts, fresh Astra,900s each,3public probes/agent,sharedserial4CPU/4GB simulation,$10 combined API estimate guard. Private final release waits for both selections frozen or agents stopped.')
    wait_phase('Task09',[ds,astra],barrier)
    while time.time()<min(STOP,CUTOFF):
        plan=read(SETUP/'task07-command.json')
        if plan and plan.get('ready'):
            record=launch('task07',plan['command'])
            message('Task07 shortened pilot STARTED before14:00 cutoff; hardstop15:00, $10combined API guard. Full private evaluation may not fit; incomplete final will not be represented as a score.')
            wait_phase('Task07',[record])
            break
        time.sleep(5)
    else:
        write(SETUP/'task07-skipped.json',{'reason':'Not running before14:00PT cutoff','at':time.time()})
        message('Task07 DROPPED: not running before14:00PT cutoff. No late model launch.')
    final={'status':'completed_or_deadline_reached','at':time.time(),'total_estimated_api_spend_usd':spend(),
           'combined_ceiling_usd':26,'results':completed,'subscription_cost_not_marginal_api_billing':True}
    write(SETUP/'summary.json',final)
    message(f'Approved shortened pilot sequence finished/deadline reached. Recorded API estimate total=${spend():.4f}; see outputs/short-pilots-20261102/summary.json for results/failures. No further model launches authorized by this sequence.')


if __name__=='__main__':main()
