"""Read-only model throughput accounting for the shortened pilots."""

import json
from pathlib import Path
import time

import short_pilot_sequence as S

RUNS={'task08':'task08-swarm-06-short-02','task09':'task09-swarm-06-short-01',
      'task07':'task07-short-01/deepseek'}


def snapshot():
    report={'at':time.time(),'runs':{},'turn_definition':'Distinct successful logical provider responses; admissions include queued or unfinished requests.'}
    for name,folder in RUNS.items():
        root=S.ROOT/'outputs'/folder
        workers={}
        for path in root.glob('shared-api/account-*/api-summary.json'):
            lane=S.read(path) or {}
            for worker,logical in lane.get('worker_logical_requests',{}).items():
                workers[worker]={'logical_requests_admitted':logical,
                    'provider_attempts_admitted':lane.get('worker_requests',{}).get(worker,0),
                    'successful_logical_requests':set(),'completed_provider_attempts':0,
                    'failed_provider_attempts':0,'provider_queue_seconds_sum':0.,
                    'pacing_wait_seconds_sum':0.,'provider_latency_seconds_sum':0.,
                    'reported_tokens':0,'estimated_api_cost_usd':0.,'account_lane':path.parent.name,
                    'api_slots_per_account':lane.get('max_concurrent')}
        for path in root.glob('shared-api/account-*/api/metrics-*.json'):
            row=S.read(path)
            if not row or row.get('worker_id') not in workers:continue
            worker=workers[row['worker_id']]
            worker['completed_provider_attempts']+=1
            if row.get('status_code')==200:
                worker['successful_logical_requests'].add((path.parents[1].name,row.get('logical_request',row.get('call'))))
            else:worker['failed_provider_attempts']+=1
            for target,source in [('provider_queue_seconds_sum','queue_seconds'),
                    ('pacing_wait_seconds_sum','pacing_wait_seconds'),('provider_latency_seconds_sum','provider_latency_seconds'),
                    ('reported_tokens','reported_tokens'),('estimated_api_cost_usd','estimated_cost_usd')]:
                worker[target]+=row.get(source,0) or 0
        ledger=S.read(root/'probe-ledger.json') or []
        for worker,row in workers.items():
            row['successful_model_turns']=len(row.pop('successful_logical_requests'))
            probes=[p for p in ledger if p.get('worker')==worker]
            row['public_probes_measured']=sum(p.get('status')=='measured' for p in probes)
            row['public_probes_queued_or_running']=sum(p.get('status') in ('queued','running') for p in probes)
            row['public_probes_failed']=sum(p.get('status') in ('error','time_exhausted') for p in probes)
            summary=S.read(root/worker/'agent-summary.json') or {}
            row['tool_calls']=summary.get('tool_calls',len(S.read(root/worker/'tools.json') or []))
            row['stop_reason']=summary.get('stop_reason')
        waits=[]
        for filename in ('simulation-wait.jsonl','simulation-waits.jsonl'):
            path=root/filename
            if path.exists():
                for line in path.read_text().splitlines():
                    try:value=json.loads(line)
                    except ValueError:continue
                    if value.get('event') in ('released','acquisition_failed'):waits.append(value)
        report['runs'][name]={'workers':workers,'successful_turn_range':
            [min((w['successful_model_turns'] for w in workers.values()),default=0),
             max((w['successful_model_turns'] for w in workers.values()),default=0)],
            'recorded_simulator_queue_seconds_sum':sum(w.get('queued_wait_seconds',0) for w in waits),
            'recorded_memory_wait_seconds_sum':sum(w.get('memory_wait_seconds',0) for w in waits),
            'waits_count_toward_fixed_development_window':True,
            'task08_legacy_simulator_wait_log_unavailable':name=='task08',
            'task08_comparison_caveat':'Astra final verification shared the simulator during swarm development' if name=='task08' else None}
    return report


def main():
    while True:
        report=snapshot()
        S.write(S.SETUP/'throughput-per-worker.json',report)
        progress=S.read(S.SETUP/'parallel-progress.json') or {}
        active=progress.get('active',{})
        if time.time()>=S.STOP+20 or active and not any(active.values()):break
        time.sleep(10)
    S.message('Final per-worker throughput recorded: outputs/short-pilots-20261102/throughput-per-worker.json. Successful model responses, admitted logical requests, retries/errors, probes and queue/pacing waits are separate;09/07 remained1slot/account without restart.')


if __name__=='__main__':main()
