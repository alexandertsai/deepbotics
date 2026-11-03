"""Matched five-minute Task08 comparison with shared parallel simulation capacity."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time

import task08_pilot as P
import task08_swarm_short as W
from task08_race_sessions import Coordinator, Session, DS_ROLES, ASTRA_ROLE
from task08_sim_pool import SimulationPool
from short_pilot_pool import AccountPool

STOP=datetime.fromisoformat('2026-11-02T14:59:30-08:00').timestamp()


def notify(text):
    import subprocess
    subprocess.run(['agent-message','claude','investigator',text],capture_output=True,text=True,timeout=20)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=P.ROOT/'outputs/task08-race-300s-01')
    args=parser.parse_args()
    if STOP-time.time()<330:raise SystemExit('Insufficient time before14:59:30 hard stop')
    checks=P.preflight()
    if not all(checks.values()):raise SystemExit('Preflight failed without model calls: '+str(checks))
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    P.write_json(P.ROOT/'outputs/short-pilots-20261102/task08-race-process.json',
        {'pid':os.getpid(),'command':[str(P.ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
        '--output',str(root)],'started_at':time.time(),'combined_api_cost_guard_usd':6})
    hard_deadline=time.monotonic()+STOP-time.time()
    phase={'deadline':hard_deadline}
    sims=SimulationPool(lambda:phase['deadline'],root/'simulation-waits.jsonl',max_concurrent=4,memory_limit_gib=6.5)
    coordinators={}
    for name,total in (('deepseek',24),('astra',4)):
        co=Coordinator(root/name,P.ROOT/'outputs/task08-reference-baseline-01',
            P.ROOT/'outputs/task08-pilot-02',seconds=300,tests=total,tests_per_worker=4,
            calls=1800,calls_per_worker=300,tokens=6_000_000,output_tokens=8192,cost=6)
        co.simulator=sims;co.logical_calls=100;co.active_worker_limit=6 if name=='deepseek' else 1
        coordinators[name]=co
    workers=[Session(coordinators['deepseek'].root/name,coordinators['deepseek'],name,name,brief)
             for name,brief in DS_ROLES]
    astra=Session(coordinators['astra'].root/'astra',coordinators['astra'],*ASTRA_ROLE[:1],
                  'astra',ASTRA_ROLE[1],condition='astra')
    pool=AccountPool(root/'shared-api',W.primary_keys(),worker_ids=[s.worker_id for s in workers],
        deadline=hard_deadline,max_concurrent=3,max_workers=6,logical_requests_per_worker=100,
        max_calls=1800,per_worker_calls=300,max_tokens=6_000_000,max_output_tokens=8192,max_cost=6,
        prompt_tpm=500000,uncached_prompt_tpm=250000,generated_tpm=60000,prompt_bytes_per_token=2.9)
    coordinators['deepseek'].client=pool
    import claude_mac_evaluation
    start=threading.Event()
    summaries={}
    def run(session):
        start.wait()
        began=time.monotonic()
        try:
            result=P.launch(session,100) if session.condition=='astra' else pool.run_worker(session,session.worker_id,100)
            session.coordinator.rehearse_remaining([session])
            return result
        except Exception as error:
            return {'worker_id':session.worker_id,'error':f'{type(error).__name__}: {error}'}
        finally:
            P.write_json(session.root/'race-timing.json',{'started_monotonic':began,
                'delay_after_common_start_seconds':began-T0,'finished_monotonic':time.monotonic()})
    began=time.monotonic()
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures={s.worker_id:executor.submit(run,s) for s in [*workers,astra]}
        T0=time.monotonic();phase['deadline']=T0+300
        for co in coordinators.values():co.start(started_monotonic=T0,deadline=T0+300)
        for s in [*workers,astra]:s.start()
        pool.deadline=T0+300
        for lane in pool.lanes:
            with lane.condition:lane.deadline=T0+300;lane.condition.notify_all()
        P.write_json(root/'protocol.json',{'task':8,'kind':'shortened_exploratory_paired_300s',
            'development_seconds_per_agent':300,'new_probes_per_agent':4,'deepseek_workers':6,'astra_workers':1,
            'api_accounts':3,'api_slots_per_account':3,'combined_api_guard_usd':6,
            'initial_tpm_per_account':{'prompt':500000,'uncached':250000,'generated':60000},
            'reported_provider_limits_target_fraction':.9,'simulations_shared_across_conditions':4,
            'memory_fallback_simulations':3,'memory_gate_gib':6.5,'common_start_monotonic':T0,
            'waits_count_against_fixed_window':True,'finals_only_after_both_selections_frozen':True,
            'hard_stop_at':'2026-11-02T14:59:30-08:00','started_at':time.time(),
            'comparison_limits':['Six workers have24aggregate probes versus Astra4; per-agent time/probes match',
                'Astra uses subscription; API turn and token budgets differ from DeepSeek guard',
                'CLI host overhead is outside simulator containers; existing Task08 containers count toward shared slot capacity']})
        start.set()
        try:
            for worker,future in futures.items():summaries[worker]=future.result()
        finally:pool.close()
    while sims.summary()['active'] and time.monotonic()<min(hard_deadline,T0+330):time.sleep(.1)
    if sims.summary()['active']:raise RuntimeError('Public simulations did not settle before final barrier')
    selections={name:co.select() for name,co in coordinators.items()}
    P.write_json(root/'both-frozen.json',{'at':time.time(),'selections':selections,'all_agents_stopped':True})
    P.write_json(root/'worker-summaries.json',summaries)
    phase['deadline']=hard_deadline
    def final(name):
        co=coordinators[name]
        started=time.monotonic()
        try:
            with sims.owner(name,'selected_final'):
                reward,reused=W.final_evaluate(co,selections[name],STOP+30,900)
            result={'status':'completed','score':round(reward['reward']*100,2),'reward':reward,
                    'reused_identical_final_from':reused}
        except Exception as error:result={'status':'final_failed','score':None,'error':str(error)}
        result.update(selected_worker=selections[name]['worker'],final_seconds=time.monotonic()-started)
        P.write_json(co.root/'summary.json',result)
        notify(f"Task08 paired300s {name}: score={result['score']}, selected={result['selected_worker']}; shortened/exploratory. No private feedback during development.")
        return result
    with ThreadPoolExecutor(max_workers=2) as executor:
        final_futures={name:executor.submit(final,name) for name in coordinators}
        finals={name:f.result() for name,f in final_futures.items()}
    api=pool.summary()
    from task09_planner_short import throughput
    turns=throughput(root,api)
    for s in [*workers,astra]:
        rows=[r for r in s.coordinator.entries if r.get('worker')==s.worker_id]
        item=turns.setdefault(s.worker_id,{})
        item.update(probes_measured=sum(r.get('status')=='measured' for r in rows),
                    probes_remaining_queued=sum(r.get('status') in ('queued','running') for r in rows),
                    probes_failed=sum(r.get('status') in ('error','time_exhausted') for r in rows),
                    tool_calls=summaries.get(s.worker_id,{}).get('tool_calls'))
    result={'status':'completed','kind':'shortened_exploratory','conditions':finals,'api':api,
        'turns_probes_per_worker':turns,'astra_subscription_usage':summaries.get('astra',{}).get('subscription_usage'),
        'simulation_pool':sims.summary(),'per_worker_simulation_wait_log':'simulation-waits.jsonl',
        'probe_event_logs':['deepseek/probe-events.jsonl','astra/probe-events.jsonl'],
        'total_seconds':time.monotonic()-began,'finished_at':time.time()}
    P.write_json(root/'summary.json',result)
    notify(f"Task08 paired300s finished: {finals}; API estimate=${api['estimated_cost_usd']:.4f}. Report {root/'summary.json'}; no repeat launched.")


if __name__=='__main__':main()
