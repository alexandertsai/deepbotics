"""Six-worker shortened exploratory Task09 run with authorized account pooling."""

import argparse
import json
from pathlib import Path
import time

import task08_pilot as P
import task09_swarm_followup as F
from short_pilot_pool import AccountPool, ACCOUNT_COUNT
from task08_swarm_short import SimulationLock, hard_stop_timestamp, primary_keys


def simulation_lock(path, deadline, log_path):
    from short_pilot_runtime import MemoryGatedSimulationLock
    return MemoryGatedSimulationLock(path,deadline,log_path)


class Session(F.SwarmSession):
    def make_prompt(self):
        text=super().make_prompt()
        return text + ('\nThis shortened exploratory run permits three probes per worker. Aim for at least two '
            'measured iterations within the shared900-second window. The public selector is known to '
            'be imperfect and is being reused experimentally, with no private feedback available. '
            'Public proxy improvement is not evidence of private improvement.\n')


def setup_coordinator(args, stop, seconds):
    deadline = time.monotonic() + max(0, stop-time.time()-30)
    evaluate = F.Q.evaluate
    def bounded_evaluate(*call_args, **kwargs):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Hard stop before starter evaluation')
        kwargs['timeout'] = min(kwargs.get('timeout',900),remaining)
        return evaluate(*call_args,**kwargs)
    with simulation_lock(args.simulation_lock,lambda:deadline,
                         args.output.with_name(args.output.name+'-baseline-waits.jsonl')):
        F.Q.evaluate = bounded_evaluate
        try:
            return F.Coordinator(args.output,P.ROOT/'outputs/task09-starter-baseline-01',
                P.ROOT/'outputs/task09-pilot-01',seconds=seconds,tests=6*args.tests_per_worker,
                calls=216,calls_per_worker=36,tokens=6_000_000,output_tokens=8192,
                cost=args.max_cost,tests_per_worker=args.tests_per_worker,worker_count=6)
        finally:
            F.Q.evaluate = evaluate


def wait_final_release(path, stop, not_before):
    if path is None:
        raise ValueError('Both-condition final barrier is required')
    while not path.exists():
        if time.time()>=stop-30:
            raise TimeoutError('Hard stop awaiting both frozen selections')
        time.sleep(1)
    release=json.loads(path.read_text())
    if not release.get('all_conditions_frozen_or_stopped') or release.get('released_at',0)<not_before:
        raise ValueError('Final barrier is invalid or stale')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=P.ROOT/'outputs/task09-swarm-06-short-01')
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--tests-per-worker',type=int,default=3)
    parser.add_argument('--max-cost',type=float,default=10.)
    parser.add_argument('--api-concurrency',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--output-tpm',type=int,default=60000)
    parser.add_argument('--simulation-lock',type=Path,default=Path('/tmp/kimirax-short-task09-sim.lock'))
    parser.add_argument('--hard-stop-at',default='2026-11-02T15:00:00-08:00')
    parser.add_argument('--final-barrier',type=Path,required=True)
    args=parser.parse_args(argv)
    invocation_started=time.time()
    stop=hard_stop_timestamp(args.hard_stop_at)
    if stop>hard_stop_timestamp():parser.error('Hard stop cannot exceed today15:00 Pacific')
    seconds=min(args.seconds,int(stop-time.time()-30))
    if not 1<=seconds<=900 or not 2<=args.tests_per_worker<=3 or not 0<args.max_cost<=10:
        parser.error('Shortened limits are900s,2..3probes per worker,$10 combined')
    checks=F.Q.preflight()
    if not all(checks.values()):raise SystemExit('Task09 preflight failed: '+str(checks))
    roles=F.ROLES[:6]
    coordinator=setup_coordinator(args,stop,seconds)
    coordinator.logical_calls_per_worker=12
    coordinator.simulator=simulation_lock(args.simulation_lock,lambda:coordinator.deadline,
                                        coordinator.root/'simulation-waits.jsonl')
    sessions=[Session(coordinator.root/name,coordinator,name,name,brief) for name,brief in roles]
    P.write_json(coordinator.root/'protocol.json',{'task':9,'kind':'shortened_exploratory_six_worker_rerun',
        'workers':[name for name,_ in roles],'seconds_total':seconds,
        'new_development_probes_per_worker_limit':args.tests_per_worker,
        'new_development_probe_limit_total':6*args.tests_per_worker,
        'logical_requests_per_worker':12,'api_concurrency':ACCOUNT_COUNT*args.api_concurrency,'api_slots_per_account':args.api_concurrency,
        'active_worker_limit':6,'authorized_accounts':ACCOUNT_COUNT,'estimated_spend_guard_usd_total':args.max_cost,
        'selection_metric':F.SELECTION_DESCRIPTION,'selection_weights':F.SELECTION_WEIGHTS,
        'selector_known_to_fail_retrospective_pairwise_checks':True,
        'launch_block_override':'User directly approved these shortened exploratory pilots only',
        'selection_used_private_results':False,'incumbent_retention':True,
        'simulator_cpus':4,'simulator_memory_gb':4,'simulator_concurrency_across_conditions':1,
        'simulation_lock':str(args.simulation_lock),'hard_stop_at':args.hard_stop_at,
        'started_at':time.time(),'token_pacing_per_account':{'prompt':500000,'uncached':250000,'generated':args.output_tpm},
        'memory_admission_gib':6.5,'queue_wait_counts_toward_fixed_development_window':True,
        'wait_log':'simulation-waits.jsonl','parallel_schedule':'Three authorized accounts with independent pacing',
        'comparison_limits':['Six workers versus one Astra agent; equal per-agent time/probe allowances but unequal total compute',
            'Public proxy is exploratory and does not cover full private validity or robustness',
            'CLI host overhead is outside simulator4CPU/4GB budget']})
    coordinator.start()
    coordinator.deadline=min(coordinator.deadline,time.monotonic()+max(0,stop-time.time()-30))
    began=time.monotonic()
    client=AccountPool(coordinator.root/'shared-api',primary_keys(),
        worker_ids=[s.worker_id for s in sessions],deadline=coordinator.deadline,
        max_concurrent=args.api_concurrency,max_workers=6,logical_requests_per_worker=12,
        prompt_tpm=500000,uncached_prompt_tpm=250000,generated_tpm=args.output_tpm,
        prompt_bytes_per_token=2.9,max_retries=2,**coordinator.api_limits)
    coordinator.client=client
    try:
        summaries=F.run_workers(coordinator,client,sessions,6)
    finally:
        client.close()
        P.write_json(coordinator.root/'api-summary.json',client.summary())
    coordinator.rehearse_remaining(sessions)
    for session,row in zip(sessions,summaries):row['probes_including_rehearsals']=session.tests
    P.write_json(coordinator.root/'worker-summaries.json',summaries)
    selection=coordinator.select()
    P.write_json(coordinator.root/'selection-frozen.json',{'selection':selection,'at':time.time()})
    print({'event':'selection_frozen','worker':selection['worker']},flush=True)
    result={'status':'selected','kind':'shortened_exploratory','selected_worker':selection['worker'],
        'score':None,'development_seconds':time.monotonic()-began,'new_development_probes':coordinator.tests,
        'api':client.summary(),'workers_with_host_errors':sum(bool(r.get('error')) for r in summaries)}
    try:
        wait_final_release(args.final_barrier,stop,invocation_started)
        final_deadline=time.monotonic()+max(0,stop-time.time()-30)
        with simulation_lock(args.simulation_lock,lambda:final_deadline,
                             coordinator.root/'simulation-waits.jsonl'):
            remaining=stop-time.time()-30
            if remaining<=0:raise TimeoutError('Hard stop reached before final evaluation')
            started=time.monotonic()
            reward=F.Q.evaluate(coordinator.root/'selected',coordinator.root/'final',final=True,timeout=min(900,remaining))
            result.update(status='completed',official_reward=reward,score=round(reward['reward']*100,2),
                          final_verification_seconds=time.monotonic()-started)
    except Exception as error:
        result.update(status='final_evaluation_failed',evaluation_error=f'{type(error).__name__}: {error}')
    result['total_seconds_including_starter_probe']=time.monotonic()-began+coordinator.starter_probe_seconds
    if selection['worker']=='unchanged_reference':result['interpretation']='Unchanged starter retained; no generated improvement selected'
    P.write_json(coordinator.root/'summary.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
