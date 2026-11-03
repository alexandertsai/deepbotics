"""Five-minute Task09 engineering attempt with six DeepSeek workers only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import threading
import time

import task08_pilot as P
import task09_pilot as Q
import task09_swarm_followup as F
import task09_parallel_probe as E
from task09_team_sessions import Coordinator,Session,DS_ROLES
from task08_sim_pool import SimulationPool
from short_pilot_pool import AccountPool
from task08_swarm_short import primary_keys
from short_pilot_runtime import docker_memory_bytes,GIB

STOP=datetime.fromisoformat('2026-11-02T14:59:30-08:00').timestamp()


def notify(value):
    import subprocess
    subprocess.run(['agent-message','claude','investigator',value],capture_output=True,text=True,timeout=20)


class FocusSession(Session):
    def make_prompt(self):
        text=super().make_prompt()
        text=text.replace('same fresh public starter','same supplied starter').replace(
            'three supplied bare GELLO models and zero-torque trim stubs','three supplied GELLO designs and trim implementations')
        return text.replace('The Astra lead makes the final engineering selection.',
            'The host selects the highest measured public proxy and retains the starter on ties or regressions.').replace(
            'submit a useful frozen branch for the lead to assess','submit a useful frozen branch for public selection')

    def dispatch(self,name,arguments):
        if self.lead and name=='submit':raise ValueError('Use choose_candidate(index,reason) to freeze your measured final choice')
        return super().dispatch(name,arguments)


class FocusCoordinator(Coordinator):
    def select(self):
        selection=F.Coordinator.select(self)
        selection['selection_policy']='public_best_deepseek_only'
        P.write_json(self.root/'selection.json',selection)
        return selection

    def test(self,session,hypothesis,source=None):
        started=time.monotonic()
        try:
            with self.simulator.owner(session.condition,session.worker_id):
                return super().test(session,hypothesis,source=source)
        finally:
            with self.lock:
                with (session.root/'probe-wall-times.jsonl').open('a') as handle:
                    handle.write(json.dumps({'at':time.time(),'total_probe_and_queue_seconds':time.monotonic()-started})+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=P.ROOT/'outputs/task09-focus-300s-01')
    parser.add_argument('--starter',type=Path)
    parser.add_argument('--max-cost',type=float,default=10.)
    args=parser.parse_args()
    if not 0<args.max_cost<=10:parser.error('Combined API guard must be at most$10')
    if STOP-time.time()<330:raise SystemExit('Insufficient time before14:59:30')
    if not all(Q.preflight().values()):raise SystemExit('Task09 preflight failed without model calls')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    command=[str(P.ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--output',str(root)]
    P.write_json(P.ROOT/'outputs/short-pilots-20261102/task09-focus-process.json',
                 {'pid':os.getpid(),'command':command,'started_at':time.time(),'api_guard_usd':args.max_cost})
    hard_deadline=time.monotonic()+STOP-time.time()
    lock=Path('/tmp/kimirax-short-task09-sim.lock').open('a')
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>=hard_deadline-330:raise TimeoutError('Task09 live-priority lock deadline')
            time.sleep(.1)
    try:
        reference=root/'reference';reference.mkdir()
        source=args.starter
        if source is None:
            source=root/'public-starter-source';Q.make_starter(source)
        from services.controller.storage import snapshot
        manifest=snapshot(source.resolve(),reference/'frozen',max_bytes=256*1024**2,workspace_path='code/task09')
        P.write_json(reference/'manifest.json',manifest)
        while docker_memory_bytes()>6.5*GIB:
            if time.monotonic()>=hard_deadline-330:raise TimeoutError('Memory admission deadline')
            time.sleep(1)
        F.Q.evaluate=E.evaluate
        host=FocusCoordinator(root/'team',reference,reference,seconds=300,tests=24,tests_per_worker=4,
            worker_count=6,calls=1800,calls_per_worker=300,tokens=30_000_000,
            output_tokens=8192,cost=args.max_cost)
        host.logical_calls_per_worker=100
        phase={'deadline':hard_deadline}
        sims=SimulationPool(lambda:phase['deadline'],root/'simulation-waits.jsonl',max_concurrent=4,
                            memory_limit_gib=6.5,container_reader=E.task09_container_ids)
        host.simulator=sims
        workers=[FocusSession(host.root/name,host,name,name,brief) for name,brief in DS_ROLES]
        pool=AccountPool(root/'shared-api',primary_keys(),worker_ids=[s.worker_id for s in workers],
            deadline=hard_deadline,max_concurrent=3,max_workers=6,logical_requests_per_worker=100,
            max_calls=1800,per_worker_calls=300,max_tokens=30_000_000,max_output_tokens=8192,
            max_cost=args.max_cost,prompt_tpm=500000,uncached_prompt_tpm=250000,
            generated_tpm=60000,prompt_bytes_per_token=2.9)
        host.client=pool
        import claude_mac_evaluation
        start=threading.Event();summaries={}
        def run(session):
            start.wait()
            t=time.monotonic()
            try:
                result=pool.run_worker(session,session.worker_id,100)
                host.rehearse_remaining([session])
                return result
            except Exception as error:return {'worker_id':session.worker_id,'error':f'{type(error).__name__}: {error}'}
            finally:P.write_json(session.root/'timing.json',{'start_delay_seconds':t-T0,'finished_at':time.time()})
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures={s.worker_id:executor.submit(run,s) for s in workers}
            T0=time.monotonic();host.start();host.deadline=T0+300;phase['deadline']=T0+300
            for session in workers:session.start()
            pool.deadline=T0+300
            for lane in pool.lanes:
                with lane.condition:lane.deadline=T0+300;lane.condition.notify_all()
            P.write_json(root/'protocol.json',{'task':9,'kind':'shortened_exploratory_design_improvement' if args.starter else 'shortened_exploratory_public_starter',
                'seconds':300,'common_start_monotonic':T0,'accounts':3,'api_slots_per_account':3,
                'initial_tpm':{'prompt':500000,'uncached':250000,'generated':60000},'reported_limit_fraction':.9,
                'explorers':6,'astra_model_calls':0,'probes_per_agent':4,'total_probes':24,'shared_simulations':4,
                'public_cpu_limit_each':1,'public_memory_gib_each':1,'aggregate_public_cpus_max':4,
                'memory_gate_gib':6.5,'memory_fallback_slots':3,'api_guard_usd':args.max_cost,
                'starter_source':str(source),'starter_manifest':manifest,'seed_chosen_after_historical_evaluation':bool(args.starter),
                'published_score_target':56.1133,'claim_scope':'Local artifact score; not an independently validated leaderboard entry',
                'final_after_all_agents_frozen':True,'no_private_feedback_to_agents':True,'hard_stop_at':STOP,
                'all_waits_count_against_300_seconds':True})
            start.set()
            try:
                for name,future in futures.items():summaries[name]=future.result()
            finally:pool.close()
        while sims.summary()['active'] and time.monotonic()<min(hard_deadline,T0+330):time.sleep(.1)
        if sims.summary()['active']:raise RuntimeError('Public probes did not settle before selection')
        development_wall_seconds=time.monotonic()-T0
        selection=host.select()
        P.write_json(root/'all-frozen.json',{'selection':selection,'at':time.time(),'all_agents_stopped':True})
        P.write_json(root/'worker-summaries.json',summaries)
        phase['deadline']=hard_deadline
        result={'status':'selected','score':None,'selection':selection,'api':pool.summary()}
        try:
            with sims.owner('team','selected_final'):
                with sims:reward=E.evaluate(host.root/'selected',host.root/'final',final=True,
                                           timeout=min(900,max(.1,hard_deadline-time.monotonic())))
            result.update(status='completed',score=round(reward['reward']*100,2),official_reward=reward)
        except Exception as error:result.update(status='final_failed',error=str(error))
        from task09_planner_short import throughput
        result['deepseek_turns']=throughput(root,result['api'])
        result.update(astra_model_calls=0,
            probes=host.entries,simulation_pool=sims.summary(),development_window_seconds=300,
            development_including_cleanup_seconds=development_wall_seconds,
            total_seconds_including_final=time.monotonic()-T0,
            wait_log='simulation-waits.jsonl',finished_at=time.time())
        P.write_json(root/'summary.json',result)
        notify(f"Task09 focused300s finished: score={result['score']}, API estimate=${result['api']['estimated_cost_usd']:.4f}, selection={selection.get('selection_policy')}; report {root/'summary.json'}. No repeat launched.")
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
