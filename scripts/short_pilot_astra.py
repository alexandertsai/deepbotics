"""Fresh Astra comparison with the same public probe and incumbent retention."""

import argparse
import json
from pathlib import Path
import time

import task08_pilot as P
from task08_swarm_short import SimulationLock, hard_stop_timestamp


def setup(task, output, seconds, tests, lock_path, hard_deadline):
    if task == 8:
        import task08_swarm_short as W
        adapter = P
        base = W.Session
        coordinator = W.Coordinator(output, P.ROOT / 'outputs/task08-reference-baseline-01',
            P.ROOT / 'outputs/task08-pilot-02', seconds=seconds, tests=tests,
            calls=12, calls_per_worker=12, tokens=6_000_000, output_tokens=8192,
            cost=6., tests_per_worker=tests)
    else:
        import task09_swarm_followup as W
        import task09_pilot as adapter
        from short_pilot_runtime import MemoryGatedSimulationLock
        base = W.SwarmSession
        evaluate=W.Q.evaluate
        def bounded_evaluate(*call_args,**kwargs):
            remaining=hard_deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Hard stop before Astra starter evaluation')
            kwargs['timeout']=min(kwargs.get('timeout',900),remaining)
            return evaluate(*call_args,**kwargs)
        with MemoryGatedSimulationLock(lock_path,lambda:hard_deadline,
                                      output.with_name(output.name+'-baseline-waits.jsonl')):
            W.Q.evaluate=bounded_evaluate
            try:
                coordinator = W.Coordinator(output, P.ROOT / 'outputs/task09-starter-baseline-01',
                    P.ROOT / 'outputs/task09-pilot-01', seconds=seconds, tests=tests,
                    calls=12, calls_per_worker=12, tokens=6_000_000, output_tokens=8192,
                    cost=10., tests_per_worker=tests, worker_count=1)
            finally:W.Q.evaluate=evaluate
    coordinator.logical_calls = 12
    coordinator.logical_calls_per_worker = None
    coordinator.active_worker_limit = 1
    if task==9:
        coordinator.simulator=MemoryGatedSimulationLock(lock_path,lambda:coordinator.deadline,
                                                       coordinator.root/'simulation-waits.jsonl')
    else:coordinator.simulator=SimulationLock(lock_path, lambda: coordinator.deadline)

    class AstraSession(base):
        def __init__(self):
            super().__init__(coordinator.root / 'astra', coordinator, 'astra', 'astra',
                'Improve the complete design conservatively across all three variants. Use public measurements and preserve the best tested incumbent.')
            self.condition = 'astra'
            self.prompt = self.make_prompt()
            (self.root / 'prompt.txt').write_text(self.prompt)

        def make_prompt(self):
            text = super().make_prompt()
            for old in ('of four independent DeepSeek workers', 'of 6 independent DeepSeek workers',
                        'of 1 independent DeepSeek workers'):
                text = text.replace(old, 'of one Astra agent')
            text = text.replace('The four workers SHARE', 'This agent has')
            text = text.replace('The 6 workers SHARE', 'This agent has')
            text = text.replace('The workers SHARE', 'This agent has')
            text = text.replace('This short run has 6 workers', 'This short comparison has one Astra agent')
            if tests == 2:
                text = text.replace('Plan three measured iterations:', 'Use your two probe opportunities:')
            text += (f'\nThis is a shortened exploratory Astra comparison. You have the same {seconds}-second '
                f'development ceiling and {tests} new public probe opportunities as each swarm worker. '
                'The host retains the best publicly measured incumbent, including the unchanged starter. '
                'Final private feedback is unavailable until selection is frozen. No prior private results are supplied.\n')
            return text

    return coordinator, AstraSession(), adapter


def main():
    invocation_started=time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, choices=(8,9), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--tests', type=int, default=2)
    parser.add_argument('--simulation-lock', type=Path, required=True)
    parser.add_argument('--hard-stop-at', default='2026-11-02T15:00:00-08:00')
    parser.add_argument('--final-barrier',type=Path)
    args = parser.parse_args()
    if args.task==9 and not args.final_barrier:parser.error('Task09 requires both-condition final barrier')
    stop = hard_stop_timestamp(args.hard_stop_at)
    hard_deadline=time.monotonic()+max(0,stop-time.time()-30)
    seconds = min(args.seconds, int(stop-time.time()-30))
    if not 1 <= seconds <= 900 or args.tests not in (2,3):
        raise SystemExit('Astra comparison needs time remaining and two or three public probes')
    coordinator, session, adapter = setup(args.task,args.output,seconds,args.tests,args.simulation_lock,hard_deadline)
    P.write_json(coordinator.root / 'protocol.json', {'task':args.task,'kind':'shortened_exploratory_fresh_astra',
        'development_seconds':seconds,'new_public_probes':args.tests,'same_starter_as_swarm':True,
        'same_public_proxy_and_incumbent_retention':True,'subscription_usage_recorded':True,
        'hard_stop_at':args.hard_stop_at,'simulation_lock':str(args.simulation_lock),
        'queue_wait_counts_toward_fixed_development_window':True,
        'comparison_limit':'One Astra agent versus six workers; equal per-agent time/probe allowances, unequal aggregate compute.'})
    coordinator.start()
    coordinator.deadline=min(coordinator.deadline,hard_deadline)
    started=time.monotonic()
    agent=adapter.launch(session,12)
    coordinator.rehearse_remaining([session])
    selection=coordinator.select()
    P.write_json(coordinator.root/'selection-frozen.json',{'selection':selection,'at':time.time()})
    print(json.dumps({'event':'selection_frozen','task':args.task,'worker':selection['worker']}),flush=True)
    remaining=stop-time.time()-30
    result={'status':'selected','agent':agent,'selected_worker':selection['worker'],
        'development_seconds':time.monotonic()-started,'new_development_probes':coordinator.tests,
        'kind':'shortened_exploratory','score':None}
    try:
        if args.task==9:
            from task09_swarm_short import wait_final_release
            wait_final_release(args.final_barrier,stop,invocation_started)
        elif args.final_barrier:
            while not args.final_barrier.exists():
                if time.time()>=stop-30:raise TimeoutError('Hard stop awaiting both frozen selections')
                time.sleep(1)
        if remaining <= 0: raise TimeoutError('Hard stop reached before final evaluation')
        if args.task==9:
            from short_pilot_runtime import MemoryGatedSimulationLock
            final_lock=MemoryGatedSimulationLock(args.simulation_lock,lambda:hard_deadline,
                                                coordinator.root/'simulation-waits.jsonl')
        else:final_lock=SimulationLock(args.simulation_lock,lambda:hard_deadline)
        with final_lock:
            remaining=stop-time.time()-30
            if remaining <= 0: raise TimeoutError('Hard stop reached while awaiting final evaluation')
            evaluate=P.docker_evaluate if args.task==8 else adapter.evaluate
            final_started=time.monotonic()
            reward=evaluate(coordinator.root/'selected',coordinator.root/'final',final=True,
                            timeout=min(900,remaining))
            result.update(status='completed',official_reward=reward,
                score=round(reward['reward']*100,2),final_verification_seconds=time.monotonic()-final_started)
    except Exception as error:
        result.update(status='final_evaluation_failed',evaluation_error=f'{type(error).__name__}: {error}')
    result['total_seconds']=time.monotonic()-started
    P.write_json(coordinator.root/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
