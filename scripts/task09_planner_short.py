"""Isolated public planning followed by the unchanged Task09 short-run selector."""

import argparse
from datetime import datetime
import json
import re
from pathlib import Path
import time

import task08_pilot as P
import task09_swarm_followup as F
import task09_pilot as Q
from short_pilot_pool import AccountPool
from short_pilot_runtime import MemoryGatedSimulationLock
from task08_swarm_short import primary_keys

ADMISSION=datetime.fromisoformat('2026-11-02T14:00:00-08:00').timestamp()
STOP=datetime.fromisoformat('2026-11-02T14:59:30-08:00').timestamp()


def notify(value):
    import subprocess
    subprocess.run(['agent-message','claude','investigator',value],capture_output=True,text=True,timeout=20)


def throughput(root,api):
    rows={}
    for lane in api['per_account']:
        for worker,admitted in lane.get('worker_logical_requests',{}).items():
            rows[worker]={'logical_requests_admitted':admitted,'successful_responses':set(),
                'provider_attempts':lane.get('worker_requests',{}).get(worker,0),
                'provider_queue_seconds':0.,'pacing_wait_seconds':0.,'failed_responses':0}
    for path in (root/'shared-api').glob('account-*/api/metrics-*.json'):
        record=json.loads(path.read_text());worker=rows.get(record.get('worker_id'))
        if worker is None:continue
        if record.get('status_code')==200:worker['successful_responses'].add(record.get('logical_request',record['call']))
        else:worker['failed_responses']+=1
        worker['provider_queue_seconds']+=record.get('queue_seconds',0) or 0
        worker['pacing_wait_seconds']+=record.get('pacing_wait_seconds',0) or 0
    for row in rows.values():row['successful_model_turns']=len(row.pop('successful_responses'))
    return rows


class Coordinator(F.Coordinator):
    def test(self,session,hypothesis,source=None):
        first=not session.first_assigned_probe_measured
        if first:
            candidate=source if source is not None else session.candidate
            if all((candidate/name).read_bytes()==(self.root/'starter'/name).read_bytes()
                   for name in F.ENTRY_FILES):
                raise ValueError('Unchanged starter does not test an implemented assigned hypothesis')
        if first:hypothesis=json.dumps(session.assignment)+'\nTest note: '+str(hypothesis)
        result=super().test(session,hypothesis,source=source)
        if first:
            session.first_assigned_probe_measured=True
            session.first_assigned_probe_origin='worker_requested' if source is None else 'host_rehearsal'
            P.write_json(session.root/'first-assigned-probe.json',{'assignment':session.assignment,
                'at':time.time(),'result_version':result.get('version'),
                'origin':session.first_assigned_probe_origin,
                'implementation_semantics_not_automatically_verified':True})
        return result


class Session(F.SwarmSession):
    def __init__(self,root,coordinator,assignment):
        self.assignment=assignment
        self.first_assigned_probe_measured=False
        self.first_assigned_probe_origin=None
        super().__init__(root,coordinator,assignment['id'],assignment['id'],assignment['hypothesis'])

    def make_prompt(self):
        text=F.Q.Session.make_prompt(self)
        files='\n'.join(f'Starter {name}:\n{(self.coordinator.root/"starter"/name).read_text()}'
                        for name in F.ENTRY_FILES)
        return text+f'''
You are one of six workers in a shortened exploratory planner-led run.
Your assigned hypothesis was produced by planners given only the public task and unchanged starter:
{json.dumps(self.assignment,ensure_ascii=False)}
Implement this hypothesis and test it before exploring alternatives. Your first test_candidate call must include
the assigned ID {self.assignment['id']} in its hypothesis argument. The host records this first test against
the assigned hypothesis. Do not claim that it was implemented unless your edits actually implement it.
Until the first assigned probe is measured, peer findings and peer probe reports are unavailable.
After that first measurement, revise from evidence and use shared public findings if helpful.
You have900shared wall seconds,12logical model requests and3new public probe opportunities.
Aim for at least two measured iterations, with the third reserved for a final revision check.
All six starter files are supplied below. Avoid repeatedly reading them or polling status.
Only public development measurements can select the incumbent; the unchanged starter wins ties/regressions.
Selection is unchanged from the preceding short-run implementation:35%hold quality,45%passive residual quality,
20%mean servo headroom. This proxy is exploratory and does not reproduce the private score.
Only measured frozen versions are eligible. submit freezes your branch and ends it.
After every worker stops and selection is frozen, the host privately evaluates the selection once.
No private outcomes are returned to workers. Any later per-worker private scoring is operator analysis only.
Queue, API pacing and memory waits count against the fixed900-second window and are logged.
Public helper paths include public/franka/harness/probe_protocol.py, torques.py and scenarios.py;
substitute ur5e or xarm7 for the other variants. Read shared findings after your first measured test.
Starter public measurement:
{json.dumps(F.compact_report(self.coordinator.baseline_report),separators=(',',':'))}
{files}
'''

    def dispatch(self,name,arguments):
        if not self.first_assigned_probe_measured:
            if name in ('read_probe','share_finding'):
                raise ValueError('Measure the assigned hypothesis before exchanging peer evidence')
            if name in ('status','swarm_status'):
                return {'worker':self.worker_id,'seconds_remaining':max(0,self.deadline-time.monotonic()),
                    'assigned_hypothesis':self.assignment,'first_assigned_probe_measured':False,
                    'new_probes_remaining':3-self.tests,'peer_evidence_available':False}
            if name=='submit':raise ValueError('Test the assigned hypothesis before submission')
            if name=='test_candidate':
                if not re.search(r'(?<![A-Za-z0-9_])'+re.escape(self.assignment['id'])+
                                 r'(?![A-Za-z0-9_])',arguments.get('hypothesis','')):
                    raise ValueError('First test must identify the assigned hypothesis')
                result=super().dispatch(name,{'hypothesis':json.dumps(self.assignment)+
                    '\nWorker test note: '+arguments['hypothesis']})
                return result
        return super().dispatch(name,arguments)


def coordinator(root,reference,hard_deadline):
    evaluate=F.Q.evaluate
    def bounded(*args,**kwargs):
        remaining=hard_deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Deadline before public starter probe')
        kwargs['timeout']=min(remaining,kwargs.get('timeout',900))
        return evaluate(*args,**kwargs)
    with MemoryGatedSimulationLock('/tmp/kimirax-short-task09-sim.lock',lambda:hard_deadline,
                                   root.parent/'baseline-simulation-waits.jsonl'):
        F.Q.evaluate=bounded
        try:
            return Coordinator(root,reference,reference,seconds=900,tests=18,calls=234,
                calls_per_worker=36,tokens=6_000_000,output_tokens=8192,cost=10,
                tests_per_worker=3,worker_count=6)
        finally:F.Q.evaluate=evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=P.ROOT/'outputs/task09-planner-short-01')
    parser.add_argument('--api-concurrency',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--output-tpm',type=int,default=60000)
    args=parser.parse_args()
    if time.time()>=ADMISSION:raise SystemExit('Dropped: Task09 planner launch missed14:00 Pacific')
    from task09_planning import build_plan
    from task09_planner_analysis import analyze_worker_finals
    from services.controller.storage import snapshot
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    P.write_json(P.ROOT/'outputs/short-pilots-20261102/task09-planner-process.json',
        {'pid':__import__('os').getpid(),'command':[__import__('sys').executable,str(Path(__file__).resolve()),
          '--output',str(root)],'started_at':time.time()})
    source=root/'public-starter-source'
    Q.make_starter(source)
    reference=root/'public-reference'
    reference.mkdir()
    manifest=snapshot(source,reference/'frozen',max_bytes=256*1024**2,workspace_path='code/task09')
    P.write_json(reference/'manifest.json',manifest)
    began=time.monotonic()
    hard_deadline=time.monotonic()+max(0,STOP-time.time())
    planning_deadline=min(time.monotonic()+180,hard_deadline,
                          time.monotonic()+max(0,ADMISSION-time.time()-5))
    ids=['planner_a','planner_b']+[f'hypothesis_{i}' for i in range(1,7)]
    pool=AccountPool(root/'shared-api',primary_keys(),worker_ids=ids,
        deadline=planning_deadline,max_concurrent=args.api_concurrency,max_workers=8,logical_requests_per_worker=12,
        max_calls=234,per_worker_calls=36,max_tokens=6_000_000,max_output_tokens=8192,max_cost=10,
        prompt_tpm=500000,uncached_prompt_tpm=250000,generated_tpm=args.output_tpm,prompt_bytes_per_token=2.9)
    result={'kind':'shortened_exploratory_planner_led','score':None,'status':'planning'}
    sessions=[]
    selected=None
    try:
        plan=build_plan(pool,reference/'frozen',root/'planning',planning_deadline)
        P.write_json(root/'plan.json',plan)
        notify('Task09 public-only planner output saved: '+str(root/'plan.json')+
               '; hypotheses: '+'; '.join(h['title'] for h in plan['hypotheses']))
        if time.time()>=ADMISSION-5:raise TimeoutError('Dropped: workers could not launch before14:00 Pacific')
        host=coordinator(root/'workers',reference,hard_deadline)
        host.logical_calls_per_worker=12
        host.start()
        host.deadline=min(host.deadline,hard_deadline)
        for lane in pool.lanes:
            with lane.condition:
                if lane.active or lane.logical_active:raise RuntimeError('Planner calls not settled before worker phase')
                lane.deadline=host.deadline
                lane.condition.notify_all()
        pool.deadline=host.deadline
        host.client=pool
        host.simulator=MemoryGatedSimulationLock('/tmp/kimirax-short-task09-sim.lock',lambda:host.deadline,
                                               root/'worker-simulation-waits.jsonl')
        sessions=[Session(host.root/h['id'],host,h) for h in plan['hypotheses']]
        if time.time()>=ADMISSION:raise TimeoutError('Dropped: worker setup missed14:00 Pacific')
        P.write_json(root/'protocol.json',{'kind':'shortened_exploratory_planner_led','task':9,
            'planner_input':'Public task instruction and six unchanged starter entry files only',
            'prior_designs_or_results_supplied':False,'human_hints_favouring_specific_methods':False,
            'planning_seconds_limit':180,'worker_seconds':900,'workers':6,'probes_per_worker':3,
            'logical_calls_per_worker':12,'selection_weights':F.SELECTION_WEIGHTS,
            'selection_scales':F.SELECTION_SCALES,'selection_implementation':'task09_swarm_followup.partial_score',
            'estimated_spend_guard_usd_including_planning':10,'authorized_accounts':3,
            'api_slots_per_account':args.api_concurrency,
            'output_tpm_per_account':args.output_tpm,'queue_wait_counts_against_fixed_dev_window':True,
            'all_workers_stop_before_private_selection_evaluation':True,
            'worker_private_evaluation_after_selection_is_analysis_only':True,'started_at':time.time(),
            'hard_stop_at':'2026-11-02T14:59:30-08:00','task07_untouched':True})
        workers_started=time.monotonic()
        summaries=F.run_workers(host,pool,sessions,6)
        P.write_json(root/'worker-summaries.json',summaries)
        host.rehearse_remaining(sessions)
        for session,row in zip(sessions,summaries):
            row['first_assigned_probe_measured']=session.first_assigned_probe_measured
            row['first_assigned_probe_origin']=session.first_assigned_probe_origin
            row['probes_including_rehearsal']=session.tests
        P.write_json(root/'worker-summaries.json',summaries)
        selected=host.select()
        P.write_json(root/'selection-frozen.json',{'selection':selected,'all_workers_stopped':True,'at':time.time()})
        result.update(status='selected',selected_worker=selected['worker'],
            development_seconds=time.monotonic()-workers_started,new_development_probes=host.tests,
            assigned_first_test_compliance={s.worker_id:s.first_assigned_probe_measured and
                s.first_assigned_probe_origin=='worker_requested' for s in sessions})
    except Exception as error:
        result.update(status='development_failed',error=f'{type(error).__name__}: {error}')
    finally:
        pool.close()
        result['api']=pool.summary()
        result['throughput_per_worker_and_planner']=throughput(root,result['api'])
        P.write_json(root/'api-summary.json',result['api'])
        P.write_json(root/'throughput-per-worker.json',result['throughput_per_worker_and_planner'])
    if selected is not None:
        try:
            with MemoryGatedSimulationLock('/tmp/kimirax-short-task09-sim.lock',lambda:hard_deadline,
                                           root/'selected-final-waits.jsonl'):
                reward=Q.evaluate(host.root/'selected',host.root/'final',final=True,
                                  timeout=min(900,max(.1,hard_deadline-time.monotonic())))
            result.update(status='completed',score=round(reward['reward']*100,2),official_reward=reward)
            P.write_json(root/'selected-result.json',result)
            notify(f"Task09 planner selected result: score={result['score']}, API estimate=${result['api']['estimated_cost_usd']:.4f}. Worker private analysis follows; no worker sees these results.")
            result['worker_private_analysis']=analyze_worker_finals(host.root,sessions,selected['manifest'],STOP,
                Path('/tmp/kimirax-short-task09-sim.lock'))
        except Exception as error:
            result.update(status='selected_or_analysis_evaluation_failed',evaluation_error=f'{type(error).__name__}: {error}')
        result['operator_comparisons_only']={'previous_swarm':19.19,'unchanged_starter':21.10,'previous_astra':31.10}
    result['total_seconds']=time.monotonic()-began
    P.write_json(root/'summary.json',result)
    notify(f"Task09 planner run finished: status={result['status']}, selected score={result.get('score')}, API estimate=${result['api']['estimated_cost_usd']:.4f}; report {root/'summary.json'}.")


if __name__=='__main__':main()
