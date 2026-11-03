"""Final DeepSeek retry with warm starts, shared public tools and asynchronous validation."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import paired_benchmarks as B
import task08_pilot as P
from short_pilot_pool import AccountPool
from task08_swarm_short import primary_keys

PREP=P.ROOT/'outputs/final-deepseek-preparation-20261102'
ROLE6=(('precision_integrator','Lead integration: preserve a measured baseline, combine only validated precision improvements.'),
 ('validation_coordinator','Coordinate heldout validation, numerical accuracy and regression evidence; stop speculative rewrites.'),
 ('pixel_calibration','Correct pixel centers, camera projection and table-plane mapping; quantify millimeter bias.'),
 ('template_precision','Improve compact canonical geometry and subpixel template registration.'),
 ('yaw_precision','Improve full-angle yaw precision and resolve asymmetric rotations.'),
 ('robust_fit','Improve edge fitting and outlier rejection without shifting the object origin.'),
 ('shape_identity','Improve T/C/F identity consistency using public silhouettes; keep exact output labels.'),
 ('translation_bias','Measure and reduce per-shape origin offsets using public examples and calibration.'),
 ('occlusion','Improve partially occluded silhouettes and pusher rejection.'),
 ('causal_tracking','Improve temporal state estimation, recovery and reset correctness.'),
 ('depth_reliability','Use depth only when it improves geometry; reject noisy depth and preserve RGB precision.'),
 ('runtime','Reduce expensive search while preserving measured pose precision.'),
 ('generalization','Test distinct public validation seeds, unseen rotations and frame boundaries.'),
 ('contracts','Verify self-contained exports, deterministic reset, finite results and no missing dependencies.'))
ROLE8=(('control_integrator','Lead integration: combine measured controller improvements while preserving a loadable robot.'),
 ('validation_coordinator','Coordinate measured three-arm validation and preserve a safe regression baseline.'),
 ('base_servo','Correct holonomic base feedback, wheel signs and actual position convergence before arm entry.'),
 ('panda_reach','Improve Panda IK, approach path and shelf hold using the correctly assembled public scene.'),
 ('ur5e_reach','Improve UR5e IK and path using the public adapter and actual mount transform.'),
 ('xarm_reach','Improve xArm7 IK and path using the public adapter and actual mount transform.'),
 ('collision_path','Reduce shelf contacts with staged approach, intermediate waypoints and explicit clearance.'),
 ('loaded_hold','Improve deterministic gravity compensation and loaded one-second holding.'),
 ('ik_search','Improve multi-start joint-limited IK with public collision-aware numerical search.'),
 ('base_goal','Optimize base approach pose, shelf standoff and arm reach without frame instability.'),
 ('stability','Improve low battery placement, support margin and dynamic stability while preserving wheel stock.'),
 ('frame_efficiency','Improve frame stock efficiency and validity without degrading measured control.'),
 ('motion_timing','Improve settling, base convergence, unfolding, shelf entry and hold phase transitions.'),
 ('contracts','Verify all3 arms, full control order, self-contained exports and exact stock components.'))
ROLES={6:ROLE6,8:ROLE8}


def copy_exports(source,destination):
    destination.mkdir(parents=True,exist_ok=True);destination.chmod(0o777)
    for path in source.rglob('*'):
        rel=path.relative_to(source)
        if '__pycache__' in rel.parts or path.name in ('MUJOCO_LOG.TXT',) or path.suffix in ('.pyc','.log'):continue
        if path.is_symlink():raise ValueError('Submission links forbidden')
        target=destination/rel
        if path.is_dir():target.mkdir(parents=True,exist_ok=True);target.chmod(0o777)
        elif path.is_file():
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());target.chmod(0o666)


class Host:
    def __init__(self,root,task,slots,executor):
        self.root,self.task,self.slots,self.executor=root,task,slots,executor
        root.mkdir(parents=True);self.lock=threading.RLock()
        self.entries,self.findings,self.guidance=[],[],{}
        self.cache,self.pending={},{};self.best=None;self.deadline=None;self.start_at=None
        self.closed=False;self.milestones={};self.public_checker=None
        self.engineering_source=P.ROOT/'scripts';self.starter=PREP/f'task{task:02d}/starter'
        self.training=PREP/f'task{task:02d}/data'

    def phase(self):
        elapsed=time.monotonic()-self.start_at if self.start_at else 0
        return ('baseline_and_diagnosis' if elapsed<180 else 'targeted_improvement' if elapsed<600 else
            'integration_and_validation' if elapsed<780 else 'freeze_best_measured')

    def status(self):
        with self.lock:
            preferred=self.public_checker.best if self.public_checker and self.public_checker.best else self.best
            return {'seconds_remaining':max(0,int(self.deadline-time.monotonic())), 'phase':self.phase(),
                'best':None if not preferred else {k:preferred[k] for k in ('index','worker','score')},
                'best_proxy':None if not self.best else {k:self.best[k] for k in ('index','worker','score')},
                'probes':[{k:r[k] for k in ('index','worker','status','score','error') if k in r} for r in self.entries[-16:]],
                'findings':self.findings[-12:],'guidance':self.guidance,'milestones':self.milestones,
                'original_public_diagnostics':[] if self.public_checker is None else
                    [{k:r[k] for k in ('index','worker','status','mean_translation_mm','mean_rotation_deg','score','error') if k in r}
                        for r in self.public_checker.records]}

    def test(self,session,hypothesis):
        with self.lock:
            if time.monotonic()>=self.deadline or self.closed:raise TimeoutError('Development ended')
            if session.tests>=8:raise ValueError('Eight validation jobs per worker')
            if sum(r['status'] in ('queued','running') for r in self.entries)>=4:
                return {'busy':True,'instruction':'Validation queue has four jobs. Work on your assigned component, read findings, or integrate the current measured best.'}
            version=session.version_count;session.version_count+=1
            bundle=session.root/'versions'/str(version);manifest=session.freeze(bundle)
            key=manifest['sha256']
            if key in self.cache:return {'cached':True,**self.cache[key]['reply']}
            if key in self.pending:return {'queued':True,'index':self.pending[key],'identical_candidate_already_testing':True}
            index=len(self.entries);session.tests+=1
            row={'index':index,'worker':session.worker_id,'version':version,'status':'queued','hypothesis':hypothesis[:2200]}
            self.entries.append(row);self.pending[key]=index;P.write_json(self.root/'probe-ledger.json',self.entries)
            self.executor.submit(self.measure,session,row,bundle,manifest,key)
            return {'queued':True,'index':index,'frozen_sha256':key,'instruction':'The frozen snapshot is testing independently. Continue focused edits or integration; inspect status later.'}

    def measure(self,session,row,bundle,manifest,key):
        began=time.monotonic()
        try:
            with self.slots:
                remaining=self.deadline-time.monotonic()
                if remaining<8:raise TimeoutError('Insufficient development time for validation')
                with self.lock:row['status']='running'
                code=(f'import json,task{self.task:02d}_engineering as engineering\n'
                    f'result=engineering.diagnose("/versions/{row["version"]}",split="validation")\n'
                    'print("ENGINEERING_RESULT="+json.dumps(result,allow_nan=False))')
                out=session.execute(code,timeout=min(140,remaining),acquire=False)
                if out['returncode']:raise ValueError(out['stderr'][-1500:] or out['stdout'][-1500:])
                lines=[x[len('ENGINEERING_RESULT='):] for x in out['stdout'].splitlines() if x.startswith('ENGINEERING_RESULT=')]
                if not lines:raise ValueError('Diagnostic returned no structured report')
                report=json.loads(lines[-1]);score=float(report['selection_score'])
                if not __import__('math').isfinite(score):raise ValueError('Nonfinite selection score')
                from services.controller.storage import verify
                verify(bundle,manifest)
                reply={'index':row['index'],'score':score,'report':report}
                record={'index':row['index'],'worker':session.worker_id,'score':score,'bundle':bundle,'manifest':manifest,'reply':reply}
                with self.lock:
                    self.cache[key]=record
                    if self.best is None or score>self.best['score']:self.best=record
                    row.update(status='measured',score=score,report=report,wall_seconds=time.monotonic()-began)
                print(json.dumps({'event':'measured','task':self.task,'index':row['index'],'worker':session.worker_id,'score':score}),flush=True)
        except Exception as error:
            with self.lock:row.update(status='error',error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-began)
        finally:
            with self.lock:
                self.pending.pop(key,None);P.write_json(self.root/'probe-ledger.json',self.entries)

    def select(self):
        chosen=self.public_checker.best if self.public_checker and self.public_checker.best else self.best
        if chosen is None:raise RuntimeError(f'Task{self.task}: no validated candidate; final evaluation not launched')
        manifest=B.freeze(chosen['bundle'],self.root/'selected',self.task)
        result={'worker':chosen['worker'],'index':chosen['index'],'public_selection_score':chosen['score'],
            'selection_basis':'original_public_aggregate' if chosen.get('public_metric_score') is not None else 'heldout_public_proxy',
            'original_public_metric_score':chosen.get('public_metric_score'),
            'used_private_scores':False,'manifest':manifest}
        P.write_json(self.root/'selection.json',result);return result


class PublicSix:
    def __init__(self,root,host):
        self.root,self.host=root,host;root.mkdir(parents=True)
        self.staging=root/'candidate';copy_exports(host.starter,self.staging)
        self.name='final-ds-public-six';self.records=[];self.best=None;self.used=set()
        self.stop=threading.Event();self.thread=None
        B.checked(['docker','run','-d','--name',self.name,'--network','none','--cpus','4','--memory','6g',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
            '-v',f'{self.staging.resolve()}:/logs/artifacts',B.IMAGES[6]])

    def start(self):
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        while not self.stop.is_set() and time.monotonic()<self.host.deadline-15:
            if len(self.records)>=6:return
            schedule=(0,180,360,540,690,795)
            if time.monotonic()-self.host.start_at<schedule[len(self.records)]:
                self.stop.wait(1);continue
            with self.host.lock:
                choices=sorted((v for k,v in self.host.cache.items() if k not in self.used),key=lambda v:v['score'],reverse=True)
            if not choices:self.stop.wait(1);continue
            chosen=choices[0];key=chosen['manifest']['sha256'];self.used.add(key)
            record={'index':chosen['index'],'worker':chosen['worker'],'status':'running'}
            self.records.append(record);began=time.monotonic()
            try:
                for p in self.staging.iterdir():
                    if p.is_dir():shutil.rmtree(p)
                    else:p.unlink()
                copy_exports(chosen['bundle'],self.staging)
                remaining=self.host.deadline-time.monotonic()
                out=B.checked(['docker','exec','--user','1000:1000','-w','/workspace',self.name,
                    'python3','-c','import json;from harness.evaluator import evaluate;print(json.dumps(evaluate()))'],timeout=max(1,min(180,remaining)))
                report=json.loads(out.stdout.strip().splitlines()[-1])
                if not report.get('ok') or report.get('errors'):raise ValueError(str(report.get('errors',report))[:1000])
                trans=sum(w*report[n]['translation_error_m']['mean']*1000 for n,w in [('single_frames',.35),('push_frames',.65)])
                yaw=sum(w*report[n]['rotation_error_rad']['mean']*180/__import__('math').pi for n,w in [('single_frames',.35),('push_frames',.65)])
                valid=sum(w*report[n]['valid_fraction'] for n,w in [('single_frames',.35),('push_frames',.65)])
                score=valid/(1+trans+.25*yaw)
                record.update(status='measured',mean_translation_mm=trans,mean_rotation_deg=yaw,score=score,report=report)
                if self.best is None or score>self.best['public_metric_score']:self.best={**chosen,'public_metric_score':score}
                with self.host.lock:self.host.findings.append({'worker':'shared_public_evaluator','notes':json.dumps({k:record[k] for k in ('index','worker','mean_translation_mm','mean_rotation_deg','score')})})
                print(json.dumps({'event':'original_public6','index':chosen['index'],'translation_mm':trans,'rotation_deg':yaw}),flush=True)
            except Exception as error:record.update(status='error',error=f'{type(error).__name__}: {error}')
            record['wall_seconds']=time.monotonic()-began
            P.write_json(self.root/'diagnostics.json',self.records)

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=2)
        subprocess.run(['docker','rm','-f',self.name],capture_output=True,timeout=10)
        if self.thread:self.thread.join(timeout=3)


class Session(B.Session):
    def __init__(self,root,host,name,brief,planner=False):
        self.root,self.host,self.worker_id,self.brief,self.planner=root,host,name,brief,planner
        root.mkdir(parents=True);self.workspace=root/'workspace';self.workspace.mkdir()
        self.candidate=root/'candidate';copy_exports(host.starter,self.candidate)
        (root/'versions').mkdir()
        self.tests=self.version_count=self.analysis_count=0
        self.history=[];self.lock=threading.RLock();self.submitted=None;self.deadline=None
        self.condition='deepseek-rax-final';self.mcp_namespace=f'task{host.task:02d}'
        self.container='final-ds-'+name.replace('_','-')
        module=host.engineering_source/f'task{host.task:02d}_engineering.py'
        command=['docker','run','-d','--name',self.container,'--network','none','--cpus','3','--memory','3g',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1',
            '-e','PYTHONPATH=/workspace:/engineering','--entrypoint','sleep',
            '-v',f'{self.candidate.resolve()}:/candidate','-v',f'{self.candidate.resolve()}:/logs/artifacts',
            '-v',f'{(root/"versions").resolve()}:/versions:ro',
            '-v',f'{module.resolve()}:/engineering/{module.name}:ro']
        if host.task==8:
            companion=host.engineering_source/'task08_engineered_controller.py'
            command+=['-v',f'{companion.resolve()}:/engineering/{companion.name}:ro']
        data=host.training
        if data.exists():command+=['-v',f'{data.resolve()}:/training:ro']
        command += [B.IMAGES[host.task],'infinity'];B.checked(command)
        self.prompt=self.make_prompt();(root/'prompt.txt').write_text(self.prompt)

    def freeze(self,destination):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='clean-export-',dir=str(self.root)) as temporary:
            clean=Path(temporary);copy_exports(self.candidate,clean)
            return B.freeze(clean.resolve(),destination.resolve(),self.host.task)

    def make_prompt(self):
        task=self.host.task
        notes=PREP/f'task{task:02d}/agent-guide.md'
        guide=notes.read_text() if notes.exists() else ''
        instruction=((B.TASK6 if task==6 else B.TASK8)/'instruction.md').read_text()
        return f'''You are DeepSeek {self.worker_id}: {self.brief}
This is the final 900-second development pass. Fourteen workers per task have complementary roles: two integrators/coordinators and twelve specialists.
You start from an engineered, runnable candidate based on our own prior DeepSeek work and public inputs. Preserve it, diagnose it, improve your assigned bottleneck, and integrate measured improvements. Do not rebuild from scratch.
The mounted public engineering module is task{task:02d}_engineering, importable in run_analysis. Public numerical tools and cached training data are supplied. Read their guide below.
Call run_analysis for a focused quick diagnosis by your second response. Request asynchronous test_candidate after a useful concrete revision, then keep working while it runs.
Stages: 0-180s diagnose baseline and make one narrow improvement; 180-600s improve assigned component; 600-780s integrators combine validated work; 780-900s preserve the best tested candidate and submit.
A test freezes a snapshot and runs heldout public validation in the background. It returns a job index immediately. Use status for results. Never spend the remaining time waiting on a queued long test.
All workers can read measured peer source using read_peer(index,path) and clone a measured complete candidate with clone_candidate(index). Prefer targeted merges over repeated rebuilding. Coordinators must publish concise instructions and track milestones.
Use report_milestone with an evidence-based summary and next action; use share_finding for numerical facts that others need. Do not publish untested guesses as fixes.
run_analysis is isolated Python, at most45seconds, no network. Current directory /candidate, exports /candidate and /logs/artifacts. Use /tmp for simulations/scratch. Frozen versions /versions are read-only. Public assets /workspace and engineering tools /engineering are read-only inputs. Cached Task06 public images/labels are /training.
Only supplied MCP tools. No other models, new agents, Astra code, prior competitor artifacts, private verifier/source, hidden seeds or final-evaluation access. The final official verifier runs once after both tasks freeze.
Keep exports self-contained under32MiB and allTask08 controller helpers inside controller.py. No bytecode caches, logs, symlinks or temporary datasets in exports. Your Python does not write bytecode.
Prioritize real pose accuracy or successful shelf control, not a merely loadable artifact. The host chooses the best measured heldout diagnostic, not the latest edit.
Team worker IDs and roles: {json.dumps([(f'f{task}_{role}',brief) for role,brief in ROLES[task]])}
Engineering guide:
{guide}
Original task (development allowance replaced by900seconds):
{instruction}
'''

    def definitions(self):
        definitions=super().definitions()
        for d in definitions:
            if d['name']=='test_candidate':d['description']='Freeze and queue an asynchronous heldout public validation; returns immediately.'
        extra=[('read_peer','Read source from a measured peer snapshot.',{'index':{'type':'integer'},'path':{'type':'string'}},['index','path']),
            ('read_engineering','Read the public numerical helper source.',{'offset':{'type':'integer'},'length':{'type':'integer'}},[]),
            ('report_milestone','Record measured progress and your next action.',{'summary':{'type':'string'},'next_action':{'type':'string'}},['summary','next_action'])]
        for name,desc,props,required in extra:definitions.append({'name':name,'description':desc,'inputSchema':{'type':'object','properties':props,'required':required,'additionalProperties':False}})
        return definitions

    def dispatch(self,name,args):
        if self.submitted is not None or time.monotonic()>=self.deadline:raise ValueError('Development ended')
        if name=='coordinate' and self.planner:
            worker=args['worker']
            if not worker.startswith(f'f{self.host.task}_'):worker=f'f{self.host.task}_{worker}'
            valid={f'f{self.host.task}_{role}' for role,_ in ROLES[self.host.task]}
            if worker not in valid:raise ValueError('Unknown team worker ID')
            with self.host.lock:self.host.guidance[worker]={'from':self.worker_id,'instruction':args['instruction'][:2500]}
            return {'sent':True,'worker':worker}
        if name=='submit' and self.planner and self.deadline-time.monotonic()>120:
            return {'deferred':True,'instruction':'Integrator remains active through integration. Read peer tests/source, coordinate focused fixes, and preserve a measured candidate; submit during the final120seconds.'}
        if name=='test_candidate':return self.host.test(self,args['hypothesis'])
        if name=='status':return {**self.host.status(),'worker':self.worker_id,'own_tests':self.tests}
        if name=='report_milestone':
            with self.host.lock:
                self.host.milestones[self.worker_id]={'summary':args['summary'][:1600],'next_action':args['next_action'][:800]}
                P.write_json(self.host.root/'milestones.json',self.host.milestones)
            return {'recorded':True,'phase':self.host.phase()}
        if name=='read_engineering':
            text=(self.host.engineering_source/f'task{self.host.task:02d}_engineering.py').read_text()
            offset=max(0,int(args.get('offset',0)));length=min(30000,max(1,int(args.get('length',20000))))
            return {'text':text[offset:offset+length],'total_characters':len(text)}
        if name=='read_peer':
            with self.host.lock:
                chosen=next(v for v in self.host.cache.values() if v['index']==args['index'])
                target=chosen['bundle']/P.safe_relative(args['path'])
                if not target.resolve().is_relative_to(chosen['bundle'].resolve()) or target.suffix not in ('.py','.json','.xml','.txt','.md'):raise ValueError('Read peer text exports only')
                return {'text':target.read_text()[:50000]}
        if name=='clone_candidate':
            with self.host.lock:chosen=next(v for v in self.host.cache.values() if v['index']==args['index'])
            for p in self.candidate.iterdir():
                if p.is_dir():shutil.rmtree(p)
                else:p.unlink()
            copy_exports(chosen['bundle'],self.candidate)
            return {'cloned':args['index'],'worker':chosen['worker'],'score':chosen['score']}
        return super().dispatch(name,args)

    def tool(self,name,arguments):
        with self.lock:
            record={'name':name,'arguments':arguments}
            try:
                result=self.dispatch(name,arguments)
                if isinstance(result,dict):
                    with self.host.lock:
                        result['team_update']={'phase':self.host.phase(),
                            'seconds_remaining':max(0,int(self.deadline-time.monotonic())),
                            'instruction':self.host.guidance.get(self.worker_id),
                            'best_index':self.host.status()['best']['index'] if self.host.status()['best'] else None}
                reply={'content':[{'type':'text','text':json.dumps(result,allow_nan=False)}],'isError':False}
            except Exception as error:reply={'content':[{'type':'text','text':str(error)}],'isError':True}
            record['reply']=reply;self.history.append(record);P.write_json(self.root/'tools.json',self.history)
            return reply

    def execute(self,code,timeout=45,acquire=True):
        if not isinstance(code,str) or len(code)>100000:raise ValueError('Use <=100000 code characters')
        if acquire:self.host.slots.acquire()
        try:
            remaining=self.deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Development ended while queued')
            timeout=max(.1,min(timeout,remaining))
            command=['docker','exec','-i','--user','1000:1000','--workdir','/candidate',self.container,
                'timeout','-k','2',str(max(1,int(timeout))),'python3','-']
            completed=subprocess.run(command,input=code,capture_output=True,text=True,timeout=timeout+3)
            return {'returncode':completed.returncode,'stdout':completed.stdout[-42000:],'stderr':completed.stderr[-4000:]}
        finally:
            if acquire:self.host.slots.release()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=900);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source=root/'harness-source';source.mkdir()
    for name in ('deepseek_final_pair.py','paired_benchmarks.py','task06_engineering.py','task08_engineering.py',
        'task08_engineered_controller.py','short_pilot_pool.py','throughput_paced_client.py','task08_pilot.py',
        'task08_swarm.py','task08_swarm_client.py','task08_swarm_short.py','task09_team15_supervise.py'):
        shutil.copy2(P.ROOT/'scripts'/name,source/name)
    manifests={}
    for task in (6,8):
        destination=root/'starting-candidates'/f'task{task:02d}';destination.parent.mkdir(exist_ok=True)
        manifests[str(task)]=B.freeze(PREP/f'task{task:02d}/starter',destination,task)
        guide=PREP/f'task{task:02d}/agent-guide.md'
        if guide.exists():shutil.copy2(guide,source/f'task{task:02d}-agent-guide.md')
    if (PREP/'task06/data').exists():shutil.copytree(PREP/'task06/data',root/'training/task06')
    sessions=[];containers=[];pool=None;public6=None
    slots=threading.BoundedSemaphore(4)
    with ThreadPoolExecutor(max_workers=4) as tests:
        hosts={t:Host(root/f'task{t:02d}/deepseek',t,slots,tests) for t in (6,8)}
        for task,host in hosts.items():
            host.engineering_source=source;host.starter=root/'starting-candidates'/f'task{task:02d}'
            host.training=root/'training'/f'task{task:02d}'
        try:
            for task,host in hosts.items():
                for i,(role,brief) in enumerate(ROLES[task]):
                    s=Session(host.root/role,host,f'f{task}_{role}',brief,planner=i<2)
                    sessions.append(s);containers.append(s.container)
            public6=PublicSix(root/'shared-public-task06',hosts[6]);hosts[6].public_checker=public6;containers.append(public6.name)
            start=time.monotonic();wall_start=time.time();deadline=start+args.seconds
            for host in hosts.values():host.deadline=deadline;host.start_at=start
            for s in sessions:s.start()
            pool=AccountPool(root/'shared-api',primary_keys(),worker_ids=[s.worker_id for s in sessions],
                deadline=deadline,max_workers=28,max_concurrent=3,logical_requests_per_worker=100,
                max_calls=8400,per_worker_calls=300,max_tokens=150_000_000,max_output_tokens=8192,max_cost=10,
                prompt_tpm=500000,uncached_prompt_tpm=250000,generated_tpm=60000,
                prompt_bytes_per_token=2.9,start_stagger_seconds=.25)
            primary={'f6_shape_identity','f8_frame_efficiency','f8_contracts'};other=0
            with pool.lock:
                for s in sessions:
                    if s.worker_id in primary:pool.assignments[s.worker_id]=0
                    else:pool.assignments[s.worker_id]=1+other%2;other+=1
                pool._persist()
            P.write_json(root/'process.json',{'pid':os.getpid(),'started_at':wall_start,
                'development_deadline_at':wall_start+args.seconds,'total_deadline_at':wall_start+args.seconds+930,'containers':containers})
            P.write_json(root/'protocol.json',{'tasks':['task06-rgb-depth','task08'],'development_seconds':args.seconds,
                'deepseek_agents_per_task':14,'coordinators_per_task':2,'fireworks_accounts':3,
                'concurrency_per_account':3,'combined_estimated_api_cost_guard_usd':10,
                'warm_start':'Our own prior DeepSeek artifacts improved with public numerical tools and cached public examples.',
                'competitor_solution_supplied':False,'asynchronous_frozen_public_validation':True,
                'starting_candidate_manifests':manifests,'shared_original_task06_diagnostic_calls':6,
                'final_evaluations_per_task':1,'final_timeout_seconds':900,
                'astra_references':{'6':'benchmarks/task06-sota.json','8':'benchmarks/task08-sota.json'}})
            for task,host in hosts.items():
                seed=next(s for s in sessions if s.host is host);host.test(seed,'Validate engineered warm-start before any agent edits')
            public6.start()
            summaries={}
            def worker(s):
                try:return pool.run_worker(s,s.worker_id,100)
                except Exception as error:return {'worker_id':s.worker_id,'error':f'{type(error).__name__}: {error}'}
            with ThreadPoolExecutor(max_workers=28) as executor:
                futures={executor.submit(worker,s):s.worker_id for s in sessions}
                for future in as_completed(futures):
                    summaries[futures[future]]=future.result();P.write_json(root/'agent-summaries.json',summaries)
            for host in hosts.values():host.closed=True
            pool.close(wait_timeout=5);api=pool.summary()
            tests.shutdown(wait=True,cancel_futures=True)
            public6.close()
            selections={t:h.select() for t,h in hosts.items()}
            P.write_json(root/'all-frozen.json',{'at':time.time(),'selections':selections,'all_model_agents_stopped':True})
            for name in containers:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
            containers=[]
            results={}
            def score(task,host):
                began=time.monotonic();out=host.root/'final'
                try:
                    reward=B.final_score(task,host.root/'selected',out,timeout=900)
                    result={'status':'completed','score':round(100*float(reward['reward']),6),'official_reward':reward}
                except Exception as error:result={'status':'final_failed','score':None,'error':f'{type(error).__name__}: {error}'}
                result['final_seconds']=time.monotonic()-began;P.write_json(host.root/'result.json',result);return result
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures={executor.submit(score,t,h):t for t,h in hosts.items()}
                for future in as_completed(futures):
                    results[str(futures[future])]=future.result();P.write_json(root/'results.json',results)
            P.write_json(root/'summary.json',{'status':'finished','results':results,'api':api,'agent_summaries':summaries,
                'wall_seconds':time.monotonic()-start,'finished_at':time.time()})
            print(json.dumps(results),flush=True)
        finally:
            if pool is not None:pool.close(wait_timeout=3)
            if public6 is not None:public6.close()
            for name in containers:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)


if __name__=='__main__':main()
