"""Parallel Task06 RGB-D and Task08: raw Astra against engineered DeepSeek teams."""

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import task08_pilot as P
import task08_swarm as S8
from short_pilot_pool import AccountPool
from task08_swarm_short import primary_keys
from task09_astra_raw import usage_audit

TASK6=P.REPO/'tasks/task06/rgb-depth'
TASK8=P.REPO/'tasks/task08'
ROLES6=(('perception_coordinator','Coordinate segmentation, canonical shapes and calibration; integrate measured results.'),
 ('tracking_coordinator','Coordinate tracking, occlusion recovery and runtime efficiency; integrate measured candidates.'),
 ('segmentation','Build robust red-object segmentation with shadow, stick occlusion and noise handling.'),
 ('depth_geometry','Backproject depth through K and T_cam_table; recover planar position and top-plane geometry.'),
 ('homography','Use camera calibration and the known 40 mm block height for accurate RGB-to-table projection.'),
 ('canonical_templates','Learn pose-aligned T/C/F templates from public rendered examples and noisy pose labels.'),
 ('shape_classification','Distinguish shape IDs 0=T,1=C,2=F from red silhouettes and topology; stabilize identity across an episode.'),
 ('orientation','Recover full asymmetric yaw, avoiding 90/180-degree ambiguity and aligning conventions to public labels.'),
 ('template_registration','Fit canonical masks or point sets over rotation and translation using scipy/OpenCV.'),
 ('occlusion_tracking','Keep causal pose and identity stable while the pusher arm crosses the object.'),
 ('calibration','Fit systematic camera/centroid offsets from public noisy labels; preserve unseen-pose generalization.'),
 ('speed','Make CPU update comfortably faster than 100 ms; preserve identity and full-angle orientation.'),
 ('robustness','Test missing depth, partial red masks, reset behavior and tracking recovery.'),
 ('integration','Combine measured geometric pose estimation, shape identity and causal tracking into one export.'))
ROLES8=(('mechanical_coordinator','Coordinate a valid compact chassis using public stock components; integrate measured designs.'),
 ('control_coordinator','Coordinate universal controller, base approach and loaded shelf reaching; integrate measured designs.'),
 ('chassis','Assemble a connected aluminum-profile frame, exact stock wheels, battery and canonical Panda preview from scratch.'),
 ('mass_efficiency','Reduce recognized profile length and chassis mass while preserving connectivity and support.'),
 ('wheel_geometry','Assemble exact mecanum X-handedness, names, orientation and drive settings; connect frame through wheel centers.'),
 ('mount_height','Optimize arm mount height and support geometry for shelf reach and stability.'),
 ('battery_stability','Place the mandatory 10 kg battery low and legally to improve loaded static/dynamic margins.'),
 ('base_motion','Build stable holonomic base approach from the specified outside initial pose.'),
 ('arm_ik','Use model-based Jacobian IK with joint limits, reachable base positions and collision-conscious entry.'),
 ('trajectory','Stage settling, safe unfolding, base alignment, shelf entry and a stable one-second hold.'),
 ('panda_controller','Improve Panda shelf reach across all twelve targets and both payloads without breaking other arms.'),
 ('ur5e_controller','Improve UR5e shelf reach while accounting for the adapter yaw and different starting posture.'),
 ('xarm_controller','Improve xArm7 reach, unfolding and loaded holding while retaining a universal controller.'),
 ('contract','Validate names, stock component integrity, compiled dimensions, full controls and artifact self-containment.'))
ROLES={6:ROLES6,8:ROLES8}
ENTRY={6:('estimator.py',),8:('robot.xml','controller.py')}
PUBLIC={6:TASK6/'environment/agent',8:TASK8/'environment/assets'}
IMAGES={6:'rlebench-task06-rgb-depth-agent:dev',8:'rlebench-task08-agent:dev'}
ASTRA_IMAGES={6:'kimirax-task06-astra:dev',8:'kimirax-task08-astra:dev'}


def checked(command,timeout=30,**kwargs):
    return subprocess.run(command,check=True,capture_output=True,text=True,timeout=timeout,**kwargs)


def freeze(source,destination,task):
    from services.controller.storage import snapshot
    return snapshot(source.resolve(),destination.resolve(),max_bytes=32*1024**2,
        workspace_path=f'code/task{task:02d}')


def public6_score(report):
    if not report.get('ok') or report.get('errors'):
        raise ValueError('Public diagnostic failed: '+json.dumps(report)[:1800])
    out=[]
    for name,weight in [('single_frames',.35),('push_frames',.65)]:
        row=report[name]
        translation=float(row['translation_error_m']['mean'])
        rotation=float(row['rotation_error_rad']['mean'])
        valid=float(row['valid_fraction'])
        if not all(math.isfinite(x) and x>=0 for x in (translation,rotation,valid)):
            raise ValueError('Nonfinite public diagnostic')
        out.append(weight*valid*(.5*math.exp(-translation/.03)+.5*math.exp(-rotation/.20)))
    return sum(out)


class Host:
    def __init__(self,root,task,slots):
        self.root,self.task,self.slots=root,task,slots
        self.root.mkdir(parents=True)
        self.lock=threading.RLock()
        self.entries,self.findings,self.guidance=[],[],{}
        self.best,self.deadline=None,None
        self.cache={}

    def status(self):
        with self.lock:
            return {'seconds_remaining':max(0,int(self.deadline-time.monotonic())),
                'best':None if self.best is None else {'index':self.best['index'],'worker':self.best['worker'],'score':self.best['score']},
                'probes':[{k:r[k] for k in ('index','worker','status','score','error') if k in r} for r in self.entries[-24:]],
                'findings':self.findings[-10:],'guidance':self.guidance}

    def test(self,session,hypothesis):
        if session.tests>=4:raise ValueError('Four measured public diagnostics per worker')
        if time.monotonic()>=self.deadline:raise TimeoutError('Development ended')
        version=session.version_count;session.version_count+=1
        bundle=session.root/'versions'/str(version);bundle.parent.mkdir(exist_ok=True)
        manifest=freeze(session.candidate,bundle,self.task)
        with self.lock:
            key=manifest['sha256']
            if key in self.cache:return {'cached':True,**self.cache[key]['reply']}
            index=len(self.entries);session.tests+=1
            row={'index':index,'worker':session.worker_id,'version':version,'status':'queued','hypothesis':hypothesis[:3000]}
            self.entries.append(row);P.write_json(self.root/'probe-ledger.json',self.entries)
        began=time.monotonic()
        try:
            with self.slots:
                remaining=self.deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Probe queued past development deadline')
                row['status']='running'
                if self.task==6:
                    code='from harness.evaluator import evaluate\nimport json\nprint(json.dumps(evaluate()))'
                    out=session.execute(code,timeout=min(180,remaining),acquire=False)
                    if out['returncode']:raise ValueError(out['stderr'][-1600:])
                    report=json.loads(out['stdout'].strip().splitlines()[-1]);value=public6_score(report)
                    from services.controller.storage import verify
                    verify(session.candidate,manifest)
                else:
                    report=P.docker_evaluate(bundle,self.root/'probes'/str(index),timeout=min(240,remaining))
                    value=S8.partial_score(report)
                reply={'index':index,'score':value,'report':report if self.task==6 else S8.compact_report(report)}
                record={'index':index,'worker':session.worker_id,'score':value,'bundle':bundle,'manifest':manifest,'reply':reply}
                with self.lock:
                    self.cache[key]=record
                    if self.best is None or value>self.best['score']+1e-10:self.best=record
                    row.update(status='measured',score=value,report=reply['report'],wall_seconds=time.monotonic()-began)
                print(json.dumps({'event':'measured','task':self.task,'worker':session.worker_id,'index':index,'score':value}),flush=True)
                return reply
        except Exception as error:
            with self.lock:row.update(status='error',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            with self.lock:P.write_json(self.root/'probe-ledger.json',self.entries)

    def select(self,sessions):
        chosen=self.best
        if chosen is None:
            choices=[s for s in sessions if s.submitted is not None]
            if not choices:raise RuntimeError('No frozen candidate exists')
            first=choices[0];chosen={'worker':first.worker_id,'index':None,'score':None,
                'bundle':first.root/'frozen','manifest':first.submitted}
        manifest=freeze(chosen['bundle'],self.root/'selected',self.task)
        result={'worker':chosen['worker'],'index':chosen['index'],'public_score':chosen['score'],
            'used_private_scores':False,'manifest':manifest}
        P.write_json(self.root/'selection.json',result)
        return result


class Session(P.Session):
    def __init__(self,root,host,name,brief,planner=False):
        self.root,self.host,self.worker_id,self.brief,self.planner=root,host,name,brief,planner
        self.root.mkdir(parents=True)
        self.workspace=self.root/'workspace';self.workspace.mkdir()
        self.candidate=self.root/'candidate';self.candidate.mkdir(mode=0o777);self.candidate.chmod(0o777)
        self.tests,self.version_count,self.analysis_count=0,0,0
        self.history=[];self.lock=threading.RLock();self.submitted=None;self.deadline=None
        self.condition='deepseek-rax';self.mcp_namespace=f'task{host.task:02d}'
        self.container='paired-'+name.replace('_','-')
        if host.task==6:
            f=self.candidate/'estimator.py'
            f.write_text('class E:\n def reset(self): pass\n def update(self, *, rgb, K, T_cam_table, t, depth=None): return (0.,0.,0.,0)\ndef make_estimator(): return E()\n')
        else:
            f=self.candidate/'controller.py'
            f.write_text('class C:\n def reset(self, context, seed): self.ctrl=context["ctrl"]\n def act(self, observation): return self.ctrl\ndef make_controller(): return C()\n')
        f.chmod(0o666)
        args=['docker','run','-d','--name',self.container,'--network','none','--cpus','2',
            '--memory','3g' if host.task==6 else '1g','--memory-swap','12g' if host.task==6 else '2g',
            '-v',f'{self.candidate.resolve()}:/logs/artifacts','-v',f'{self.candidate.resolve()}:/candidate',IMAGES[host.task]]
        checked(args)
        self.prompt=self.make_prompt();(root/'prompt.txt').write_text(self.prompt)

    def start(self):self.deadline=self.host.deadline

    def make_prompt(self):
        task=self.host.task
        instruction=(TASK6 if task==6 else TASK8)/'instruction.md'
        recipe=('Use calibrated RGB-D geometry: segment red pixels, backproject depth using K, transform with T_cam_table, '
            'and use the known 40 mm height. Learn canonical masks/templates from public rendered examples and noisy pose labels; '
            'classify T/C/F by asymmetric geometry and map exactly 0=T,1=C,2=F. Planar origin can differ from the red centroid: fit this offset '
            'using pose labels. Fit full-angle orientation and use causal shape/pose tracking through occlusion. Shape identity must be correct, '
            'not merely position. Public clients: from harness import episodes; episodes.single_frame(11), episodes.push_episode(23,max_frames=40). '
            'Frame has .obs and noisy .gt=(x,y,theta), without a shape label. You may derive more public training seeds. Keep training arrays in /tmp '
            'or /workspace and export only code and compact numeric templates into /candidate. No private meshes or shape draws are readable.' if task==6 else
            'Build robot.xml FROM THE PUBLIC COMPONENTS, not from an oracle or prior solution. Copy the left mecanum XML for FL/RR and right for '
            'FR/RL, replace EXAMPLE exactly and preserve every wheel parameter and roller. Programmatically assemble XML with ElementTree or '
            'MuJoCo MjSpec; do not print thousands of roller lines. Compile in /workspace so public relative assets resolve, then export only robot.xml and controller.py to /candidate. Connect recognized aluminum profile boxes through all four drive centers. '
            'Mount the stock 10 kg battery low; arm_mount_site is a direct child of base. Attach the public Panda subtree with arm_ names and '
            'preserve meshdir assets/franka_emika_panda/assets. Compile with MuJoCo 3.5 before testing. Use scipy/MuJoCo Jacobians to plan base '
            'alignment and arm IK from actual context qpos; settle and unfold safely before entering the shelf. Return full actuator controls in '
            'context order, with a single controller handling Panda, UR5e and xArm7. Optimize frame stock, COM and support polygon before '
            'aggressive motion. No assembled chassis starter is supplied. Read stock component_catalog.json and component XML once, then build.')
        return f'''You are DeepSeek worker {self.worker_id}: {self.brief}
Fourteen DeepSeek agents share one 900-second development window on this benchmark: two coordinators and twelve executors.
Your tools are much stronger than the raw single-agent baseline: isolated numerical Python, shared measurements, candidate cloning and public tests.
Only the provided MCP tools are available; no additional agents, external filesystem, network or private evaluation access.
Implement a complete candidate immediately and request a test by response three. Use measurements and peer findings to revise and integrate.
run_analysis(code) executes Python for at most 45 seconds with numpy, scipy, OpenCV and MuJoCo in your persistent public environment.
It runs as the unprivileged agent user, cwd /candidate. /candidate and /logs/artifacts contain your exported submission.
Public assets and clients are under /workspace. Data and scratch scripts may persist in /tmp or /workspace; no private inputs are exposed.
Keep exports under 32 MiB. All files in /candidate are frozen together. Never place a symlink or scratch dataset there.
You have at most four measured public diagnostic calls, up to 80 numerical analyses and 100 logical model responses; time is the main limit.
Use compact Python code rather than large escaped file dumps. Group file edits across functions. Correct imports, interfaces and finite outputs first.
{recipe}
Use status for measured candidates, share_finding for concise evidence, and clone_candidate(index) to build on a tested peer result.
Coordinators should actively use coordinate(worker,instruction) to assign complementary work and integrate candidates.
The host selects the best measured public diagnostic; final private scoring occurs only after the agents stop and selection freezes.
Task06 public diagnostics report pose errors without shape-ID answers; validate shape mapping from the public images yourself.
Do not submit after mere source inspection. Submit a useful tested implementation while time remains. No emojis; minimal code comments.
All team roles: {json.dumps(ROLES[task])}
Original public task; its time budget is replaced by 900 seconds:
{instruction.read_text()}
'''

    def definitions(self):
        specs=[('list_files','List public text inputs and exported candidate files.',{},[]),
            ('read_file','Read a public/ or candidate/ text file.',{'path':{'type':'string'},'offset':{'type':'integer'},'length':{'type':'integer'}},['path']),
            ('write_file','Replace a relative candidate export file.',{'name':{'type':'string'},'content':{'type':'string'}},['name','content']),
            ('run_analysis','Execute isolated Python using public clients and your own candidate; max 45 seconds.',{'code':{'type':'string'}},['code']),
            ('test_candidate','Freeze and measure a public development diagnostic.',{'hypothesis':{'type':'string'}},['hypothesis']),
            ('status','Read measured peer evidence and time.',{},[]),
            ('share_finding','Share concise evidence or a hypothesis.',{'notes':{'type':'string'}},['notes']),
            ('clone_candidate','Copy a measured candidate into your own branch.',{'index':{'type':'integer'}},['index']),
            ('submit','Freeze your branch and end development.',{},[])]
        if self.planner:specs.append(('coordinate','Send concise guidance to another worker.',{'worker':{'type':'string'},'instruction':{'type':'string'}},['worker','instruction']))
        return [{'name':n,'description':desc,'inputSchema':{'type':'object','properties':props,'required':req,'additionalProperties':False}} for n,desc,props,req in specs]

    def freeze(self,destination):return freeze(self.candidate,destination,self.host.task)

    def execute(self,code,timeout=45,acquire=True):
        if not isinstance(code,str) or len(code)>100000:raise ValueError('Use <=100000 code characters')
        remaining=self.deadline-time.monotonic()
        timeout=max(.1,min(timeout,remaining))
        command=['docker','exec','-i','--user','1000:1000','--workdir','/candidate',self.container,
            'timeout','-k','2',str(max(1,int(timeout))),'python3','-']
        if acquire:self.host.slots.acquire()
        try:
            completed=subprocess.run(command,input=code,capture_output=True,text=True,timeout=timeout+5)
            return {'returncode':completed.returncode,'stdout':completed.stdout[-22000:],'stderr':completed.stderr[-4000:]}
        finally:
            if acquire:self.host.slots.release()

    def dispatch(self,name,args):
        if self.submitted is not None or time.monotonic()>=self.deadline:raise ValueError('Development ended')
        if name=='status':return {**self.host.status(),'worker':self.worker_id,'own_tests':self.tests}
        if name=='list_files':
            return {'candidate':[str(p.relative_to(self.candidate)) for p in self.candidate.rglob('*') if p.is_file()],
                'public':[str(p.relative_to(PUBLIC[self.host.task])) for p in PUBLIC[self.host.task].rglob('*') if p.is_file() and p.suffix in ('.py','.xml','.json','.txt','.md')]}
        if name=='read_file':
            path=P.safe_relative(args['path']);base={'public':PUBLIC[self.host.task],'candidate':self.candidate}.get(path.parts[0])
            if base is None or len(path.parts)<2:raise ValueError('Use public/ or candidate/')
            target=base.joinpath(*path.parts[1:]).resolve()
            if not target.is_relative_to(base.resolve()) or target.suffix not in ('.py','.xml','.json','.txt','.md'):raise ValueError('Read public/candidate text only')
            text=target.read_text();offset=max(0,int(args.get('offset',0)));length=min(50000,max(1,int(args.get('length',24000))))
            return {'text':text[offset:offset+length],'total_characters':len(text)}
        if name=='write_file':
            target=self.candidate/P.safe_relative(args['name'])
            if not target.resolve().is_relative_to(self.candidate.resolve()) or len(args['content'])>1024**2:raise ValueError('Invalid candidate file')
            target.parent.mkdir(parents=True,exist_ok=True);target.parent.chmod(0o777);target.write_text(args['content']);target.chmod(0o666)
            return {'saved':True}
        if name=='run_analysis':
            if self.analysis_count>=80:raise ValueError('Analysis allowance exhausted')
            self.analysis_count+=1;path=self.root/'analysis'/str(self.analysis_count);path.mkdir(parents=True)
            (path/'code.py').write_text(args['code']);result=self.execute(args['code']);P.write_json(path/'result.json',result)
            result['guidance']=self.host.guidance.get(self.worker_id);return result
        if name=='test_candidate':return self.host.test(self,args['hypothesis'])
        if name=='share_finding':
            with self.host.lock:self.host.findings.append({'worker':self.worker_id,'notes':args['notes'][:3500]});P.write_json(self.host.root/'findings.json',self.host.findings)
            return {'shared':True}
        if name=='coordinate' and self.planner:
            with self.host.lock:self.host.guidance[args['worker']]={'from':self.worker_id,'instruction':args['instruction'][:2500]}
            return {'sent':True}
        if name=='clone_candidate':
            with self.host.lock:
                index=args['index'];row=self.host.entries[index]
                chosen=next(r for r in self.host.cache.values() if r['index']==index)
                for p in list(self.candidate.iterdir()):
                    if p.is_dir():shutil.rmtree(p)
                    else:p.unlink()
                shutil.copytree(chosen['bundle'],self.candidate,dirs_exist_ok=True)
                for p in self.candidate.rglob('*'):p.chmod(0o777 if p.is_dir() else 0o666)
            return {'cloned':index,'worker':row['worker']}
        if name=='submit':
            self.submitted=self.freeze(self.root/'frozen');P.write_json(self.root/'frozen-manifest.json',self.submitted)
            return {'submitted':True,'sha256':self.submitted['sha256']}
        raise ValueError('Unknown tool')


class RawAstra:
    def __init__(self,root,task,name=None):
        self.root,self.task=root,task;root.mkdir(parents=True)
        self.artifacts=root/'artifacts';self.artifacts.mkdir(mode=0o777);self.artifacts.chmod(0o777)
        self.name=name or f'paired-raw-astra-{task}'
        command=['docker','run','-d','--name',self.name,'--cpus','4',
            '--memory','3g' if task==6 else '2g','--memory-swap','12g' if task==6 else '4g',
            '-v',f'{self.artifacts.resolve()}:/logs/artifacts',ASTRA_IMAGES[task]]
        checked(command)
        auth=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'auth.json'
        checked(['docker','exec',self.name,'mkdir','-p','/home/agent/.codex'])
        checked(['docker','cp',str(auth),f'{self.name}:/home/agent/.codex/auth.json'])
        from harness.codex_cli.configuration import write_config
        write_config(root/'config.toml',{'model':'gpt-6-astra','model_reasoning_effort':'max','approval_policy':'never',
            'sandbox_mode':'danger-full-access','web_search':'disabled','features':{k:False for k in (
                'multi_agent','multi_agent_v2','apps','plugins','skill_search','browser_use','computer_use','image_generation')}})
        checked(['docker','cp',str(root/'config.toml'),f'{self.name}:/home/agent/.codex/config.toml'])
        checked(['docker','exec',self.name,'chown','-R','1000:1000','/home/agent'])
        checked(['docker','exec',self.name,'chmod','600','/home/agent/.codex/auth.json'])
        instruction=(TASK6 if task==6 else TASK8)/'instruction.md'
        self.prompt='''Complete the following benchmark using ONE raw native Codex Astra agent and its normal coding tools.
You have 900 wall seconds for development. Save the required final submission under /logs/artifacts before the cutoff.
The original public workspace is /workspace. No prior solutions, custom engineering MCP tools, peer findings or oracle artifacts are supplied.
Use ordinary shell and Python tools to inspect public assets and develop your own solution. Do not spawn agents or call other models.
Do not access /opt/private, /tests, authentication files or external websites. Native model authentication is managed by Codex.
No hidden evaluation feedback is available. The host evaluates your saved final submission afterward.
No emojis; minimal comments. The original task follows; its time allowance is replaced by 900 seconds.
'''+instruction.read_text()
        (root/'prompt.txt').write_text(self.prompt)
        self.process=None

    def copy_usage(self):
        home=self.root/'agent-home';home.mkdir(exist_ok=True)
        subprocess.run(['docker','cp',f'{self.name}:/home/agent/.codex/sessions',str(home)],capture_output=True,timeout=10)
        return usage_audit(self.root)

    def stop_agent(self,signal_name):
        code='''import os,signal
sig=getattr(signal,os.environ['AGENT_SIGNAL'])
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 try:
  status=open('/proc/'+pid+'/status').read()
  uid=int(next(row for row in status.splitlines() if row.startswith('Uid:')).split()[1])
  if uid==1000:os.kill(int(pid),sig)
 except (OSError,StopIteration):pass
'''
        checked(['docker','exec','-e',f'AGENT_SIGNAL={signal_name}',self.name,'python3','-c',code],timeout=5)

    def run(self,deadline,start):
        command=['docker','exec','--user','1000:1000','--workdir','/workspace',
            '-e','HOME=/home/agent','-e','CODEX_HOME=/home/agent/.codex',self.name,
            'codex','exec','--skip-git-repo-check','--json','--',self.prompt]
        with (self.root/'agent.jsonl').open('w') as out,(self.root/'agent.stderr').open('w') as err:
            self.process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            next_usage=time.monotonic()+10
            while self.process.poll() is None and time.monotonic()<deadline:
                if time.monotonic()>=next_usage:self.copy_usage();next_usage=time.monotonic()+15
                time.sleep(.25)
            if self.process.poll() is None:
                self.stop_agent('SIGTERM')
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.stop_agent('SIGKILL')
                    self.process.kill();self.process.wait(timeout=5)
        usage=self.copy_usage()
        manifest=freeze(self.artifacts,self.root/'frozen',self.task)
        result={'agent_count':1,'model':'gpt-6-astra','raw_codex':True,'returncode':self.process.returncode,
            'seconds_after_common_start':time.monotonic()-start,'usage':usage,'manifest':manifest}
        P.write_json(self.root/'agent-summary.json',result)
        return result


def final_score(task,bundle,output,timeout=900):
    if task==8:return P.docker_evaluate(bundle,output,final=True,timeout=timeout)
    output.mkdir(parents=True,exist_ok=False)
    name='paired-final6-'+hashlib.sha256(str(output).encode()).hexdigest()[:10]
    command=['docker','run','--rm','--name',name,'--network','none','--cpus','4','--memory','5g','--memory-swap','12g',
        '-e','RLEBENCH_ARTIFACTS=/submission','-v',f'{bundle.resolve()}:/submission:ro',
        '-v',f'{output.resolve()}:/logs/verifier','rlebench-task06-rgb-depth-verifier:dev','/tests/test.sh']
    try:
        out=checked(command,timeout=timeout);(output/'container.log').write_text(out.stdout+out.stderr)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
        raise TimeoutError('Task06 final deadline; owned container stopped') from None
    return json.loads((output/'reward.json').read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    slots=threading.BoundedSemaphore(4)
    hosts={t:Host(root/f'task{t:02d}/deepseek',t,slots) for t in (6,8)}
    sessions=[];raw={};containers=[];pool=None
    try:
        for task,host in hosts.items():
            for i,(role,brief) in enumerate(ROLES[task]):
                s=Session(host.root/role,host,f'b{task}_{role}',brief,planner=i<2)
                sessions.append(s);containers.append(s.container)
            raw[task]=RawAstra(root/f'task{task:02d}/astra',task);containers.append(raw[task].name)
        start=time.monotonic();deadline=start+900;wall_start=time.time()
        for host in hosts.values():host.deadline=deadline
        for s in sessions:s.start()
        pool=AccountPool(root/'shared-api',primary_keys(),worker_ids=[s.worker_id for s in sessions],
            deadline=deadline,max_workers=28,max_concurrent=3,logical_requests_per_worker=100,
            max_calls=8400,per_worker_calls=300,max_tokens=150_000_000,max_output_tokens=8192,max_cost=10,
            prompt_tpm=500000,uncached_prompt_tpm=250000,generated_tpm=60000,
            prompt_bytes_per_token=2.9,start_stagger_seconds=.25)
        primary={'b6_shape_classification','b8_wheel_geometry','b8_contract'};other=0
        with pool.lock:
            for s in sessions:
                if s.worker_id in primary:pool.assignments[s.worker_id]=0
                else:pool.assignments[s.worker_id]=1+other%2;other+=1
            pool._persist()
        P.write_json(root/'process.json',{'pid':os.getpid(),'started_at':wall_start,'development_deadline_at':wall_start+900,
            'total_deadline_at':wall_start+1830,'containers':containers})
        P.write_json(root/'protocol.json',{'tasks':['task06-rgb-depth','task08'],'development_seconds':900,
            'astra_agents_per_task':1,'astra_mode':'raw native codex exec in public environment',
            'deepseek_agents_per_task':14,'deepseek_coordinators_per_task':2,'deepseek_custom_tools':True,
            'fireworks_accounts':3,'api_concurrency_per_account':3,'combined_estimated_api_cost_guard_usd':10,
            'max_output_tokens':8192,'global_simulation_slots':4,'parallel_development':True,
            'task08_start':'from-scratch public components; no assembled reference or oracle supplied',
            'task06_start':'public blind RGB-D client and estimator template; no meshes or shape labels supplied',
            'fixed_account_assignments':{k:v+1 for k,v in pool.assignments.items()},
            'routing':'Three short-context branches on the fallback-limited account; remaining branches split over the two accounts with higher reported prompt capacity.',
            'finals_after_all_branches_frozen':True,'final_timeout_seconds_per_condition':900})
        summaries={}
        def worker(s):
            try:return pool.run_worker(s,s.worker_id,100)
            except Exception as error:return {'worker_id':s.worker_id,'error':f'{type(error).__name__}: {error}'}
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures={executor.submit(worker,s):s.worker_id for s in sessions}
            futures.update({executor.submit(a.run,deadline,start):f'astra{t}' for t,a in raw.items()})
            for f in as_completed(futures):
                key=futures[f]
                try:summaries[key]=f.result()
                except Exception as error:summaries[key]={'error':f'{type(error).__name__}: {error}'}
                P.write_json(root/'agent-summaries.json',summaries)
        pool.close(wait_timeout=5);api=pool.summary()
        selections={task:host.select([s for s in sessions if s.host is host]) for task,host in hosts.items()}
        P.write_json(root/'all-frozen.json',{'at':time.time(),'all_agents_stopped':True,'selections':selections})
        for name in containers:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
        containers=[]
        results={}
        def score_condition(task,condition,bundle):
            began=time.monotonic();output=root/f'task{task:02d}'/condition/'final'
            try:
                reward=final_score(task,bundle,output,timeout=900)
                result={'status':'completed','score':100*float(reward['reward']),'official_reward':reward}
            except Exception as error:result={'status':'final_failed','score':None,'error':f'{type(error).__name__}: {error}'}
            result['final_seconds']=time.monotonic()-began
            P.write_json(output.parent/'result.json',result);return result
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures={}
            for task,host in hosts.items():
                futures[executor.submit(score_condition,task,'deepseek',host.root/'selected')]=(task,'deepseek')
                futures[executor.submit(score_condition,task,'astra',raw[task].root/'frozen')]=(task,'astra')
            for f in as_completed(futures):
                task,condition=futures[f];results.setdefault(str(task),{})[condition]=f.result()
                P.write_json(root/'results.json',results)
        P.write_json(root/'summary.json',{'status':'finished','results':results,'api':api,
            'agent_summaries':summaries,'total_wall_seconds':time.monotonic()-start,
            'development_seconds':900,'finished_at':time.time()})
        print(json.dumps(results),flush=True)
    finally:
        if pool is not None and not all(l.closed.is_set() for l in pool.lanes):pool.close(wait_timeout=3)
        for name in containers:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)


if __name__=='__main__':main()
