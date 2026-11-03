"""Fourteen DeepSeek agents with a 900-second end-to-end Task09 deadline."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import task08_pilot as P
import task09_pilot as Q
import task09_swarm_followup as F
import task09_team_sessions as W
from task08_sim_pool import SimulationPool
from task08_swarm_short import primary_keys
from short_pilot_pool import AccountPool
from task09_planner_short import throughput

ROLES = (
    ('mechanical_coordinator', 'Coordinate six hardware branches, assign distinct measured experiments, and integrate useful hardware with measured software.'),
    ('software_coordinator', 'Coordinate software and calibration branches, prevent nominal-only overfitting, and integrate useful measured designs.'),
    ('spring_fit', 'Fit nonnegative joint stiffness and spring reference poses from public workspace gravity torques using scipy.optimize; test all variants.'),
    ('spring_conservative', 'Fit conservative low-stiffness springs, balancing workspace residual against passive motion and backdrive resistance.'),
    ('density_300', 'Explore printed infill density around 300 kg/m3, preserving all stock geometry and masses, plus model-based trim.'),
    ('density_500', 'Explore printed infill density around 500 kg/m3 with fitted legal springs and model-based trim.'),
    ('density_gradient', 'Explore varied proximal and distal printed densities and legal springs without changing geometry.'),
    ('hardware_integration', 'Integrate measured low-mass and spring designs, preserving geometry, printable structural mass and useful motion.'),
    ('nominal_trim', 'Implement general MuJoCo gravity-minus-spring residual feedforward from the supplied runtime model for every variant.'),
    ('linear_adaptation', 'Implement regularized mass-scale and payload-torque adaptation using the 16 noisy settled hold measurements and runtime nominal model.'),
    ('robust_adaptation', 'Infer residual torque from clipped servo equilibrium; reject saturated/noisy measurements; regularize adaptation so nominal behavior remains safe.'),
    ('probe_design', 'Implement informative workspace calibration poses to identify body-mass and unknown handle payload effects from 16 samples.'),
    ('software_integration', 'Integrate measured general feedforward and stable adaptation; test the original public reference model as well as your candidate.'),
    ('independent_codesign', 'Independently implement a conservative complete hardware/software design, then integrate peer improvements from public evidence.'),
)
WEIGHTS = {'hold': .15, 'passive_residual': .15, 'servo_headroom': .05,
           'passive_hold': .20, 'spring_motion': .10, 'transfer_nominal': .15, 'transfer_adapted': .20}


def components(report):
    F.hold_score(report)
    values = {key: [] for key in WEIGHTS}
    for row in report['variants'].values():
        for mode in ('nominal', 'adapted', 'transfer_nominal', 'transfer_adapted'):
            if mode not in row:
                raise ValueError('Missing public robustness measurement')
            for field in ('hold_rms_error_rad', 'servo_headroom_mean'):
                number = float(row[mode][field])
                if not math.isfinite(number) or number < 0:
                    raise ValueError('Invalid public robustness measurement')
        values['hold'].append(sum(1/(1+row[m]['hold_rms_error_rad']/.10) for m in ('nominal', 'adapted'))/2)
        values['passive_residual'].append(1/(1+row['passive_residual_max_nm']/.35))
        values['servo_headroom'].append(sum(row[m]['servo_headroom_mean'] for m in ('nominal', 'adapted'))/2)
        values['passive_hold'].append(1/(1+row['passive_drift_rad']/.15))
        values['spring_motion'].append(1/(1+row['spring_motion_rms_nm']/.35))
        for mode in ('transfer_nominal', 'transfer_adapted'):
            values[mode].append(1/(1+row[mode]['hold_rms_error_rad']/.10))
    return {key: sum(value)/len(value) for key, value in values.items()}


def score(report):
    return sum(WEIGHTS[key]*value for key, value in components(report).items())


def evaluate(bundle, output, *, final=False, timeout=900):
    if final:
        return Q.evaluate(bundle, output, final=True, timeout=timeout)
    output.mkdir(parents=True, exist_ok=False)
    name = 'task09-team15-probe-' + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    command = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only',
        '--cpus', '2', '--memory', '1g', '--memory-swap', '1g', '--pids-limit', '128',
        '--tmpfs', '/tmp:rw,nosuid,size=256m', '--security-opt', 'no-new-privileges',
        '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'MUJOCO_GL=disable',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-v', f'{bundle.resolve()}:/submission:ro',
        '-v', f'{output.resolve()}:/logs',
        '-v', f'{P.ROOT}/scripts/task09_team_probe.py:/pilot/probe.py:ro',
        '-v', f'{P.ROOT}/scripts/task09_followup_probe.py:/pilot/base_probe.py:ro',
        Q.AGENT_IMAGE, 'python3', '/pilot/probe.py']
    try:
        completed = P.checked(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=5)
        raise TimeoutError('Public probe deadline; owned container removed') from None
    except subprocess.CalledProcessError as error:
        (output/'container.log').write_text((error.stdout or '')+(error.stderr or ''))
        raise RuntimeError('Public probe failed; see container.log') from None
    (output/'container.log').write_text(completed.stdout+completed.stderr)
    return json.loads((output/'probe.json').read_text())


class Coordinator(W.Coordinator):
    def __init__(self, *args, **kwargs):
        self.guidance, self.nominations = {}, []
        super().__init__(*args, **kwargs)

    def status(self):
        with self.lock:
            return {'seconds_remaining': max(0, round(self.deadline-time.monotonic())),
                'best_partial_score': self.best['score'], 'best_worker': self.best['worker'],
                'best_components': components(self.best['report']), 'starter_partial_score': self.baseline_score,
                'probes': [{k:r[k] for k in ('index','worker','status','partial_score','selection_components') if k in r}
                           for r in self.entries[-30:]],
                'findings': self.findings[-12:], 'guidance': self.guidance,
                'new_probes_remaining': self.max_tests-self.tests, 'nominations': self.nominations[-4:]}

    def test(self, session, hypothesis, source=None):
        candidate = source or session.candidate
        for variant in Q.VARIANTS:
            tree = ET.parse(candidate/variant/'lead.xml')
            original = ET.parse(self.root/'starter'/variant/'lead.xml')
            stock = {g.get('name'):g for g in original.findall('.//geom') if g.get('name')}
            for geom in tree.findall('.//geom'):
                name = geom.get('name', '')
                if name in stock:
                    old = stock.pop(name)
                    for key in ('type','mesh','pos','quat','size'):
                        if geom.get(key) != old.get(key):
                            raise ValueError('Preserve supplied geom '+name+' attribute '+key)
                    if name.startswith('servo') or name == 'printed_trigger':
                        for key in ('mass','density'):
                            if geom.get(key) != old.get(key):
                                raise ValueError('Preserve stock servo and trigger mass')
                    if name.startswith('printed_') and float(geom.get('density', '1000')) < 250:
                        raise ValueError('Run engineering guard: printed density must be at least 250 kg/m3')
            if stock:
                raise ValueError('Preserve all supplied geoms')
        with self.simulator.owner('deepseek', session.worker_id):
            return super().test(session, hypothesis, source=source)


class Session(W.Session):
    def __init__(self, *args, planner=False, **kwargs):
        self.planner = planner
        self.analysis_count = 0
        super().__init__(*args, **kwargs)

    def make_prompt(self):
        text = Q.Session.make_prompt(self).replace('after both agents stop', 'after all agents stop')
        files = '\n'.join(f'{name}:\n{(self.coordinator.root/"starter"/name).read_text()}' for name in F.ENTRY_FILES)
        return text + f'''
You are {self.worker_id}: {self.brief}
This is a DeepSeek-only team of TWO planners/coordinators and TWELVE executors. All start together.
You have at most 810 seconds for development; final verification and cleanup must finish within 900 total seconds.
Start from the ORIGINAL PUBLIC starter. No Astra artifacts or private results are supplied.
Use run_analysis(code) to execute Python with mujoco, scipy and numpy in an isolated CPU container.
The cwd is /submission; your six entry files are writable there and meshes are read-only.
Public files are /workspace/franka, /workspace/ur5e and /workspace/xarm7. To import a variant's helpers,
insert /workspace/VARIANT into sys.path; import harness.scenarios and harness.spec.
For multiple variants use separate subprocesses or clear harness modules between imports.
Analysis has no network or private verifier access, 2 CPUs, 1 GiB, 45 seconds and bounded output.
Use this tool to fit stiffness/reference poses, generate edits across all variants, and validate syntax.
Do useful numerical engineering immediately. First implemented measured test within 90 seconds.
Do not spend many turns inspecting files already supplied. Batch code edits using run_analysis.
General nominal trim: construct MjData(model), set qpos, zero qvel, call mj_forward, then compute
qfrc_bias minus conservative spring torque; the supplied model can be a DIFFERENT reference arm.
The settled probe measurements imply servo torque clip(8*(target-measured),-.35,.35), with noise
and possible saturation. Account for passive springs, friction and saturation when fitting adaptation.
Adapted state must serialize between processes; no reliance on globals or an XML path.
Hardware must preserve stock geometry, kinematic frames, servo mass .018 kg, trigger mass and meshes.
Run guard: printed densities >=250 kg/m3, total mass <=2.30 kg, no inertial overrides or fake links.
A test runs all variants, six workspace poses, passive drift, low spring-motion resistance,
nominal and calibrated holds under disclosed mass variation (.88..1.12), .06 kg handle payload,
encoder noise .005 rad, and transfer to the original public nominal model.
These are PUBLIC development tests, not a reproduction of private scores. Public selection weights:
{json.dumps(WEIGHTS)}. Only completed measured frozen designs are eligible; starter wins regressions.
Read swarm_status once, then use shared measured candidates and coordinator guidance.
clone_candidate(index) copies measured six entry files into your own branch; -1 is original starter.
Improve and measure at least three meaningful versions if time permits. Use at most eight new tests
and 24 analyses. Avoid early submit while useful measured improvement remains possible.
For integration, combine the best measured XML with the best general trim, then test the combination.
For coordination, use coordinate(worker,instruction) to direct distinct executors and nominate(index,reason)
to advise final selection from measured public evidence. Coordinator nominations are recorded.
wait_for_results(seconds) blocks up to 15 seconds without API polling; use it between useful coordination rounds.
Do not change peer artifacts. Do not spawn more agents. No private results are available during development.
The final official scorer is unchanged and runs once after the entire team freezes. Never claim a private score.
Team roles: {json.dumps(ROLES)}
Starter public development evidence: {json.dumps(self.coordinator.baseline_report, separators=(',',':'))}
Full original starter entry files:
{files}
'''

    def definitions(self):
        result = super().definitions()
        for name, description, properties, required in (
            ('run_analysis','Run isolated Python analysis and edit your six entry files; no private verifier or network.', {'code':{'type':'string'}}, ['code']),
            ('clone_candidate','Copy the six entry files of a measured candidate into your branch.', {'index':{'type':'integer','minimum':-1}}, ['index']),
            ('wait_for_results','Wait up to 15 seconds for new measured results without repeated API calls.', {'seconds':{'type':'number','minimum':1,'maximum':15}}, ['seconds']),
        ):
            result.append({'name':name,'description':description,'inputSchema':{'type':'object','properties':properties,'required':required,'additionalProperties':False}})
        if self.planner:
            result.extend([
                {'name':'coordinate','description':'Send concise engineering guidance to an executor.','inputSchema':{'type':'object','properties':{'worker':{'type':'string'},'instruction':{'type':'string'}},'required':['worker','instruction'],'additionalProperties':False}},
                {'name':'nominate','description':'Record a measured candidate recommendation; host uses the declared public metric.','inputSchema':{'type':'object','properties':{'index':{'type':'integer','minimum':-1},'reason':{'type':'string'}},'required':['index','reason'],'additionalProperties':False}}])
        return result

    def analysis(self, code):
        if not isinstance(code, str) or not 0 < len(code) <= 60000 or self.analysis_count >= 24:
            raise ValueError('Use <=60000 characters and <=24 analysis calls')
        self.analysis_count += 1
        name = f'task09-team15-analysis-{self.worker_id}-{self.analysis_count}'
        record = self.root/'analysis'/str(self.analysis_count)
        record.mkdir(parents=True)
        (record/'code.py').write_text(code)
        command = ['docker','run','--rm','-i','--name',name,'--network','none','--read-only',
            '--cpus','2','--memory','1g','--memory-swap','1g','--pids-limit','128',
            '--tmpfs','/tmp:rw,nosuid,size=256m','--security-opt','no-new-privileges',
            '-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','MUJOCO_GL=disable',
            '-e','PYTHONDONTWRITEBYTECODE=1','-v',f'{self.candidate}:/submission:ro']
        for entry in F.ENTRY_FILES:
            command += ['-v',f'{self.candidate/entry}:/submission/{entry}:rw']
        command += ['-w','/submission',Q.AGENT_IMAGE,'python3','-']
        started = time.monotonic()
        with self.coordinator.simulator.owner('deepseek_analysis', self.worker_id):
            with self.coordinator.simulator:
                try:
                    completed = subprocess.run(command,input=code,capture_output=True,text=True,
                        timeout=max(.1,min(45,self.deadline-time.monotonic())))
                    result = {'returncode':completed.returncode,'stdout':completed.stdout[-18000:], 'stderr':completed.stderr[-4000:]}
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=5)
                    result = {'error':'Analysis timeout; owned container removed'}
        result['wall_seconds_including_queue'] = time.monotonic()-started
        P.write_json(record/'result.json',result)
        return result

    def dispatch(self, name, arguments):
        if self.submitted is not None or time.monotonic() >= self.deadline:
            raise ValueError('Development has ended')
        if name == 'run_analysis':
            result = self.analysis(arguments['code'])
        elif name == 'clone_candidate':
            chosen = self.coordinator.measured_candidate(arguments['index'])
            for entry in F.ENTRY_FILES:
                shutil.copyfile(chosen['bundle']/entry, self.candidate/entry)
            result = {'copied':arguments['index'],'source_worker':chosen['worker']}
        elif name == 'wait_for_results':
            time.sleep(max(0,min(15,float(arguments['seconds']),self.deadline-time.monotonic())))
            result = self.coordinator.status()
        elif name == 'coordinate' and self.planner:
            if arguments['worker'] not in dict(ROLES) or len(arguments['instruction']) > 2500:
                raise ValueError('Use a listed worker and <=2500 characters')
            with self.coordinator.lock:
                self.coordinator.guidance[arguments['worker']] = {'from':self.worker_id,'instruction':arguments['instruction'],'at':time.time()}
                P.write_json(self.coordinator.root/'guidance.json',self.coordinator.guidance)
            result = {'sent':True}
        elif name == 'nominate' and self.planner:
            self.coordinator.measured_candidate(arguments['index'])
            with self.coordinator.lock:
                self.coordinator.nominations.append({'planner':self.worker_id,**arguments,'at':time.time()})
                P.write_json(self.coordinator.root/'nominations.json',self.coordinator.nominations)
            result = {'recorded':True}
        else:
            result = super().dispatch(name,arguments)
        if isinstance(result,dict):
            result['coordinator_guidance'] = self.coordinator.guidance.get(self.worker_id)
            result['seconds_remaining'] = max(0,int(self.deadline-time.monotonic()))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-cost', type=float, default=10.)
    args = parser.parse_args()
    if not 0 < args.max_cost <= 10:
        parser.error('Shared API guard must be positive and at most $10')
    keys = primary_keys()
    from short_pilot_pool import secondary_keys, third_keys
    secondary, third = secondary_keys(), third_keys()
    if not all(Q.preflight().values()):
        raise SystemExit('Read-only preflight failed; no inference started')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    lock = Path('/tmp/kimirax-short-task09-sim.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start = time.monotonic()
    hard_deadline, development_deadline = start+900, start+810
    P.write_json(root/'process.json',{'pid':os.getpid(),'started_at':time.time(),'total_deadline_at':time.time()+900})
    F.Q.evaluate = evaluate
    F.partial_score, F.metric_components = score, components
    F.compact_report = lambda report: report.get('variants', {})
    phase = {'deadline':development_deadline}
    sims = SimulationPool(lambda:phase['deadline'],root/'simulation-waits.jsonl',max_concurrent=4,memory_limit_gib=6.5,
        container_reader=lambda:set())
    reference = P.ROOT/'outputs/task09-starter-baseline-01'
    host = Coordinator(root/'team',reference,reference,seconds=810,tests=112,tests_per_worker=8,worker_count=14,
        calls=4200,calls_per_worker=300,tokens=100_000_000,output_tokens=4096,cost=args.max_cost)
    host.logical_calls_per_worker = 100
    host.start()
    host.deadline = development_deadline
    host.simulator = sims
    sessions = [Session(host.root/name,host,name,name,brief,planner=i<2) for i,(name,brief) in enumerate(ROLES)]
    pool = AccountPool(root/'shared-api',keys,secondary=secondary,third=third,worker_ids=[s.worker_id for s in sessions],
        deadline=development_deadline,max_concurrent=3,max_workers=14,logical_requests_per_worker=100,
        max_calls=4200,per_worker_calls=300,max_tokens=100_000_000,max_output_tokens=4096,max_cost=args.max_cost,
        prompt_tpm=500000,uncached_prompt_tpm=250000,generated_tpm=60000,prompt_bytes_per_token=2.9,
        start_stagger_seconds=.25)
    host.client = pool
    for session in sessions:
        session.start()
    protocol = {'task':9,'kind':'fourteen_deepseek_public_starter_15min','roles':ROLES,'planner_count':2,'executor_count':12,
        'accounts':3,'api_slots_per_account':3,'total_wall_limit_seconds':900,'development_cutoff_seconds':810,
        'max_cost_usd_shared':args.max_cost,'public_selection_weights':WEIGHTS,
        'public_probe_script_sha256':hashlib.sha256((P.ROOT/'scripts/task09_team_probe.py').read_bytes()).hexdigest(),
        'original_public_starter_sha256':host.manifest['sha256'],'astra_artifacts_supplied':False,
        'historical_astra_best_target':41.483333333333334,'historical_astra_latest_target':31.096666666666668,
        'historical_target_not_a_fresh_matched_control':True,'official_verifier_unchanged':True,
        'final_private_evaluations':1,'private_feedback_to_agents':False,'simulation_slots':4,'cpus_per_slot':2,
        'memory_gib_per_slot':1,'docker_memory_admission_gib':6.5,'printed_density_floor_kg_m3':250,
        'all_baseline_planning_queue_pacing_cleanup_and_final_time_counts':True,
        'initial_tpm_per_account':{'prompt':500000,'uncached':250000,'generated':60000},
        'reported_limits_target_fraction':.9,'max_output_tokens':4096,'finished_preparation_seconds':time.monotonic()-start}
    P.write_json(root/'protocol.json',protocol)
    summaries = {}
    def run(session):
        try:
            result = pool.run_worker(session,session.worker_id,100)
            host.rehearse_remaining([session])
            return result
        except Exception as error:
            return {'worker_id':session.worker_id,'error':f'{type(error).__name__}: {error}'}
    try:
        with ThreadPoolExecutor(max_workers=14) as executor:
            futures = {executor.submit(run,s):s.worker_id for s in sessions}
            for future in as_completed(futures):
                summaries[futures[future]] = future.result()
                P.write_json(root/'worker-summaries.json',summaries)
        pool.close(wait_timeout=5)
        development_wall = time.monotonic()-start
        selection = F.Coordinator.select(host)
        P.write_json(root/'all-frozen.json',{'all_agents_stopped':True,'selection':selection,'at':time.time()})
        phase['deadline'] = hard_deadline-5
        result = {'status':'selected','score':None,'selection':selection,'api':pool.summary(),
            'development_seconds_including_baseline_and_cleanup':development_wall,'planner_nominations':host.nominations}
        try:
            with sims.owner('deepseek','final'):
                with sims:
                    reward = evaluate(host.root/'selected',host.root/'final',final=True,
                        timeout=max(.1,hard_deadline-time.monotonic()-5))
            result.update(status='completed',score=round(100*reward['reward'],4),official_reward=reward)
        except Exception as error:
            result.update(status='final_failed',error=f'{type(error).__name__}: {error}')
        result.update(worker_summaries=summaries,turns=throughput(root,result['api']),
            simulation_pool=sims.summary(),public_probes=host.entries,
            analysis_calls={s.worker_id:s.analysis_count for s in sessions},total_wall_seconds=time.monotonic()-start,
            completed_within_900_seconds=time.monotonic()-start<=900,finished_at=time.time())
        P.write_json(root/'summary.json',result)
        print(json.dumps({k:result[k] for k in ('status','score','total_wall_seconds','completed_within_900_seconds')}),flush=True)
    finally:
        if not all(lane.closed.is_set() for lane in pool.lanes):
            pool.close(wait_timeout=3)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
