"""Single native Codex agent on public Task09, with usage recording and one final score."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import task08_pilot as P
import task09_pilot as Q
import codex_usage as U
from task08_swarm_client import SharedClient


def usage_audit(root):
    report = U.write(root)
    requests, seen = [], set()
    for path in (root/'agent-home/sessions').rglob('*.jsonl'):
        for event in U.events(path):
            payload = event.get('payload') or {}
            if event.get('type') != 'event_msg' or payload.get('type') != 'token_count':
                continue
            info = payload.get('info') or {}
            total, last = info.get('total_token_usage'), info.get('last_token_usage')
            if not isinstance(total,dict) or not isinstance(last,dict):
                continue
            signature = (str(path),json.dumps(total,sort_keys=True))
            if signature in seen:
                continue
            seen.add(signature)
            value = U.normalize(last)
            cached = value['cached_input_tokens']
            writes = last.get('cache_write_input_tokens',0)
            uncached = max(0,value['input_tokens']-cached-writes)
            long = value['input_tokens'] > 272000
            cost = (uncached*10+cached+writes*12.5)*(2 if long else 1)/1e6
            cost += value['output_tokens']*50*(1.5 if long else 1)/1e6
            requests.append({'timestamp':event.get('timestamp'),'usage':value,'cache_write_input_tokens':writes,
                'long_context':long,'hypothetical_standard_api_usd':cost})
    report['request_usage_records'] = requests
    report['request_level_api_equivalent_usd'] = sum(r['hypothetical_standard_api_usd'] for r in requests) if requests else None
    if report.get('usage'):
        u = report['usage']
        report['standard_credit_equivalent'] = ((u['input_tokens']-u['cached_input_tokens'])*250
            +u['cached_input_tokens']*25+u['output_tokens']*1250)/1e6
    report['pricing_sources'] = ['https://developers.openai.com/api/docs/models/gpt-6-astra',
        'https://learn.chatgpt.com/docs/pricing']
    report['cost_scope'] = 'Token-based hypothetical API and credit equivalents; subscription invoice charge is unknown. Local compute excluded.'
    P.write_json(root/'usage-audit.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    auth = Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'auth.json'
    if not auth.is_file():
        raise SystemExit('Codex login unavailable; no model run started')
    work = Path(tempfile.mkdtemp(prefix='task09-astra-raw-')).resolve()
    for variant in Q.VARIANTS:
        shutil.copytree(Q.PUBLIC/variant,work/variant)
    Q.make_starter(work/'submission')
    home = root/'agent-home'
    home.mkdir()
    (home/'auth.json').symlink_to(auth)
    from harness.codex_cli.configuration import write_config
    write_config(home/'config.toml',{'model':'gpt-6-astra','model_reasoning_effort':'max',
        'approval_policy':'never','sandbox_mode':'workspace-write','web_search':'disabled',
        'features':{name:False for name in ('multi_agent','multi_agent_v2','apps','plugins','skill_search',
            'browser_use','computer_use','image_generation')}})
    start = time.monotonic()
    wall_start = time.time()
    deadline = start+810
    P.write_json(root/'process.json',{'pid':os.getpid(),'started_at':wall_start,'total_deadline_at':wall_start+900})
    instruction = (Q.TASK/'instruction.md').read_text().replace('/logs/artifacts/','submission/').replace('/workspace/','./')
    prompt = f'''Complete this Task09 benchmark as ONE Astra agent using your ordinary Codex coding tools.
Your working directory contains the public franka/, ur5e/ and xarm7/ assets and public nominal helpers.
The submission/ directory starts with the original public models and zero trim stubs.
Write the six required final entry files and referenced meshes under submission/VARIANT/.
You have 810 seconds for coding and development, followed by separate host final scoring within a 900-second total limit.
Do not spawn subagents or use other models. Do not inspect files outside this working directory,
except installed Python packages and the Python executable {P.ROOT/'.venv/bin/python'}.
That Python environment provides mujoco, numpy and scipy. No custom MCP tools or development harness is supplied.
You may write and run your own analyses and public tests with ordinary coding tools.
No web, private verifier source, prior solutions, peer artifacts, or hidden evaluation feedback is available.
Finish by saving your best solution under submission/ and return a concise completion message.
No emojis. Use minimal code comments. The original task follows; its longer time allowance is replaced above.
{instruction}
'''
    (root/'prompt.txt').write_text(prompt)
    env = {key:os.environ[key] for key in ('PATH','LANG','USER','LOGNAME','TMPDIR') if key in os.environ}
    env.update(CODEX_HOME=str(home),HOME=str(home),PYTHONUNBUFFERED='1')
    command = [shutil.which('codex'),'exec','--skip-git-repo-check','--json','-C',str(work),'--',prompt]
    P.write_json(root/'protocol.json',{'task':9,'label':'SOTA','label_scope':'Project single-agent Astra reference',
        'model':'gpt-6-astra','reasoning_effort':'max','agents':1,'kind':'raw_codex_exec',
        'custom_mcp_tools':False,'engineered_development_probe':False,'private_feedback_during_development':False,
        'original_public_starter':True,'deepseek_artifacts_supplied':False,'development_seconds_limit':810,
        'total_seconds_limit':900,'native_coding_tools':True,'local_python':str(P.ROOT/'.venv/bin/python'),
        'workspace':str(work),'billing_route':'ChatGPT_Codex_subscription','invoice_cost_available':False,
        'official_final_evaluations':1,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'difference_from_deepseek':'One raw native Codex agent; no engineered team harness, analysis MCP or robust public selector.'})
    process = None
    try:
        with (root/'agent.jsonl').open('w') as out,(root/'agent.stderr').open('w') as err:
            process = subprocess.Popen(command,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            while process.poll() is None and time.monotonic()<deadline:
                usage_audit(root)
                time.sleep(2)
            if process.poll() is None:
                SharedClient._stop(process)
        usage = usage_audit(root)
        from services.controller.storage import snapshot
        manifest = snapshot(work/'submission',root/'frozen',max_bytes=256*1024**2,workspace_path='code/task09')
        P.write_json(root/'frozen-manifest.json',manifest)
        result = {'status':'frozen','score':None,'agent_returncode':process.returncode,
            'development_and_freeze_seconds':time.monotonic()-start,'usage':usage}
        try:
            reward = Q.evaluate(root/'frozen',root/'final',final=True,timeout=max(.1,start+895-time.monotonic()))
            result.update(status='completed',score=round(100*reward['reward'],4),official_reward=reward)
        except Exception as error:
            result.update(status='final_failed',error=f'{type(error).__name__}: {error}')
        result.update(total_wall_seconds=time.monotonic()-start,completed_within_900_seconds=time.monotonic()-start<=900,
            finished_at=time.time(),workspace_preserved=str(work))
        P.write_json(root/'summary.json',result)
        if result['status']=='completed':
            target = P.ROOT/'benchmarks/task09-sota.json'
            P.write_json(target,{'label':'SOTA','scope':'Project single-agent raw Astra reference; not a global leaderboard claim',
                'task':9,'model':'gpt-6-astra','agents':1,'reasoning_effort':'max','run':str(root.relative_to(P.ROOT)),
                'score':result['score'],'total_wall_seconds':result['total_wall_seconds'],
                'usage_status':usage['usage_status'],'usage':usage['usage'],
                'api_equivalent_estimate_usd':usage.get('request_level_api_equivalent_usd'),
                'standard_credit_equivalent':usage.get('standard_credit_equivalent'),
                'subscription_invoice_charge_usd':None,'cost_scope':usage['cost_scope'],
                'artifact_sha256':manifest['sha256'],'updated_at':time.time()})
        print(json.dumps({k:result[k] for k in ('status','score','total_wall_seconds')}),flush=True)
    finally:
        if process is not None and process.poll() is None:
            SharedClient._stop(process)


if __name__ == '__main__':main()
