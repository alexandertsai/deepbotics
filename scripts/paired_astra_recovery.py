import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

from paired_benchmarks import RawAstra, final_score
import task08_pilot as P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    agents = {}
    try:
        for task in (6, 8):
            agents[task] = RawAstra(root/f'task{task:02d}', task, f'paired-repaired-astra-{task}')
        start = time.monotonic()
        wall_start = time.time()
        deadline = start+900
        P.write_json(root/'process.json', {'pid':__import__('os').getpid(), 'started_at':wall_start,
            'development_deadline_at':wall_start+900, 'total_deadline_at':wall_start+1830,
            'containers':[a.name for a in agents.values()],
            'reason':'Prior Astra starts could not execute tools because codex-code-mode-host was absent; no submission produced.'})
        summaries = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(a.run, deadline, start):t for t,a in agents.items()}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    summaries[str(task)] = future.result()
                except Exception as error:
                    summaries[str(task)] = {'error':f'{type(error).__name__}: {error}'}
                P.write_json(root/'agent-summaries.json', summaries)
        for agent in agents.values():
            subprocess.run(['docker','rm','-f',agent.name],capture_output=True,timeout=10)
        results = {}
        def score(task, agent):
            began = time.monotonic()
            output = agent.root/'final'
            try:
                reward = final_score(task, agent.root/'frozen', output, timeout=900)
                result = {'status':'completed','score':100*float(reward['reward']), 'official_reward':reward}
            except Exception as error:
                result = {'status':'final_failed','score':None,'error':f'{type(error).__name__}: {error}'}
            result['final_seconds'] = time.monotonic()-began
            P.write_json(agent.root/'result.json',result)
            return result
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(score,t,a):t for t,a in agents.items()}
            for future in as_completed(futures):
                results[str(futures[future])] = future.result()
                P.write_json(root/'results.json',results)
        P.write_json(root/'summary.json', {'results':results, 'agent_summaries':summaries,
            'status':'finished','wall_seconds':time.monotonic()-start})
        print(json.dumps(results),flush=True)
    finally:
        for agent in agents.values():
            subprocess.run(['docker','rm','-f',agent.name],capture_output=True,timeout=10)


if __name__ == '__main__':
    main()
