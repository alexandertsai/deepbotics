import argparse
from collections import defaultdict
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepseek', type=Path, required=True)
    parser.add_argument('--astra', type=Path, required=True)
    args = parser.parse_args()
    usage = defaultdict(lambda: {'provider_attempts':0, 'estimated_api_cost_usd':0.,
        'reported_tokens':0, 'token_usage':defaultdict(int), 'pacing_wait_seconds':0.,
        'queue_seconds':0., 'rate_limited_attempts':0, 'accounts':set(), 'turns':set(),
        'workers':set(), 'intervals':[], 'models':set()})
    for path in (args.deepseek/'shared-api').glob('account-*/api/metrics-*.json'):
        row = read(path)
        worker = row.get('worker_id','')
        if not worker.startswith(('b6_','b8_','f6_','f8_')):
            continue
        task = int(worker[1])
        item = usage[task]
        item['provider_attempts'] += 1
        item['estimated_api_cost_usd'] += row.get('estimated_cost_usd',0.)
        item['reported_tokens'] += row.get('reported_tokens',0)
        for key, value in row.get('usage',{}).items():
            if isinstance(value,(int,float)):
                item['token_usage'][key] += value
        item['pacing_wait_seconds'] += row.get('pacing_wait_seconds',0.)
        item['queue_seconds'] += row.get('queue_seconds',0.)
        item['rate_limited_attempts'] += row.get('status_code') == 429
        item['accounts'].add(path.parents[1].name)
        item['turns'].add((worker,row.get('logical_request')))
        item['workers'].add(worker)
        item['models'].add(row.get('model'))
        if row.get('provider_started_at') is not None and row.get('provider_latency_seconds') is not None:
            began=row['provider_started_at']
            item['intervals'].extend(((began,1),(began+row['provider_latency_seconds'],-1)))
    comparisons = {}
    for task in (6,8):
        label = f'task{task:02d}'
        deep_result = read(args.deepseek/label/'deepseek/result.json')
        astra_result = read(args.astra/label/'result.json')
        astra_agent = read(args.astra/label/'agent-summary.json')
        selection = read(args.deepseek/label/'deepseek/selection.json')
        item = usage[task]
        item['accounts'] = sorted(item['accounts'])
        item['account_count'] = len(item['accounts'])
        item['logical_model_turns'] = len(item.pop('turns'))
        item['actual_model_workers'] = len(item.pop('workers'))
        active=peak=0
        for _,delta in sorted(item.pop('intervals')):
            active+=delta
            peak=max(peak,active)
        item['peak_concurrent_api_requests']=peak
        item['token_usage'] = dict(item['token_usage'])
        item['models'] = sorted(item['models'])
        development_seconds=(read(args.deepseek/'all-frozen.json')['at']-
            read(args.deepseek/'process.json')['started_at'])
        deepseek = {**item, 'agents':14, 'coordinators':2, 'development_budget_seconds':900,
            'development_and_freeze_seconds':development_seconds,
            'score':deep_result['score'], 'status':deep_result['status'],
            'final_seconds':deep_result['final_seconds'], 'run':str(args.deepseek/label/'deepseek'),
            'selection':selection}
        astra_usage = astra_agent['usage']
        astra = {'agents':1, 'model':'gpt-6-astra','mode':'raw native codex exec','reasoning_effort':'max',
            'score':astra_result['score'], 'status':astra_result['status'],
            'development_budget_seconds':900,
            'native_agent_seconds':astra_agent['seconds_after_common_start'],
            'final_seconds':astra_result['final_seconds'],
            'reported_tokens':astra_usage['usage']['total_tokens'],
            'token_usage':astra_usage['usage'],
            'usage_status':astra_usage['usage_status'],
            'api_equivalent_cost_is_lower_bound':astra_usage['usage_status'].startswith('partial'),
            'request_records':len(astra_usage.get('request_usage_records',[])),
            'agent_returncode':astra_agent['returncode'],
            'artifact_sha256':astra_agent['manifest']['sha256'],
            'api_equivalent_cost_usd':astra_usage.get('request_level_api_equivalent_usd'),
            'actual_subscription_charge_usd':None,'run':str(args.astra/label)}
        comparison = {'task':task, 'deepseek':deepseek,'astra':astra,
            'comparison_scope':f'{label}: fourteen DeepSeek agents with a custom harness versus one raw native Codex Astra agent; 15 minutes development each.',
            'cost_scope':'DeepSeek estimated API spend and Astra standard API-equivalent estimate; local compute excluded.',
            'created_at':time.time()}
        if deep_result['score'] is not None and astra_result['score'] is not None:
            comparison['deepseek_points_ahead'] = deep_result['score']-astra_result['score']
            comparison['winner'] = ('deepseek' if comparison['deepseek_points_ahead']>0 else
                'astra' if comparison['deepseek_points_ahead']<0 else 'tie')
        else:
            comparison['winner'] = None
        write(Path('benchmarks')/f'{label}-comparison.json',comparison)
        if astra_result['status']=='completed':
            write(Path('benchmarks')/f'{label}-sota.json',{'label':'SOTA', 'scope':'Project single-agent raw Astra reference',
                'task':task,**astra})
        comparisons[str(task)] = comparison
    write(args.deepseek/'comparison.json',comparisons)
    print(json.dumps({task:{'deepseek':d['deepseek']['score'],'astra':d['astra']['score'],
        'winner':d['winner']} for task,d in comparisons.items()},indent=2))


if __name__ == '__main__':
    main()
