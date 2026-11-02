"""Record Codex subscription token usage and a separate API-equivalent estimate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ASTRA_PRICING = {
    'model': 'gpt-6-astra', 'checked_on': '2026-11-02',
    'source': 'https://developers.openai.com/api/docs/pricing',
    'service_tier': 'standard', 'short_context_limit': 272000,
    'usd_per_million': {'input': 10.0, 'cached_input': 1.0, 'output': 50.0},
    'long_context_multipliers': {'input': 2.0, 'cached_input': 2.0, 'output': 1.5},
}


def events(path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    yield value


def normalize(usage):
    fields = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens')
    result = {}
    for field in fields:
        value = usage.get(field, 0)
        if type(value) is not int or value < 0:
            raise ValueError('Token counts must be nonnegative integers')
        result[field] = value
    if result['cached_input_tokens'] > result['input_tokens']:
        raise ValueError('Cached input is a subset of input tokens')
    if result['reasoning_output_tokens'] > result['output_tokens']:
        raise ValueError('Reasoning output is a subset of output tokens')
    result['total_tokens'] = result['input_tokens'] + result['output_tokens']
    return result


def estimate(usage):
    rates = ASTRA_PRICING['usd_per_million']
    short = ((usage['input_tokens'] - usage['cached_input_tokens']) * rates['input']
        + usage['cached_input_tokens'] * rates['cached_input']
        + usage['output_tokens'] * rates['output']) / 1e6
    long = ((usage['input_tokens'] - usage['cached_input_tokens']) * rates['input'] * 2
        + usage['cached_input_tokens'] * rates['cached_input'] * 2
        + usage['output_tokens'] * rates['output'] * 1.5) / 1e6
    return {'short_context_usd': round(short, 6), 'all_long_context_usd': round(long, 6),
        'assumptions': 'Hypothetical standard API pricing; cache-write charges and tool fees excluded. Context tier is per request, not cumulative run tokens. No subscription invoice amount is inferred.',
        'pricing': ASTRA_PRICING}


def collect(root):
    root = Path(root)
    completed = []
    thread_ids = []
    for event in events(root / 'agent.jsonl'):
        if event.get('type') == 'thread.started':
            thread_ids.append(event.get('thread_id'))
        if event.get('type') == 'turn.completed' and isinstance(event.get('usage'), dict):
            completed.append(normalize(event['usage']))
    usage, source = None, None
    if completed:
        usage = {key: sum(row[key] for row in completed) for key in completed[0]}
        source = 'codex_exec_turn_completed'
    else:
        snapshots = []
        home = root / 'agent-home'
        for directory in ('sessions', 'archived_sessions'):
            for path in (home / directory).rglob('*.jsonl'):
                if thread_ids and not any(t and t in path.name for t in thread_ids):
                    continue
                for event in events(path):
                    payload = event.get('payload') or {}
                    if event.get('type') == 'event_msg' and payload.get('type') == 'token_count':
                        value = (payload.get('info') or {}).get('total_token_usage')
                        if isinstance(value, dict):
                            snapshots.append(normalize(value))
        if snapshots:
            usage = max(snapshots, key=lambda row: row['total_tokens'])
            source = 'codex_session_cumulative_token_count'
    return {'model': 'gpt-6-astra', 'billing_route': 'ChatGPT_Codex_subscription',
        'usage_status': 'complete_turn' if completed else ('partial_recorded_lower_bound' if usage else 'unavailable'),
        'usage': usage, 'source': source, 'completed_turns': len(completed),
        'reasoning_tokens_included_in_output': True, 'cached_tokens_included_in_input': True,
        'subscription_incremental_charge_usd': None,
        'api_equivalent_estimate': estimate(usage) if usage else None,
        'note': 'Missing usage is unknown, not zero. Interrupted runs may omit usage from their last request.'}


def write(root):
    report = collect(root)
    (Path(root) / 'token-usage.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(write(args.root), indent=2))


if __name__ == '__main__':
    main()
