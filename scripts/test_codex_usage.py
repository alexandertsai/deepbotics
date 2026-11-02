"""Check token aggregation, partial accounting and pricing arithmetic."""

import json
from pathlib import Path
import tempfile
import unittest

import codex_usage as U


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_reasoning_and_cache_are_not_double_counted(self):
        self.save(self.root / 'agent.jsonl', [{'type': 'turn.completed', 'usage': {
            'input_tokens': 1000, 'cached_input_tokens': 700, 'output_tokens': 200,
            'reasoning_output_tokens': 150}}])
        result = U.collect(self.root)
        self.assertEqual(result['usage']['total_tokens'], 1200)
        self.assertEqual(result['api_equivalent_estimate']['short_context_usd'], .0137)
        self.assertIsNone(result['subscription_incremental_charge_usd'])

    def test_multiple_completed_turns_sum_once(self):
        row = {'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 2}}
        self.save(self.root / 'agent.jsonl', [row, row])
        self.assertEqual(U.collect(self.root)['usage']['total_tokens'], 24)

    def test_cumulative_session_snapshots_not_summed_and_other_threads_excluded(self):
        self.save(self.root / 'agent.jsonl', [{'type': 'thread.started', 'thread_id': 'test-thread'}])
        def row(n): return {'type': 'event_msg', 'payload': {'type': 'token_count',
            'info': {'total_token_usage': {'input_tokens': n, 'output_tokens': 2}}}}
        self.save(self.root / 'agent-home/sessions/rollout-test-thread.jsonl', [row(10), row(10), row(20)])
        self.save(self.root / 'agent-home/sessions/rollout-other.jsonl', [row(1000)])
        result = U.collect(self.root)
        self.assertEqual(result['usage']['total_tokens'], 22)
        self.assertEqual(result['usage_status'], 'partial_recorded_lower_bound')

    def test_missing_usage_is_unknown_and_partial_last_line_is_ignored(self):
        (self.root / 'agent.jsonl').write_text('{"type":')
        result = U.collect(self.root)
        self.assertIsNone(result['usage'])
        self.assertIsNone(result['api_equivalent_estimate'])

    def test_invalid_subsets_rejected(self):
        with self.assertRaises(ValueError):
            U.normalize({'input_tokens': 1, 'cached_input_tokens': 2})
        with self.assertRaises(ValueError):
            U.normalize({'output_tokens': 1, 'reasoning_output_tokens': 2})


if __name__ == '__main__': unittest.main()
