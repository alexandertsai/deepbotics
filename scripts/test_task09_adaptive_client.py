"""Verify adaptive pacing without inference or wall-clock ramp waits."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from task09_adaptive_client import AdaptivePacedClient


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def success(body, timeout):
        return 200, json.dumps({'usage': {'input_tokens': 5, 'cache_read_input_tokens': 7,
                                        'output_tokens': 3}, 'content': []}).encode(), 'application/json', {}

    def client(self, **kwargs):
        options = dict(deadline=time.monotonic() + 10, start_server=False, transport=self.success,
                       start_stagger_seconds=0, max_output_tokens=8192)
        options.update(kwargs)
        client = AdaptivePacedClient(self.root, {'FIREWORKS_API_KEY': 'test-key'}, **options)
        self.addCleanup(client.close)
        return client

    def eligible(self, client):
        client.next_ramp_at = time.monotonic() - 1
        client.clean_successes = 3
        client.last_success_at = time.monotonic()

    def test_idle_time_and_insufficient_traffic_do_not_raise_rates(self):
        c = self.client()
        c.next_ramp_at = time.monotonic() - 1
        with c.condition:
            c._rates_unlocked()
        self.assertEqual(c.pacing_config, c.initial_pacing)
        self.eligible(c)
        c.last_success_at = time.monotonic() - 121
        with c.condition: c._rates_unlocked()
        self.assertEqual(c.pacing_config, c.initial_pacing)

    def test_clean_window_increases_ten_percent_once(self):
        c = self.client()
        self.eligible(c)
        with c.condition: c._rates_unlocked()
        self.assertAlmostEqual(c.pacing_config['prompt'], 132000)
        self.assertAlmostEqual(c.pacing_config['generated'], 33000)
        with c.condition: c._rates_unlocked()
        self.assertEqual(len(c.rate_adjustments), 1)

    def test_raw_reported_limit_retained_and_eighty_percent_cap_applied(self):
        c = self.client()
        c._apply_headers(200, {'x-ratelimit-limit-tokens-generated': '70312'}, 0)
        self.assertEqual(c.observed_provider_limits['generated'], 70312)
        for _ in range(20):
            self.eligible(c)
            with c.condition: rates = c._rates_unlocked()
        self.assertLessEqual(rates['generated'], 70312 * .8)
        self.assertLessEqual(rates['prompt'], 240000)
        self.assertLessEqual(len(c.summary()['adaptive_pacing']['recent_adjustments']), 5)
        self.assertEqual(len(json.loads((self.root / 'pacing-adjustments.json').read_text())), len(c.rate_adjustments))

    def test_lower_provider_header_reduces_effective_rate_immediately(self):
        c = self.client()
        c._apply_headers(200, {'x-ratelimit-limit-tokens-generated': '20000'}, 0)
        with c.condition: rates = c._rates_unlocked()
        self.assertLessEqual(rates['generated'], 16000)
        self.assertEqual(c.observed_provider_limits['generated'], 20000)

    def test_429_backs_off_and_preserves_financial_guard(self):
        c = self.client()
        c.pacing_config.update(prompt=200000, uncached=200000, generated=50000)
        delay = c._apply_headers(429, {'retry-after': '1.5'}, 0)
        self.assertEqual(delay, 1.5)
        self.assertEqual(c.pacing_config, {'prompt': 150000, 'uncached': 150000, 'generated': 37500})
        self.assertEqual(c.clean_successes, 0)
        self.assertGreater(c.next_ramp_at, time.monotonic())
        self.assertEqual(c.max_cost, 5)
        self.assertEqual(c.max_concurrent, 3)

    def test_recovered_throttle_stays_one_logical_request(self):
        calls = []
        def transport(body, timeout):
            calls.append(1)
            if len(calls) == 1:
                return 429, b'{"error":"throttled"}', 'application/json', {'retry-after': '0'}
            return self.success(body, timeout)
        c = self.client(transport=transport, _window_seconds=.01)
        c.register_worker('a')
        result = c.request('a', {'messages': [{'role': 'user', 'content': 'test'}]})
        self.assertEqual(result[0], 200)
        self.assertEqual((c.calls, c.logical_requests), (2, 1))
        self.assertFalse(c.workers['a']['refused'].is_set())
        self.assertGreater(c.rejected_cost_allowance, 0)
        self.assertLessEqual(c.cost, 5)

    def test_three_provider_calls_can_overlap(self):
        barrier = threading.Barrier(3)
        def transport(body, timeout):
            barrier.wait(2)
            return self.success(body, timeout)
        c = self.client(transport=transport)
        for name in ('a', 'b', 'c'): c.register_worker(name)
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda name: c.request(name, {'messages': [{'role': 'user', 'content': 'test'}]}), ('a', 'b', 'c')))
        self.assertTrue(all(result[0] == 200 for result in results))
        self.assertEqual(c.peak_active, 3)

    def test_invalid_controller_settings_are_rejected(self):
        for settings in ({'ramp_fraction': 0}, {'reported_limit_fraction': 1},
                         {'max_prompt_tpm': 100}, {'ramp_interval_seconds': float('inf')}):
            with self.assertRaises(ValueError): self.client(**settings)


if __name__ == '__main__': unittest.main()
