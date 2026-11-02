"""Provider-free pacing, retry accounting, and cancellation checks."""

from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task08_swarm_client import BudgetRefused, MODEL
from task09_paced_client import PacedClient, retry_after_seconds, safe_rate_headers


class PacedClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def client(self, **options):
        settings = dict(root=Path(self.temporary.name), keys={"FIREWORKS_API_KEY": "mock-key"},
            deadline=time.monotonic() + 8, transport=self.success, start_server=False,
            max_output_tokens=100, start_stagger_seconds=0, base_backoff_seconds=.005,
            jitter_seconds=0, _window_seconds=.03)
        settings.update(options)
        client = PacedClient(**settings)
        self.addCleanup(client.close)
        return client

    @staticmethod
    def success(body, timeout):
        return 200, json.dumps({"usage": {"input_tokens": 5, "cache_read_input_tokens": 7,
                                        "output_tokens": 3}, "content": []}).encode(), "application/json"

    @staticmethod
    def body():
        return {"model": "other", "max_tokens": 100, "system": "unchanged prefix",
                "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}]}

    @staticmethod
    def wait_until(predicate):
        end = time.monotonic() + 2
        while not predicate():
            if time.monotonic() > end:
                raise AssertionError("Expected state was not reached")
            time.sleep(.002)

    def test_retry_after_and_safe_headers(self):
        self.assertEqual(retry_after_seconds("2.5"), 2.5)
        self.assertEqual(retry_after_seconds(formatdate(102, usegmt=True), now=100), 2)
        self.assertEqual(retry_after_seconds(formatdate(90, usegmt=True), now=100), 0)
        for value in (None, "", "invalid", "NaN", "Infinity", "-1"):
            self.assertIsNone(retry_after_seconds(value))
        headers = safe_rate_headers({"Retry-After": "2\r\n", "Authorization": "secret",
                                     "x-ratelimit-reset-tokens-prompt": "soon"})
        self.assertEqual(headers, {"retry-after": "2", "x-ratelimit-reset-tokens-prompt": "soon"})

    def test_recovered_429_preserves_logical_worker_and_meters_attempts(self):
        bodies = []
        def transport(body, timeout):
            bodies.append(json.loads(body))
            if len(bodies) == 1:
                return 429, b'{"error":"limited"}', "application/json", {"Retry-After": ".01"}
            return self.success(body, timeout)
        client = self.client(transport=transport)
        client.register_worker("a")
        self.assertEqual(client.request("a", self.body())[0], 200)
        summary = client.summary()
        self.assertEqual((summary["logical_requests"], summary["provider_attempts"]), (1, 2))
        self.assertFalse(client.workers["a"]["refused"].is_set())
        self.assertGreater(summary["rejected_request_estimated_cost_allowance_usd"], 0)
        self.assertGreater(summary["unknown_usage_token_allowance"], 0)
        self.assertEqual(summary["pacing"]["reserved_requests"], 0)
        self.assertEqual(client.records[0]["safe_response_headers"]["retry-after"], ".01")
        self.assertTrue(client.records[0]["rejected_billing_unknown"])
        self.assertEqual(client.records[1]["logical_attempt"], 2)
        for body in bodies:
            self.assertEqual(body["system"], "unchanged prefix")
            self.assertEqual(body["model"], MODEL)
            self.assertEqual(body["output_config"], {"effort": "max"})
            notice = body["messages"][-1]["content"][-1]["text"]
            self.assertIn("Logical model request 1/6", notice)
            self.assertIn("do not consume another logical request", notice)

    def test_three_attempt_max_and_six_distinct_logical_requests(self):
        client = self.client(transport=lambda *args: (429, b'{}', "application/json", {"Retry-After": "0"}))
        client.register_worker("a")
        for _ in range(6):
            self.assertEqual(client.request("a", self.body())[0], 429)
        self.assertEqual(client.summary()["requests"], 18)
        self.assertEqual(client.summary()["worker_logical_requests"]["a"], 6)
        with self.assertRaisesRegex(BudgetRefused, "Logical"):
            client.request("a", self.body())
        self.assertEqual(client.summary()["requests"], 18)

    def test_retry_attempts_cannot_exceed_global_or_worker_cap(self):
        for limits in ({"max_calls": 1}, {"per_worker_calls": 1}):
            with self.subTest(limits=limits):
                client = self.client(**limits, transport=lambda *args: (429, b'{}', "application/json", {"Retry-After": "0"}))
                client.register_worker("a")
                with self.assertRaises(BudgetRefused):
                    client.request("a", self.body())
                self.assertEqual(client.calls, 1)
                self.assertEqual(client.logical_requests, 1)

    def test_429_allowance_blocks_retry_when_estimated_cost_cannot_fit(self):
        client = self.client(max_cost=.0005,
            transport=lambda *args: (429, b'{}', "application/json", {"Retry-After": "0"}))
        client.register_worker("a")
        with self.assertRaises(BudgetRefused):
            client.request("a", self.body())
        self.assertEqual(client.calls, 1)
        self.assertLessEqual(client.cost, client.max_cost)

    def test_timeout_is_not_retried(self):
        def transport(*args):
            raise TimeoutError("ambiguous timeout")
        client = self.client(transport=transport)
        client.register_worker("a")
        self.assertEqual(client.request("a", self.body())[0], 502)
        self.assertEqual(client.calls, 1)
        self.assertGreater(client.summary()["unknown_usage_token_allowance"], 0)

    def test_invalid_retry_after_uses_bounded_backoff_and_staggers_attempts(self):
        starts = []
        def transport(body, timeout):
            starts.append(time.monotonic())
            if len(starts) <= 2:
                return 429, b'{}', "application/json", {"Retry-After": "invalid"}
            return self.success(body, timeout)
        client = self.client(transport=transport, base_backoff_seconds=.02,
                             max_backoff_seconds=.03, start_stagger_seconds=.01)
        client.register_worker("a")
        self.assertEqual(client.request("a", self.body())[0], 200)
        self.assertEqual([r["applied_retry_delay_seconds"] for r in client.records], [.02, .03, None])
        self.assertGreaterEqual(starts[1] - starts[0], .019)
        self.assertGreaterEqual(starts[2] - starts[1], .029)

    def test_submit_deadline_and_close_interrupt_cooldown(self):
        for cancel in ("submit", "deadline", "close"):
            with self.subTest(cancel=cancel):
                client = self.client(transport=lambda *args: (429, b'{}', "application/json", {"Retry-After": "30"}))
                session = SimpleNamespace(submitted=None)
                client.register_worker("a", session=session)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(client.request, "a", self.body())
                    self.wait_until(lambda: client.calls == 1 and len(client.pacing_queue) == 1)
                    if cancel == "close":
                        client.close(wait_timeout=2)
                    else:
                        with client.condition:
                            if cancel == "submit":
                                session.submitted = {"frozen": True}
                            else:
                                client.deadline = time.monotonic() - 1
                            client.condition.notify_all()
                    with self.assertRaises(BudgetRefused):
                        pending.result(timeout=2)
                self.assertEqual(client.calls, 1)
                self.assertEqual(client.summary()["active_logical_requests"], 0)
                self.assertFalse(client.pacing_leases)

    def test_three_actual_provider_calls_overlap_but_fourth_waits(self):
        release, lock = threading.Event(), threading.Lock()
        active, peak = 0, 0
        def transport(body, timeout):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            release.wait(2)
            with lock:
                active -= 1
            return self.success(body, timeout)
        client = self.client(transport=transport)
        for i in range(15):
            client.register_worker(f"w{i}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(client.request, f"w{i}", self.body()) for i in range(4)]
            try:
                self.wait_until(lambda: peak == 3)
                self.assertEqual(client.calls, 3)
                self.assertEqual(peak, 3)
            finally:
                release.set()
            self.assertEqual([f.result(timeout=2)[0] for f in futures], [200] * 4)
        self.assertEqual(client.summary()["peak_active_requests"], 3)

    def test_pacing_reservations_wait_before_api_admission(self):
        entered, release = threading.Event(), threading.Event()
        def transport(body, timeout):
            entered.set()
            release.wait(2)
            return self.success(body, timeout)
        client = self.client(transport=transport, prompt_tpm=2200, uncached_prompt_tpm=2200)
        for worker in ("a", "b"):
            client.register_worker(worker)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.request, "a", self.body())
            self.assertTrue(entered.wait(2))
            second = pool.submit(client.request, "b", self.body())
            try:
                self.wait_until(lambda: len(client.pacing_queue) == 1)
                self.assertEqual(client.calls, 1)
                self.assertEqual(client.active, 1)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=2)[0], 200)
            self.assertEqual(second.result(timeout=2)[0], 200)

    def test_header_caps_only_lower_and_output_is_clamped(self):
        seen = []
        def transport(body, timeout):
            seen.append(json.loads(body)["max_tokens"])
            return (*self.success(body, timeout), {"x-ratelimit-limit-tokens-generated": "40",
                "x-ratelimit-limit-tokens-prompt": "999999"})
        client = self.client(transport=transport)
        client.register_worker("a")
        client.request("a", self.body())
        client.request("a", self.body())
        self.assertEqual(seen, [100, 40])
        self.assertEqual(client.summary()["pacing"]["effective_tpm"]["prompt"], 120000)
        self.assertTrue(client.records[1]["output_cap_reduced"])

    def test_temporary_scale_recovers_but_hard_prompt_impossibility_stops(self):
        client = self.client(prompt_tpm=1800, uncached_prompt_tpm=1800)
        client.register_worker("a")
        with client.condition:
            client.rate_scale = .6
            client.last_429_at = time.monotonic()
        self.assertEqual(client.request("a", self.body())[0], 200)
        self.assertEqual(client.rate_scale, 1)
        with self.assertRaisesRegex(BudgetRefused, "pacing ceiling"):
            client.request("a", {"messages": [{"role": "user", "content": "x" * 10000}]})


if __name__ == "__main__":
    unittest.main()
