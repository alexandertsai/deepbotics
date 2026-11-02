"""Shared request budgets and FIFO admission without model calls."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task08_swarm_client import BudgetRefused, MODEL, SharedClient


class SharedClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.key = "test-fireworks-credential"

    def client(self, **kwargs):
        values = dict(root=self.root, keys={"FIREWORKS_API_KEY": self.key},
                      deadline=time.monotonic() + 10, start_server=False,
                      transport=self.success)
        values.update(kwargs)
        client = SharedClient(**values)
        self.addCleanup(client.close)
        return client

    @staticmethod
    def success(body, timeout):
        return 200, json.dumps({"content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 5, "cache_read_input_tokens": 7,
                      "output_tokens": 3}}).encode(), "application/json"

    @staticmethod
    def body(label="a"):
        return {"model": "untrusted-other-model", "max_tokens": 99999,
                "messages": [{"role": "user", "content": label}]}

    @staticmethod
    def wait_until(predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError("Thread did not reach expected state")
            time.sleep(.005)

    def test_global_request_cap_is_shared_across_workers(self):
        client = self.client(max_calls=2)
        for worker in ("a", "b", "c", "d"):
            client.register_worker(worker)
        def call(worker):
            try:
                return client.request(worker, self.body(worker))[0]
            except BudgetRefused:
                return "refused"
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(call, ("a", "b", "c", "d")))
        self.assertEqual(results.count(200), 2)
        self.assertEqual(results.count("refused"), 2)
        self.assertEqual(client.summary()["requests"], 2)
        self.assertTrue(client.summary()["budget_exhausted"])

    def test_submitted_worker_cannot_start_another_provider_request(self):
        client = self.client()
        session = SimpleNamespace(submitted={"sha256": "frozen"})
        client.register_worker("a", session=session)
        client.register_worker("b")
        with self.assertRaises(BudgetRefused):
            client.request("a", self.body())
        self.assertEqual(client.summary()["requests"], 0)
        self.assertFalse(client.exhausted.is_set())
        self.assertEqual(client.request("b", self.body())[0], 200)

    def test_queued_worker_submission_is_checked_before_admission(self):
        entered, release = threading.Event(), threading.Event()
        def transport(body, timeout):
            entered.set()
            release.wait(2)
            return self.success(body, timeout)
        client = self.client(transport=transport)
        session = SimpleNamespace(submitted=None)
        client.register_worker("a")
        client.register_worker("b", session=session)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.request, "a", self.body())
            self.assertTrue(entered.wait(2))
            second = pool.submit(client.request, "b", self.body())
            try:
                self.wait_until(lambda: client.summary()["queued_requests"] == 1)
                session.submitted = {"sha256": "frozen"}
                with client.condition:
                    client.condition.notify_all()
                with self.assertRaises(BudgetRefused):
                    second.result(timeout=2)
                self.assertEqual(client.summary()["requests"], 1)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=2)[0], 200)
    def test_worker_limit_does_not_exhaust_other_workers(self):
        client = self.client(per_worker_calls=1)
        for worker in ("a", "b"):
            client.register_worker(worker)
        client.request("a", self.body())
        with self.assertRaises(BudgetRefused):
            client.request("a", self.body())
        self.assertFalse(client.exhausted.is_set())
        self.assertEqual(client.request("b", self.body())[0], 200)
        self.assertEqual(client.summary()["worker_requests"], {"a": 1, "b": 1})

    def test_fifo_requests_never_overlap_and_commit_usage_first(self):
        first_entered, release = threading.Event(), threading.Event()
        state = {"active": 0, "peak": 0, "order": []}
        lock = threading.Lock()
        def transport(body, timeout):
            label = json.loads(body)["messages"][0]["content"].split("\n\nBudget notice:", 1)[0]
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["order"].append(label)
            if label == "a":
                first_entered.set()
                if not release.wait(2):
                    raise TimeoutError("test release")
            with lock:
                state["active"] -= 1
            return self.success(body, timeout)
        client = self.client(transport=transport)
        for worker in ("a", "b", "c"):
            client.register_worker(worker)
        with ThreadPoolExecutor(max_workers=3) as pool:
            a = pool.submit(client.request, "a", self.body("a"))
            self.assertTrue(first_entered.wait(2))
            b = pool.submit(client.request, "b", self.body("b"))
            self.wait_until(lambda: client.summary()["queued_requests"] == 1)
            c = pool.submit(client.request, "c", self.body("c"))
            self.wait_until(lambda: client.summary()["queued_requests"] == 2)
            release.set()
            self.assertEqual([a.result()[0], b.result()[0], c.result()[0]], [200, 200, 200])
        self.assertEqual(state["order"], ["a", "b", "c"])
        self.assertEqual(state["peak"], 1)
        self.assertEqual(client.summary()["reported_tokens"], 45)

    def test_threshold_is_checked_before_next_serialized_call(self):
        client = self.client(max_tokens=10)
        client.register_worker("a")
        client.register_worker("b")
        self.assertEqual(client.request("a", self.body())[0], 200)
        self.assertEqual(client.summary()["reported_tokens"], 15)
        with self.assertRaises(BudgetRefused):
            client.request("b", self.body())
        self.assertEqual(client.summary()["requests"], 1)
        self.assertEqual(client.summary()["usage"]["cache_read_input_tokens"], 7)

    def test_model_output_cap_and_credentials_are_not_logged(self):
        seen = []
        def transport(body, timeout):
            seen.append(json.loads(body))
            return self.success(body, timeout)
        client = self.client(transport=transport)
        token = client.register_worker("a")
        body = self.body(self.key + token)
        client.request("a", body)
        self.assertEqual(seen[0]["model"], MODEL)
        self.assertEqual(seen[0]["max_tokens"], 8192)
        self.assertEqual(seen[0]["output_config"], {"effort": "max"})
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(self.key, path.read_text())
                self.assertNotIn(token, path.read_text())

    def test_estimated_cost_refusal_forwards_nothing(self):
        calls = []
        def transport(body, timeout):
            calls.append(body)
            return self.success(body, timeout)
        client = self.client(max_cost=.000001, transport=transport)
        client.register_worker("a")
        with self.assertRaises(BudgetRefused):
            client.request("a", self.body())
        self.assertEqual(calls, [])
        self.assertEqual(client.summary()["requests"], 0)

    def test_deadline_refuses_without_provider_request(self):
        client = self.client(deadline=time.monotonic() - 1)
        client.register_worker("a")
        with self.assertRaises(BudgetRefused):
            client.request("a", self.body())
        self.assertEqual(client.summary()["requests"], 0)

    def test_provider_failure_consumes_one_request_and_unknown_usage_is_conservative(self):
        def transport(body, timeout):
            raise TimeoutError("unknown provider outcome")
        client = self.client(transport=transport)
        client.register_worker("a")
        self.assertEqual(client.request("a", self.body())[0], 502)
        summary = client.summary()
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["reported_tokens"], 0)
        self.assertGreater(summary["unknown_usage_token_allowance"], 8192)
        self.assertEqual(summary["errors"], 1)

    def test_consumed_global_budget_does_not_abort_paid_response_actions(self):
        from unittest.mock import Mock
        client = self.client(start_server=True)
        client.exhausted.set()
        worker_root = self.root / "worker"
        workspace = worker_root / "workspace"
        workspace.mkdir(parents=True)
        session = SimpleNamespace(root=worker_root, workspace=workspace, prompt="test",
            deadline=client.deadline, submitted=None, history=[], tests=0, lock=threading.RLock())
        def freeze(destination):
            destination.mkdir()
            return {"sha256": "test-manifest"}
        session.freeze = freeze
        class Process:
            pid = 1
            returncode = None
            polls = 0
            def poll(self):
                self.polls += 1
                if self.returncode is None and self.polls == 6:
                    session.history.append("paid response tool action completed")
                    self.returncode = 0
                return self.returncode
        process = Process()
        premature_stops = []
        def stop(target):
            if target.poll() is None:
                premature_stops.append(True)
                target.returncode = -15
        with patch("claude_mac_evaluation.mcp_server", return_value=Mock()), \
             patch("task08_swarm_client.subprocess.Popen", return_value=process), \
             patch("task08_swarm_client.time.sleep"), patch.object(client, "_stop", side_effect=stop):
            result = client.run_worker(session, "a")
        self.assertEqual(premature_stops, [])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["returncode"], 0)

    def test_local_http_worker_route_requires_its_own_token(self):
        client = self.client(start_server=True)
        token = client.register_worker("a")
        url = f"http://127.0.0.1:{client.server.server_port}/worker/a/v1/messages"
        def request(credential):
            return urllib.request.urlopen(urllib.request.Request(url,
                data=json.dumps(self.body()).encode(),
                headers={"x-api-key": credential, "Content-Type": "application/json"}), timeout=2)
        with self.assertRaises(urllib.error.HTTPError) as failure:
            request("invalid")
        self.assertEqual(failure.exception.code, 401)
        self.assertEqual(client.summary()["requests"], 0)
        with request(token) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response)["content"][0]["text"], "ok")
        self.assertEqual(client.summary()["requests"], 1)

    def test_expanded_allowances_forward_beyond_old_caps_and_enforce_new_worker_cap(self):
        seen = []
        def transport(body, timeout):
            seen.append(json.loads(body))
            return self.success(body, timeout)
        client = self.client(max_calls=48, per_worker_calls=12, max_tokens=3_000_000,
            max_output_tokens=16384, max_cost=3.0, transport=transport)
        client.register_worker("a")
        client.register_worker("b")
        for _ in range(12):
            self.assertEqual(client.request("a", self.body())[0], 200)
        with self.assertRaises(BudgetRefused):
            client.request("a", self.body())
        self.assertEqual(client.request("b", self.body())[0], 200)
        self.assertEqual(len(seen), 13)
        self.assertTrue(all(request["max_tokens"] == 16384 for request in seen))
        self.assertEqual(client.summary()["limits"], {"requests": 48,
            "reported_token_stop_threshold": 3_000_000, "output_tokens_per_request": 16384,
            "requests_per_worker": 12, "estimated_cost_usd": 3.0})

    def test_expanded_global_allowance_is_enforced_across_workers(self):
        client = self.client(max_calls=14, per_worker_calls=12, max_tokens=3_000_000,
            max_output_tokens=16384, max_cost=3.0)
        client.register_worker("a")
        client.register_worker("b")
        for _ in range(12):
            client.request("a", self.body())
        for _ in range(2):
            client.request("b", self.body())
        with self.assertRaises(BudgetRefused):
            client.request("b", self.body())
        self.assertEqual(client.summary()["requests"], 14)
        self.assertTrue(client.summary()["budget_exhausted"])

    def test_budget_notice_preserves_system_and_tool_result_order(self):
        seen = []
        def transport(body, timeout):
            seen.append(json.loads(body))
            return self.success(body, timeout)
        client = self.client(max_calls=48, per_worker_calls=12, max_tokens=3_000_000,
            max_output_tokens=16384, max_cost=3.0, transport=transport)
        client.register_worker("a")
        system = [{"type": "text", "text": "fixed prefix", "cache_control": {"type": "ephemeral"}}]
        content = [{"type": "tool_result", "tool_use_id": "test", "content": "public observation"}]
        body = {"system": system, "messages": [{"role": "user", "content": content}]}
        for _ in range(10):
            client.request("a", body)
        self.assertEqual(seen[0]["system"], system)
        self.assertEqual(seen[0]["messages"][0]["content"][:-1], content)
        self.assertIn("worker request 1/12", seen[0]["messages"][0]["content"][-1]["text"])
        self.assertIn("remaining after this response: 11", seen[0]["messages"][0]["content"][-1]["text"])
        final_notice = seen[-1]["messages"][0]["content"][-1]["text"]
        self.assertIn("worker request 10/12", final_notice)
        self.assertIn("implementing or editing a concrete candidate", final_notice)
        self.assertEqual(body["messages"][0]["content"], content)

    def test_budget_notice_supports_user_text_and_system_fallback(self):
        body = {"system": "fixed", "messages": [{"role": "user", "content": "question"}]}
        result = SharedClient._append_notice(body, "notice")
        self.assertEqual(result["system"], "fixed")
        self.assertEqual(result["messages"][0]["content"], "question\n\nnotice")
        self.assertEqual(body["messages"][0]["content"], "question")
        self.assertEqual(SharedClient._append_notice({"system": "fixed"}, "notice")["system"],
                         "fixed\n\nnotice")

    def test_nonfinite_or_invalid_budgets_are_rejected(self):
        for values in ({"max_calls": True}, {"per_worker_calls": 1.5}, {"max_tokens": 0},
                       {"max_output_tokens": 32769}, {"max_cost": float("nan")},
                       {"max_cost": float("inf")}, {"deadline": float("nan")},
                       {"max_concurrent": 9}, {"max_workers": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.client(**values)

    def test_eight_requests_overlap_and_report_slot_metrics(self):
        entered, release = threading.Event(), threading.Event()
        active, peak = 0, 0
        lock = threading.Lock()
        def transport(body, timeout):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 8:
                    entered.set()
            if not release.wait(3):
                raise TimeoutError("test release")
            with lock:
                active -= 1
            return self.success(body, timeout)
        client = self.client(max_concurrent=8, max_workers=8, transport=transport)
        for index in range(8):
            client.register_worker(str(index))
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(client.request, str(i), self.body(str(i))) for i in range(8)]
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(client.summary()["active_requests"], 8)
            finally:
                release.set()
            self.assertEqual([f.result()[0] for f in futures], [200] * 8)
        summary = client.summary()
        self.assertEqual(peak, 8)
        self.assertEqual(summary["peak_active_requests"], 8)
        self.assertEqual(summary["active_requests"], 0)
        self.assertEqual(summary["inflight_token_reservations"], 0)
        self.assertGreater(summary["aggregate_active_slot_seconds"], summary["provider_busy_wall_seconds"])

    def test_worker_registration_uses_configured_limit(self):
        client = self.client(max_workers=8)
        for index in range(8):
            client.register_worker(str(index))
        with self.assertRaises(ValueError):
            client.register_worker("ninth")

    def test_parallel_requests_respect_slots_and_global_request_cap(self):
        active, peak = 0, 0
        lock = threading.Lock()
        def transport(body, timeout):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.025)
            with lock:
                active -= 1
            return self.success(body, timeout)
        client = self.client(max_calls=5, max_concurrent=3, max_workers=8, transport=transport)
        for index in range(8):
            client.register_worker(str(index))
        def call(index):
            try:
                return client.request(str(index), self.body(str(index)))[0]
            except BudgetRefused:
                return "refused"
        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(call, range(8)))
        self.assertEqual(outcomes.count(200), 5)
        self.assertEqual(outcomes.count("refused"), 3)
        self.assertLessEqual(peak, 3)
        self.assertGreater(peak, 1)
        self.assertEqual(client.summary()["requests"], 5)

    def reservation_case(self, *, failure=False, cost=False):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def transport(body, timeout):
            calls.append(body)
            if len(calls) == 1:
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test release")
                if failure:
                    raise TimeoutError("unknown provider outcome")
            return self.success(body, timeout)
        limits = {"max_cost": .0009, "max_output_tokens": 500} if cost else {
            "max_tokens": 6000, "max_output_tokens": 100}
        client = self.client(max_concurrent=8, max_workers=8, transport=transport, **limits)
        client.register_worker("a")
        client.register_worker("b")
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.request, "a", self.body())
            self.assertTrue(entered.wait(2))
            second = pool.submit(client.request, "b", self.body())
            try:
                self.wait_until(lambda: client.summary()["queued_requests"] == 1 or second.done())
                self.assertEqual(len(calls), 1)
                snapshot = client.summary()
                self.assertLessEqual(snapshot["inflight_token_reservations"], client.max_tokens)
                self.assertLessEqual(snapshot["inflight_estimated_cost_usd"], client.max_cost)
            finally:
                release.set()
            self.assertEqual(first.result()[0], 502 if failure else 200)
            if failure:
                with self.assertRaises(BudgetRefused):
                    second.result()
                self.assertEqual(len(calls), 1)
            else:
                self.assertEqual(second.result()[0], 200)
                self.assertEqual(len(calls), 2)
        self.assertEqual(client.summary()["active_requests"], 0)
        self.assertEqual(client.summary()["inflight_token_reservations"], 0)
        return client

    def test_inflight_token_reservations_wait_then_release(self):
        self.reservation_case()

    def test_unknown_failure_keeps_token_allowance_charged(self):
        client = self.reservation_case(failure=True)
        self.assertGreater(client.summary()["unknown_usage_token_allowance"], 4096)

    def test_inflight_cost_reservations_wait_then_release(self):
        self.reservation_case(cost=True)

    def test_unknown_failure_keeps_cost_allowance_charged(self):
        client = self.reservation_case(failure=True, cost=True)
        self.assertGreater(client.summary()["estimated_cost_usd"], .0004)

    def test_close_cancels_every_active_http_client(self):
        entered = threading.Event()
        created = []
        lock = threading.Lock()
        class HttpClient:
            def __init__(inner, **kwargs):
                inner.cancelled = threading.Event()
                with lock:
                    created.append(inner)
            def __enter__(inner):
                return inner
            def __exit__(inner, *_):
                inner.close()
            def close(inner):
                inner.cancelled.set()
            def post(inner, *_args, **_kwargs):
                with lock:
                    if len(created) == 8:
                        entered.set()
                if not inner.cancelled.wait(3):
                    raise TimeoutError("test cleanup")
                time.sleep(.025)
                raise OSError("client closed")
        client = self.client(max_concurrent=8, max_workers=8)
        client.transport = client._post_upstream
        for i in range(8):
            client.register_worker(str(i))
        with patch("httpx2.Client", side_effect=HttpClient), ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(client.request, str(i), self.body()) for i in range(8)]
            self.assertTrue(entered.wait(2))
            self.wait_until(lambda: len(client.active_clients) == 8)
            client.close()
            self.assertEqual(client.summary()["active_requests"], 0)
            self.assertEqual(len(client.records), 8)
            self.assertEqual([f.result()[0] for f in futures], [502] * 8)
        self.assertTrue(all(instance.cancelled.is_set() for instance in created))
        self.assertEqual(len(client.active_clients), 0)
        self.assertEqual(client.summary()["active_requests"], 0)


if __name__ == "__main__":
    unittest.main()
