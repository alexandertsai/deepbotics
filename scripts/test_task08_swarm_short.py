"""Six-worker interface and bounded simulator checks without inference."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import copy
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task08_swarm_short as W


class ShortSwarmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.coordinator = W.Coordinator(self.root / "run", W.P.ROOT / "outputs/task08-reference-baseline-01",
            W.P.ROOT / "outputs/task08-pilot-02", seconds=60, tests=12, calls=216,
            calls_per_worker=36, tokens=6000000, cost=3, tests_per_worker=2)
        self.coordinator.logical_calls = 12
        self.coordinator.active_worker_limit = 6
        self.coordinator.start()

    def sessions(self):
        sessions = [W.Session(self.coordinator.root / name, self.coordinator, name, name, brief)
                    for name, brief in W.ROLES]
        for session in sessions:
            session.start()
        return sessions

    def test_six_isolated_sessions_have_correct_logical_prompt_and_deadline(self):
        sessions = self.sessions()
        self.assertEqual(len(sessions), 6)
        self.assertEqual(len({s.candidate for s in sessions}), 6)
        for session in sessions:
            self.assertIn("of 6 independent DeepSeek workers", session.prompt)
            self.assertNotIn("four independent DeepSeek workers", session.prompt)
            self.assertIn("at most 12 logical model requests", session.prompt)
            self.assertNotIn("at most 36 model requests", session.prompt)
            self.assertEqual(session.deadline, self.coordinator.deadline)

    def test_two_probes_per_worker_are_serial_and_status_omits_reports(self):
        sessions = self.sessions()
        active, peak = 0, 0
        lock = threading.Lock()
        def evaluate(*args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.002)
            with lock:
                active -= 1
            return copy.deepcopy(self.coordinator.baseline_report)
        def run(session):
            for i in range(2):
                path = session.candidate / "controller.py"
                path.write_text(path.read_text() + f"\n# {session.worker_id} {i}\n")
                session.dispatch("test_candidate", {"hypothesis": "Small candidate revision"})
        with patch.object(W.P, "docker_evaluate", side_effect=evaluate) as simulator:
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(run, sessions))
        self.assertEqual(simulator.call_count, 12)
        self.assertEqual(peak, 1)
        self.assertEqual(self.coordinator.tests, 12)
        self.assertEqual([s.tests for s in sessions], [2] * 6)
        status = self.coordinator.status()
        self.assertEqual(status["workers"], 6)
        self.assertTrue(all("report" not in row and "hypothesis" not in row for row in status["probes"]))

    def test_runner_obeys_active_process_limit(self):
        sessions = self.sessions()
        active, peak = 0, 0
        lock = threading.Lock()
        class Client:
            def run_worker(self, session, worker_id):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.01)
                with lock:
                    active -= 1
                return {"worker_id": worker_id, "api_requests": 0}
        results = W.run_workers(self.coordinator, Client(), sessions, 3)
        self.assertEqual(len(results), 6)
        self.assertEqual(peak, 3)

    def test_separate_lock_instances_serialize_and_timeout(self):
        path = self.root / "sim.lock"
        first = W.SimulationLock(path, lambda: time.monotonic() + 2)
        second = W.SimulationLock(path, lambda: time.monotonic() - 1)
        with first:
            with self.assertRaises(TimeoutError):
                with second:
                    self.fail("Expired lock entered")
        with W.SimulationLock(path, lambda: time.monotonic() + 2):
            self.assertTrue(path.exists())

    def test_default_budgets_and_primary_credentials(self):
        args = W.parser().parse_args([])
        self.assertEqual((args.workers, args.tests, args.tests_per_worker, args.logical_calls_per_worker), (6, 12, 2, 12))
        self.assertEqual((args.max_cost, args.max_tokens), (6, 6000000))
        self.assertEqual((args.active_workers,args.api_concurrency),(6,3))
        import mac_evaluation as M
        with patch.object(M, "load_keys", return_value={"FIREWORKS_API_KEY": "primary", "DEEPSEEK_API_KEY": "other"}):
            self.assertEqual(W.primary_keys(), {"FIREWORKS_API_KEY": "primary"})
        stop = W.hard_stop_timestamp(now=datetime.fromisoformat("2026-11-02T12:00:00-08:00").timestamp())
        self.assertEqual(datetime.fromtimestamp(stop, W.ZoneInfo("America/Los_Angeles")).hour, 15)

    def test_pooled_summary_remains_usable_without_flat_worker_limits(self):
        class Pool:
            def summary(self):
                return {"requests": 3, "logical_requests": 2, "reported_tokens": 42,
                    "estimated_cost_usd": .01, "active_requests": 1,
                    "per_account": [{"worker_logical_requests": {"a": 1}},
                                    {"worker_logical_requests": {"b": 1}}]}
        self.coordinator.client = Pool()
        budget = self.coordinator.status()["api_budget"]
        self.assertEqual(budget["worker_logical_requests"], {"a": 1, "b": 1})
        self.assertEqual(budget["limits"]["max_cost"], 3)


if __name__ == "__main__":
    unittest.main()
