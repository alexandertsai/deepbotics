"""Check shared simulator budgets and evidence-based swarm selection."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import task08_swarm as S


class SwarmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.coordinator = S.Coordinator(self.root / "swarm",
            S.P.ROOT / "outputs/task08-reference-baseline-01",
            S.P.ROOT / "outputs/task08-pilot-02", seconds=60)
        self.coordinator.start()

    def session(self, name):
        session = S.SwarmSession(self.root / name, self.coordinator, name, name, "Try one change")
        session.start()
        return session

    def test_agent_claims_do_not_change_selection(self):
        session = self.session("claims")
        session.dispatch("share_finding", {"notes": "I measured 100 percent success"})
        session.dispatch("edit_file", {"name": "controller.py", "old": "import math", "new": "import math\n# untested"})
        selection = self.coordinator.select()
        self.assertEqual(selection["worker"], "unchanged_reference")

    def test_repeated_starter_probe_uses_cache(self):
        session = self.session("cache")
        with patch.object(S.P, "docker_evaluate") as evaluate:
            result = session.dispatch("test_candidate", {"hypothesis": "Check starter"})
        self.assertTrue(result["cached"])
        self.assertEqual(self.coordinator.tests, 0)
        evaluate.assert_not_called()

    def test_four_workers_cannot_exceed_three_new_probes(self):
        sessions = [self.session(f"worker{i}") for i in range(4)]
        for i, session in enumerate(sessions):
            file = session.candidate / "controller.py"
            file.write_text(file.read_text() + f"\n# candidate {i}\n")
        active, peak = 0, 0
        lock = S.threading.Lock()
        def evaluate(*args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.02)
            with lock:
                active -= 1
            return copy.deepcopy(self.coordinator.baseline_report)
        with patch.object(S.P, "docker_evaluate", side_effect=evaluate) as run:
            with ThreadPoolExecutor(max_workers=4) as pool:
                replies = list(pool.map(lambda s: s.tool("test_candidate", {"hypothesis": "Try"}), sessions))
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sum(r["isError"] for r in replies), 1)
        self.assertEqual(peak, 1)
        self.assertEqual(self.coordinator.tests, 3)

    def test_scorer_uses_worst_arm_and_rejects_incomplete_evidence(self):
        report = copy.deepcopy(self.coordinator.baseline_report)
        before = S.partial_score(report)
        report["arms"]["panda"]["checkpoints"]["S3.static_margin"] = 0
        self.assertAlmostEqual(S.partial_score(report), before - .15)
        del report["arms"]["ur5e"]
        with self.assertRaises(ValueError):
            S.partial_score(report)

    def test_identical_inflight_candidates_share_one_simulation(self):
        sessions = [self.session(f"same{i}") for i in range(4)]
        for session in sessions:
            file = session.candidate / "controller.py"
            file.write_text(file.read_text() + "\n# same candidate\n")
        def evaluate(*args, **kwargs):
            time.sleep(.05)
            return copy.deepcopy(self.coordinator.baseline_report)
        with patch.object(S.P, "docker_evaluate", side_effect=evaluate) as run:
            with ThreadPoolExecutor(max_workers=4) as pool:
                replies = list(pool.map(lambda s: s.tool("test_candidate", {"hypothesis": "Try"}), sessions))
        self.assertEqual(run.call_count, 1)
        self.assertFalse(any(r["isError"] for r in replies))
        self.assertEqual(self.coordinator.tests, 1)

    def test_expanded_budget_is_reflected_in_worker_prompt(self):
        c = S.Coordinator(self.root / "expanded", S.P.ROOT / "outputs/task08-reference-baseline-01",
            S.P.ROOT / "outputs/task08-pilot-02", seconds=1800, tests=8, calls=48,
            calls_per_worker=12, tokens=3000000, output_tokens=16384, cost=3,
            tests_per_worker=2)
        s = S.SwarmSession(self.root / "expanded-worker", c, "a", "timing", "Try timing")
        self.assertIn("48 provider requests", s.prompt)
        self.assertIn("12 model requests", s.prompt)
        self.assertIn("8 NEW development probes", s.prompt)
        self.assertIn("Starter robot.xml:", s.prompt)

    def test_per_worker_probe_limit_leaves_budget_for_peers(self):
        self.coordinator.max_tests = 8
        self.coordinator.tests_per_worker = 2
        a, b = self.session("a"), self.session("b")
        with patch.object(S.P, "docker_evaluate", return_value=self.coordinator.baseline_report) as run:
            for i in range(3):
                file = a.candidate / "controller.py"
                file.write_text(file.read_text() + f"\n# revision {i}\n")
                reply = a.tool("test_candidate", {"hypothesis": "Try"})
                self.assertEqual(reply["isError"], i == 2)
            file = b.candidate / "controller.py"
            file.write_text(file.read_text() + "\n# peer revision\n")
            self.assertFalse(b.tool("test_candidate", {"hypothesis": "Peer try"})["isError"])
        self.assertEqual(run.call_count, 3)
        self.assertEqual(a.tests, 2)
        self.assertEqual(b.tests, 1)

    def test_rehearsal_tests_frozen_submission_not_later_working_copy(self):
        session = self.session("rehearsal")
        file = session.candidate / "controller.py"
        file.write_text(file.read_text() + "\n# frozen candidate\n")
        session.dispatch("submit", {})
        frozen = (session.root / "frozen/controller.py").read_bytes()
        file.write_text("invalid working copy")
        observed = []
        def evaluate(bundle, *args, **kwargs):
            observed.append((bundle / "controller.py").read_bytes())
            return copy.deepcopy(self.coordinator.baseline_report)
        with patch.object(S.P, "docker_evaluate", side_effect=evaluate):
            outcome = self.coordinator.rehearse_remaining([session])
        self.assertEqual(observed, [frozen])
        self.assertIn("partial_score", outcome[0])
        self.assertEqual(self.coordinator.tests, 1)


if __name__ == "__main__":
    unittest.main()
