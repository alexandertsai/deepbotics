import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import task09_planner_analysis as A
from services.controller.storage import snapshot


class NoDockerLock:
    def __init__(self, path, deadline, log_path):
        self.deadline = deadline

    def __enter__(self):
        if time.monotonic() >= self.deadline():
            raise TimeoutError("Expired analysis lock")
        return self

    def __exit__(self, *args):
        pass


class PlannerOperatorAnalysis(unittest.TestCase):
    def fixture(self, root, codes):
        source = root / "source"
        source.mkdir()
        sessions = []
        for index, code in enumerate(codes):
            (source / "trim.py").write_text(code)
            worker = root / f"worker-{index}"
            worker.mkdir()
            manifest = snapshot(source, worker / "frozen", max_bytes=1024, workspace_path="code/task09")
            (worker / "agent-summary.json").write_text("{}")
            sessions.append(SimpleNamespace(root=worker, worker_id=f"worker-{index}", submitted=manifest))
        selected = snapshot(sessions[0].root / "frozen", root / "selected", max_bytes=1024, workspace_path="code/task09")
        (root / "selection.json").write_text(json.dumps({"manifest": selected}))
        (root / "final").mkdir()
        (root / "final/reward.json").write_text(json.dumps({"reward": .25}))
        return sessions, selected

    def analyze(self, root, sessions, selected, evaluator, stop=None):
        with patch.object(A, "MemoryGatedSimulationLock", NoDockerLock), patch.object(A.Q, "evaluate", side_effect=evaluator) as evaluate:
            result = A.analyze_worker_finals(root, sessions, selected, stop or time.time() + 10, root / "task09.lock")
        return result, evaluate

    def test_selected_reuse_and_distinct_design_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sessions, selected = self.fixture(root, ["a", "a", "b", "b", "c", "c"])
            def evaluator(bundle, output, **kwargs):
                self.assertTrue(output.is_relative_to(root / "operator-analysis"))
                self.assertTrue(kwargs["final"])
                self.assertGreater(kwargs["timeout"], 0)
                output.mkdir(parents=True)
                (output / "reward.json").write_text('{"reward":0.5}')
                return {"reward": .5}
            result, evaluate = self.analyze(root, sessions, selected, evaluator)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(result["unique_private_evaluations"], 2)
            self.assertEqual([row["score"] for row in result["workers"]], [25, 25, 50, 50, 50, 50])
            self.assertFalse(result["feedback_available_to_workers"])
            self.assertFalse(result["selection_used_analysis_results"])
            self.assertEqual(json.loads((root / "selection.json").read_text())["manifest"], selected)

    def test_live_or_unfinished_worker_blocks_every_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sessions, selected = self.fixture(root, list("abcdef"))
            (sessions[-1].root / "agent-summary.json").unlink()
            with patch.object(A.Q, "evaluate") as evaluate:
                with self.assertRaises(ValueError):
                    A.analyze_worker_finals(root, sessions, selected, time.time() + 10, root / "lock")
                evaluate.assert_not_called()
            self.assertFalse((root / "operator-analysis").exists())

    def test_corrupt_frozen_design_has_no_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sessions, selected = self.fixture(root, list("abcdef"))
            target = sessions[2].root / "frozen/trim.py"
            target.chmod(0o600)
            target.write_text("tampered")
            result, evaluate = self.analyze(root, sessions, selected, lambda *args, **kwargs: {"reward": .5})
            self.assertIsNone(result["workers"][2]["score"])
            self.assertIn("changed", result["workers"][2]["error"])
            self.assertEqual(evaluate.call_count, 4)

    def test_deadline_failures_are_recorded_without_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sessions, selected = self.fixture(root, list("abcdef"))
            result, evaluate = self.analyze(root, sessions, selected, lambda *args, **kwargs: self.fail("Expired evaluation started"), time.time() - 1)
            evaluate.assert_not_called()
            self.assertEqual(result["workers"][0]["score"], 25)
            for row in result["workers"][1:]:
                self.assertIsNone(row["score"])
                self.assertIn("TimeoutError", row["error"])

    def test_invalid_verifier_output_is_not_a_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sessions, selected = self.fixture(root, list("abcdef"))
            result, _ = self.analyze(root, sessions, selected, lambda *args, **kwargs: {"reward": float("nan")})
            self.assertTrue(all(row["score"] is None for row in result["workers"][1:]))
            self.assertEqual(result["status"], "completed_with_failures")

    def test_duplicate_failed_design_is_not_evaluated_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sessions, selected = self.fixture(root, ["a", "b", "b", "b", "b", "b"])
            def evaluator(*args, **kwargs):
                raise RuntimeError("Synthetic evaluator failure")
            result, evaluate = self.analyze(root, sessions, selected, evaluator)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(result["unique_private_evaluations"], 1)
            self.assertTrue(all(row["score"] is None for row in result["workers"][1:]))
            self.assertTrue(all(row["error"] for row in result["workers"][1:]))

    def test_mismatched_frozen_selection_blocks_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sessions, selected = self.fixture(root, list("abcdef"))
            selected = dict(selected, sha256="invalid")
            with patch.object(A.Q, "evaluate") as evaluate:
                with self.assertRaises(ValueError):
                    A.analyze_worker_finals(root, sessions, selected, time.time() + 10, root / "lock")
                evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
