import fcntl
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import swarm_operator_analysis as A
from services.controller.storage import snapshot


class MockMemoryLock:
    def __init__(self, path, deadline, log_path):
        self.deadline = deadline

    def __enter__(self):
        if self.deadline() <= time.monotonic():
            raise TimeoutError("Expired")
        return self

    def __exit__(self, *args):
        pass


class OperatorSweepTests(unittest.TestCase):
    def fixture(self, root, codes, task=9):
        source = root / f"task{task:02d}-completed"
        source.mkdir()
        names = list(A.TASK07_WORKERS) if task == 7 else [f"worker{i}" for i in range(6)]
        protocol = {"workers": names, "simulation_lock": str(root / "source-task.lock")}
        (source / "protocol.json").write_text(json.dumps(protocol))
        (source / "summary.json").write_text('{"status":"completed"}')
        condition = source / "deepseek" if task == 7 else source
        condition.mkdir(exist_ok=True)
        original = root / "candidate"
        original.mkdir()
        manifests = []
        for name, code in zip(names, codes):
            (original / "entry.py").write_text(code)
            worker = condition / name
            worker.mkdir()
            manifest = snapshot(original, worker / "frozen", max_bytes=1024, workspace_path=f"code/task{task:02d}")
            (worker / "frozen-manifest.json").write_text(json.dumps(manifest))
            manifests.append(manifest)
        selected = snapshot(condition / names[0] / "frozen", condition / "selected", max_bytes=1024,
                            workspace_path=f"code/task{task:02d}")
        (condition / "selection.json").write_text(json.dumps({"worker": names[0], "manifest": selected}))
        (condition / "final").mkdir()
        (condition / "final/reward.json").write_text('{"reward":0.25}')
        return source, condition, names

    def sweep(self, root, source, task=9, seconds=2):
        return A.Sweep(root / "operator", stop_at=time.time() + seconds, runs=[(task, source)],
                       operator_lock=root / "global.lock", priority_pids=(), poll_seconds=.001)

    def test_selected_final_reuse_and_distinct_frozen_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, condition, names = self.fixture(root, ["a", "a", "b", "b", "c", "c"])
            source_bytes = (condition / "selection.json").read_bytes()
            sweep = self.sweep(root, source)
            def evaluate(task, bundle, output, remaining):
                self.assertTrue(bundle.is_relative_to(root / "operator"))
                self.assertTrue(output.is_relative_to(root / "operator"))
                self.assertGreater(remaining, 0)
                return {"reward": .5}
            with patch.object(A, "run_is_eligible", return_value=(True, None)), patch.object(A, "runtime_blockers", return_value=[]), patch.object(A, "MemoryGatedSimulationLock", MockMemoryLock), patch.object(A, "evaluate", side_effect=evaluate) as evaluate:
                report = sweep.run()
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual([row["score"] for row in report["runs"][0]["workers"]], [25, 25, 50, 50, 50, 50])
            self.assertEqual(report["model_calls"], 0)
            self.assertFalse(report["feedback_available_to_workers"])
            self.assertEqual((condition / "selection.json").read_bytes(), source_bytes)
            comparison = report["runs"][0]["selected_vs_best_available"]
            self.assertEqual(comparison["gap_best_available_minus_selected"], 25)
            self.assertFalse(comparison["partial"])

    def test_live_source_is_never_copied_or_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, _, _ = self.fixture(root, list("abcdef"))
            sweep = self.sweep(root, source, seconds=.01)
            with patch.object(A, "run_is_eligible", return_value=(False, "live parent")), patch.object(A, "evaluate") as evaluate:
                result = sweep.run()
            evaluate.assert_not_called()
            self.assertFalse((root / "operator" / source.name).exists())
            self.assertTrue(result["runs"][0]["partial"])
            self.assertTrue(all(row["score"] is None and row["error"] for row in result["runs"][0]["workers"]))

    def test_missing_frozen_bundle_is_explicitly_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, condition, names = self.fixture(root, ["a"] * 6, task=7)
            (condition / names[-1] / "frozen-manifest.json").unlink()
            sweep = self.sweep(root, source, task=7)
            with patch.object(A, "run_is_eligible", return_value=(True, None)), patch.object(A, "runtime_blockers", return_value=[]), patch.object(A, "evaluate") as evaluate:
                result = sweep.run()
            evaluate.assert_not_called()
            missing = result["runs"][0]["workers"][-1]
            self.assertIsNone(missing["score"])
            self.assertIn("FileNotFoundError", missing["error"])
            self.assertTrue(result["runs"][0]["selected_vs_best_available"]["partial"])

    def test_busy_source_task_lock_defers_without_holding_global_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, _, _ = self.fixture(root, ["a", "b", "b", "b", "b", "b"])
            sweep = self.sweep(root, source)
            sweep.prepare(sweep.runs[0])
            row = sweep.runs[0]["workers"][1]
            with (root / "source-task.lock").open("a") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(A, "runtime_blockers", return_value=[]), patch.object(A, "MemoryGatedSimulationLock", MockMemoryLock), patch.object(A, "evaluate") as evaluate:
                    self.assertFalse(sweep.score_row(sweep.runs[0], row))
                evaluate.assert_not_called()
            self.assertEqual(row["status"], "pending")

    def test_admission_allows_other_tasks_but_not_same_task_or_third_container(self):
        records = [{"pid": 123, "output": Path("/outputs/task09-planner")},
                   {"pid": 456, "output": Path("/outputs/task07-short") }]
        containers = '\n'.join(json.dumps({"Names": name}) for name in ("task09-sim", "task07-sim"))
        with patch.object(A, "registered_runs", return_value=records), patch.object(A, "pid_alive", return_value=True), patch.object(A.subprocess, "run", return_value=SimpleNamespace(stdout=containers)):
            self.assertEqual(A.runtime_blockers(8, ()), [])
            self.assertTrue(A.runtime_blockers(9, ()))
        containers += '\n' + json.dumps({"Names": "other-sim"})
        with patch.object(A, "registered_runs", return_value=[]), patch.object(A.subprocess, "run", return_value=SimpleNamespace(stdout=containers)):
            self.assertTrue(any("container_admission_limit" in reason for reason in A.runtime_blockers(8, ())))

    def test_failed_duplicate_has_no_score_and_runs_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, _, _ = self.fixture(root, ["a", "b", "b", "b", "b", "b"])
            sweep = self.sweep(root, source)
            with patch.object(A, "run_is_eligible", return_value=(True, None)), patch.object(A, "runtime_blockers", return_value=[]), patch.object(A, "MemoryGatedSimulationLock", MockMemoryLock), patch.object(A, "evaluate", side_effect=RuntimeError("synthetic failure")) as evaluate:
                result = sweep.run()
            self.assertEqual(evaluate.call_count, 1)
            self.assertTrue(all(row["score"] is None and row["error"] for row in result["runs"][0]["workers"][1:]))

    def test_per_candidate_final_timeout_is_bounded(self):
        with patch.object(A.P, "docker_evaluate", return_value={}) as task08, patch.object(A.Q, "evaluate", return_value={}) as task09, patch.object(A.T, "evaluate", return_value={}) as task07:
            for task in (7, 8, 9):
                A.evaluate(task, Path("bundle"), Path("output"), 10000)
            self.assertEqual(task08.call_args.kwargs["timeout"], 900)
            self.assertEqual(task09.call_args.kwargs["timeout"], 900)
            self.assertEqual(task07.call_args.kwargs["timeout"], 3600)
            A.evaluate(7, Path("bundle"), Path("output"), 12)
            self.assertEqual(task07.call_args.kwargs["timeout"], 12)


if __name__ == "__main__":
    unittest.main()
