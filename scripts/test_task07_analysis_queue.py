import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import task07_analysis_queue as Q
import test_swarm_operator_analysis as fixtures
from services.controller.storage import snapshot


class Task07DeferredQueue(unittest.TestCase):
    def fixture(self, root):
        source, condition, names = fixtures.OperatorSweepTests().fixture(root, ["b", "b", "b", "b", "b", "a"], task=7)
        selection = json.loads((condition / "selection.json").read_text())
        selection["worker"] = names[-1]
        selection["manifest"] = json.loads((condition / names[-1] / "frozen-manifest.json").read_text())
        (condition / "selection.json").write_text(json.dumps(selection))
        shutil.rmtree(condition / "selected")
        snapshot(condition / names[-1] / "frozen", condition / "selected", max_bytes=1024, workspace_path="code/task07")
        return source, names

    def test_known_selected_match_is_scored_before_first_slow_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, names = self.fixture(root)
            queue = Q.Task07AnalysisQueue(root / "operator", stop_at=time.time()+1,
                source=source, poll_seconds=.001)
            def slow_evaluate(*args, **kwargs):
                report = json.loads((root / "operator/report.json").read_text())
                known = report["runs"][0]["workers"][-1]
                self.assertEqual(known["worker"], names[-1])
                self.assertEqual(known["status"], "scored")
                self.assertEqual(known["score"], 25)
                queue.deadline = time.monotonic() - 1
                raise TimeoutError("Synthetic deadline")
            with patch.object(Q.A, "run_is_eligible", return_value=(True, None)), patch.object(Q.A, "pid_alive", return_value=False), patch.object(Q.A, "runtime_blockers", return_value=[]), patch.object(Q.A, "MemoryGatedSimulationLock", fixtures.MockMemoryLock), patch.object(Q.A, "evaluate", side_effect=slow_evaluate) as evaluate:
                result = queue.run()
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(result["runs"][0]["scored_workers"], 1)
            self.assertEqual(result["runs"][0]["best_available"]["score"], 25)
            self.assertTrue(result["runs"][0]["partial"])
            self.assertTrue(all(row["score"] is None and row["error"] for row in result["runs"][0]["workers"][:-1]))

    def test_live_primary_blocks_even_when_source_summary_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, _ = self.fixture(root)
            queue = Q.Task07AnalysisQueue(root / "operator", stop_at=time.time()+.02,
                source=source, poll_seconds=.001)
            with patch.object(Q.A, "run_is_eligible", return_value=(True, None)), patch.object(Q.A, "pid_alive", return_value=True), patch.object(Q.A, "evaluate") as evaluate:
                result = queue.run()
            evaluate.assert_not_called()
            self.assertEqual(result["runs"][0]["scored_workers"], 0)
            self.assertFalse((root / "operator" / source.name).exists())
            self.assertTrue(all(row["error"] for row in result["runs"][0]["workers"]))

    def test_summary_and_stopped_parent_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, _ = self.fixture(root)
            queue = Q.Task07AnalysisQueue(root / "operator", stop_at=time.time()+.02,
                source=source, poll_seconds=.001)
            with patch.object(Q.A, "run_is_eligible", return_value=(False, "Source summary is not complete")), patch.object(Q.A, "pid_alive", return_value=False), patch.object(Q.A, "evaluate") as evaluate:
                result = queue.run()
            evaluate.assert_not_called()
            self.assertEqual(result["status"], "partial_best_available")
            self.assertEqual(result["runs"][0]["scored_workers"], 0)


if __name__ == "__main__":
    unittest.main()
