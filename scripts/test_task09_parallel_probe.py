import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import task09_parallel_probe as A


class Task09ParallelProbe(unittest.TestCase):
    def test_public_metrics_unchanged_and_resources_are_one_cpu_one_gib(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "probe"
            report = {"variants": {"franka": {"nominal": {"servo_headroom_mean": .5}}}}
            seen = []
            def checked(command, **kwargs):
                seen.extend(command)
                (output / "probe.json").write_text(json.dumps(report))
                return SimpleNamespace(stdout="", stderr="")
            with patch.object(A.P, "checked", side_effect=checked), patch.object(A, "hard_stop_timestamp", return_value=time.time()+100):
                self.assertEqual(A.evaluate(root, output, timeout=20), report)
            for flag, value in (("--cpus", "1"), ("--memory", "1g"), ("--memory-swap", "1g"), ("--network", "none")):
                self.assertEqual(seen[seen.index(flag) + 1], value)
            self.assertIn("--read-only", seen)
            self.assertIn(A.AGENT_IMAGE, seen)
            self.assertNotIn(A.IMAGE, seen)
            self.assertIn(str(A.WORKER)+":/pilot/probe.py:ro", seen)
            self.assertTrue(any(value.endswith(":/submission:ro") for value in seen))
            self.assertFalse(any("/tests" in value for value in seen))

    def test_final_delegates_to_unchanged_scorer_with_capped_deadline(self):
        with patch.object(A, "_FINAL_EVALUATE", return_value={"reward": .5}) as final, patch.object(A.P, "checked") as checked, patch.object(A, "hard_stop_timestamp", return_value=1010), patch.object(A.time, "time", return_value=1000):
            self.assertEqual(A.evaluate(Path("bundle"), Path("output"), final=True, timeout=500), {"reward": .5})
            self.assertEqual(final.call_args.kwargs, {"final": True, "timeout": 10})
            checked.assert_not_called()

    def test_expired_or_invalid_timeout_never_starts_container(self):
        with patch.object(A.P, "checked") as checked, patch.object(A, "hard_stop_timestamp", return_value=time.time()-1):
            with self.assertRaises(TimeoutError):
                A.evaluate(Path("bundle"), Path("output"))
            for value in (0, -1, float("nan"), True):
                with self.assertRaises(ValueError):
                    A.evaluate(Path("bundle"), Path("output"), timeout=value)
            checked.assert_not_called()

    def test_timeout_stops_only_the_owned_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(A.P, "checked", side_effect=subprocess.TimeoutExpired("docker", 1)) as checked, patch.object(A.subprocess, "run") as cleanup, patch.object(A, "hard_stop_timestamp", return_value=time.time()+100):
                with self.assertRaises(TimeoutError):
                    A.evaluate(root, root / "probe", timeout=1)
            command = checked.call_args.args[0]
            owned_name = command[command.index("--name") + 1]
            self.assertEqual(cleanup.call_args.args[0], ["docker", "rm", "-f", owned_name])

    def test_container_inventory_only_matches_task09(self):
        rows = [{"ID": "old09", "Names": "task09-pilot-abc", "Image": "x"},
                {"ID": "new09", "Names": "other", "Image": "rlebench-task09-agent:dev"},
                {"ID": "old08", "Names": "task08-pilot-abc", "Image": "x"},
                {"ID": "not09", "Names": "task090-else", "Image": "x"}]
        with patch.object(A.subprocess, "run", return_value=SimpleNamespace(stdout="\n".join(json.dumps(row) for row in rows))):
            self.assertEqual(A.Task09_container_ids(), {"old09", "new09"})


if __name__ == "__main__":
    unittest.main()
