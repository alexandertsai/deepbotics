from datetime import datetime
import fcntl
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import task07_adapter as Q
import task07_runner as R


def report(fraction=0.0):
    return {"episodes": [{"public_seed": 35, "episode_seconds": 10.0, "ticks": 200,
        "clear_frac": fraction, "floor_drops": 0, "damage_events": 0, "bin_hits": 0,
        "policy_faults": 0, "aborted": None}]}


class Task07Boundaries(unittest.TestCase):
    def test_deadlines_are_today_at_fourteen_and_fifteen_pacific(self):
        now = datetime(2026, 11, 2, 12, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        admission, stop = Q.local_deadlines(now)
        self.assertEqual(stop - admission, 3600)
        self.assertEqual(datetime.fromtimestamp(admission, now.tzinfo).hour, 14)

    def test_incomplete_and_faulted_episodes_cannot_win(self):
        self.assertEqual(Q.partial_score(report(.5)), .5)
        for field, value in (("ticks", 199), ("public_seed", 36), ("clear_frac", float("nan")),
                             ("policy_faults", 1), ("floor_drops", -1), ("episode_seconds", 9.0)):
            broken = report(.9)
            broken["episodes"][0][field] = value
            with self.assertRaises((ValueError, TypeError)):
                Q.partial_score(broken)

    def test_candidate_paths_cannot_escape_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "candidate"
            base.mkdir()
            (base / "escape").symlink_to(root, target_is_directory=True)
            for value in ("../secret.py", "/etc/passwd", "escape/test.py", "policy.so"):
                with self.assertRaises(ValueError):
                    R.safe_artifact(base, value)
            self.assertEqual(R.safe_artifact(base, "helpers/geometry.py"), base.resolve() / "helpers/geometry.py")

    def test_freeze_rejects_links_and_snapshots_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Q.make_starter(root / "starter")
            manifest = R.freeze(root / "starter", root / "frozen")
            self.assertTrue(manifest["sha256"])
            (root / "starter/link.py").symlink_to(root / "frozen/policy.py")
            with self.assertRaises(ValueError):
                R.freeze(root / "starter", root / "bad")

    def test_ties_regressions_and_untested_edits_retain_incumbent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Q.make_starter(root / "starter")
            coordinator = R.Coordinator(root / "run", root / "starter", report(.4), seconds=900,
                tests_per_worker=2, simulator=threading.Lock(), stop_at=time.time() + 100)
            coordinator.deadline = time.monotonic() + 100
            session = R.Session(root / "agent", coordinator, "test", "test", astra=True)
            session.start()
            session.dispatch("write_file", {"name": "policy.py", "content": Q.ZERO_POLICY + "\n# changed\n"})
            with patch.object(Q, "evaluate", return_value=report(.4)):
                coordinator.test(session, "tie")
            self.assertEqual(coordinator.best["worker"], "starter")
            session.dispatch("write_file", {"name": "policy.py", "content": Q.ZERO_POLICY + "\n# worse\n"})
            with patch.object(Q, "evaluate", return_value=report(.2)):
                coordinator.test(session, "regression")
            self.assertEqual(coordinator.best["worker"], "starter")
            session.dispatch("write_file", {"name": "policy.py", "content": Q.ZERO_POLICY + "\n# untested\n"})
            self.assertEqual(coordinator.select()["worker"], "starter")
            self.assertFalse(json.loads((coordinator.root / "selection.json").read_text())["selection_used_final_scores"])

    def test_shutdown_deadline_does_not_start_docker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Q.P, "checked") as checked:
            with self.assertRaises(TimeoutError):
                Q.evaluate(Path(directory), Path(directory) / "out", stop_at=time.time() - 1)
            checked.assert_not_called()

    def test_public_docker_has_fixed_caps_and_no_private_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "probe"
            seen = []
            def checked(command, **kwargs):
                seen.extend(command)
                (output / "probe.json").write_text(json.dumps(report()))
                return SimpleNamespace(stdout="", stderr="")
            with patch.object(Q.P, "checked", side_effect=checked):
                Q.evaluate(root, output)
            self.assertIn(Q.AGENT_IMAGE, seen)
            self.assertNotIn(Q.IMAGE, seen)
            self.assertEqual(seen[seen.index("--network") + 1], "none")
            self.assertEqual(seen[seen.index("--cpus") + 1], "4")
            self.assertEqual(seen[seen.index("--memory") + 1], "4g")
            self.assertIn("--read-only", seen)
            self.assertTrue(any(value.endswith(":/submission:ro") for value in seen))
            self.assertFalse(any("/tests" in value for value in seen))

    def test_admission_cutoff_prevents_provider_worker_launch(self):
        client = SimpleNamespace(run_worker=lambda *args: self.fail("Provider launcher called after cutoff"))
        with self.assertRaises(TimeoutError):
            R.launch_deepseek(client, SimpleNamespace(worker_id="test"), time.time() - 1)

    def test_contended_simulator_lock_stops_at_fixed_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulation.lock"
            with path.open("a") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                started = time.monotonic()
                deadline = started + .03
                with self.assertRaises(TimeoutError):
                    with R.SimulationLock(path, lambda: deadline):
                        self.fail("Contended lock was acquired")
                self.assertLess(time.monotonic() - started, .5)

    def test_memory_gated_conditions_keep_separate_wait_evidence(self):
        from short_pilot_runtime import GIB, MemoryGatedSimulationLock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deadline = time.monotonic() + 1
            observations = iter((8 * GIB, 2 * GIB))
            deep = MemoryGatedSimulationLock(root / "shared.lock", lambda: deadline,
                root / "deep.jsonl", memory_reader=lambda: next(observations), poll_seconds=.001)
            astra = MemoryGatedSimulationLock(root / "shared.lock", lambda: deadline,
                root / "astra.jsonl", memory_reader=lambda: 2 * GIB, poll_seconds=.001)
            with deep:
                pass
            with astra:
                pass
            for name in ("deep", "astra"):
                rows = [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
                self.assertEqual([row["event"] for row in rows], ["acquired", "released"])
                self.assertEqual(rows[0]["deadline_monotonic"], deadline)
                self.assertEqual(rows[0]["memory_limit_gib"], 6.5)
            deep_rows = [json.loads(line) for line in (root / "deep.jsonl").read_text().splitlines()]
            self.assertGreater(deep_rows[0]["memory_wait_seconds"], 0)
            self.assertEqual(deep_rows[0]["stats_samples"], 2)

    def test_memory_wait_does_not_move_the_development_deadline(self):
        from short_pilot_runtime import GIB, MemoryGatedSimulationLock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = time.monotonic()
            deadline = started + .02
            lock = MemoryGatedSimulationLock(root / "shared.lock", lambda: deadline,
                root / "wait.jsonl", memory_reader=lambda: 8 * GIB, poll_seconds=.001)
            with self.assertRaises(TimeoutError):
                with lock:
                    self.fail("High-memory simulation was admitted")
            self.assertLess(time.monotonic() - started, .5)
            row = json.loads((root / "wait.jsonl").read_text().splitlines()[-1])
            self.assertEqual(row["event"], "acquisition_failed")
            self.assertEqual(row["deadline_monotonic"], deadline)


if __name__ == "__main__":
    unittest.main()
