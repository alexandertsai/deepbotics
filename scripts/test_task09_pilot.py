"""Budget, artifact and public-data boundaries for the Task 09 adapter."""

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task09_pilot as Q


class Task09Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.public = self.root / "public"
        for variant in Q.VARIANTS:
            directory = self.public / variant
            (directory / "meshes").mkdir(parents=True)
            (directory / "lead.xml").write_text('<mujoco model="bare"/>')
            (directory / "meshes/part.stl").write_bytes(b"mesh")
        self.public_patch = patch.object(Q, "PUBLIC", self.public)
        self.public_patch.start()
        self.addCleanup(self.public_patch.stop)
        Q.make_starter(self.root / "starter")
        self.session = Q.Session(self.root / "run", "astra", 60, 1, self.root / "starter")
        self.session.start()

    def test_entry_edits_cannot_change_meshes_or_public_helpers(self):
        for name in ("franka/meshes/part.stl", "franka/harness/spec.py", "../secret", "trim.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.session.dispatch("write_file", {"name": name, "content": "changed"})
        self.assertEqual((self.session.candidate / "franka/meshes/part.stl").read_bytes(), b"mesh")

    def test_private_source_reads_are_rejected(self):
        for path in ("public/../tests/score_task.py", "candidate/../../secret", "/etc/passwd"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.session.dispatch("read_file", {"path": path})
        (self.session.candidate / "secret.py").symlink_to(Q.ROOT / "scripts/mac_evaluation.py")
        with self.assertRaises(ValueError):
            self.session.dispatch("read_file", {"path": "candidate/secret.py"})

    def test_freeze_includes_all_variants_and_closes_development(self):
        self.session.dispatch("submit", {})
        manifest = json.loads((self.session.root / "frozen-manifest.json").read_text())
        self.assertEqual(len(manifest["files"]), 9)
        self.assertIn("xarm7/meshes/part.stl", manifest["files"])
        with self.assertRaises(ValueError):
            self.session.dispatch("write_file", {"name": "franka/trim.py", "content": "changed"})

    def test_failed_probe_consumes_one_allowance(self):
        with patch.object(Q, "evaluate", side_effect=TimeoutError("failed")) as evaluate:
            self.assertTrue(self.session.tool("test_candidate", {"hypothesis": "test"})["isError"])
            self.assertTrue(self.session.tool("test_candidate", {"hypothesis": "retry"})["isError"])
            self.assertEqual(evaluate.call_count, 1)

    def test_expired_budget_blocks_edits(self):
        self.session.deadline = time.monotonic() - 1
        with self.assertRaises(ValueError):
            self.session.dispatch("write_file", {"name": "franka/trim.py", "content": "changed"})

    def test_public_starter_has_complete_zero_trim_contract(self):
        scope = {}
        exec(Q.ZERO_TRIM, scope)
        model = SimpleNamespace(nv=7)
        q = np.zeros(7)
        self.assertEqual(scope["plan_probe"](model, (q, q), q-1, q+1).shape, (15, 7))
        params = scope["adapt"](model, [(q, q)])
        np.testing.assert_array_equal(scope["make_trim"](model)(q), q)
        np.testing.assert_array_equal(scope["make_trim_adapted"](model, params)(q), q)

    def test_rax_prompt_and_notes_are_supported(self):
        session = Q.Session(self.root / "rax", "deepseek-rax", 60, 1, self.root / "starter")
        session.start()
        self.assertNotIn("{%", session.prompt)
        session.dispatch("memory_write", {"notes": "Only the nominal probe has been measured."})
        self.assertIn("nominal", session.dispatch("memory_read", {})["notes"])

    def test_failed_preflight_starts_no_model(self):
        with patch.object(Q, "preflight", return_value={"docker_running": False}), \
             patch.object(Q, "launch") as launch, \
             patch.object(sys, "argv", ["task09_pilot.py", "run"]):
            with self.assertRaises(SystemExit):
                Q.main()
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
