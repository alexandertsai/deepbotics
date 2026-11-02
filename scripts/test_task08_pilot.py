"""Check pilot boundaries without model calls or Docker."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task08_pilot as P


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = P.Session(self.root / "run", "astra", 60, 1)
        self.session.start()

    def test_public_reads_cannot_reach_verifier_or_credentials(self):
        for path in ("public/../../harness/config.py", "/etc/passwd", "candidate/../secret", "public\\secret"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.session.dispatch("read_file", {"path": path})
        link = self.session.candidate / "secret.py"
        link.symlink_to(P.ROOT / "scripts/mac_evaluation.py")
        with self.assertRaises(ValueError):
            self.session.dispatch("read_file", {"path": "candidate/secret.py"})

    def test_freezing_preserves_bytes_and_ends_writes(self):
        original = (self.session.candidate / "robot.xml").read_bytes()
        self.session.dispatch("submit", {})
        self.assertEqual((self.session.root / "frozen/robot.xml").read_bytes(), original)
        with self.assertRaises(ValueError):
            self.session.dispatch("write_file", {"name": "robot.xml", "content": "changed"})

    def test_failed_probes_still_consume_budget(self):
        with patch.object(P, "docker_evaluate", side_effect=TimeoutError("test timeout")) as evaluate:
            reply = self.session.tool("test_candidate", {"hypothesis": "Check the reference"})
            self.assertTrue(reply["isError"])
            self.assertEqual(self.session.tests, 1)
            again = self.session.tool("test_candidate", {"hypothesis": "Retry"})
            self.assertTrue(again["isError"])
            self.assertEqual(evaluate.call_count, 1)

    def test_expired_budget_prevents_mutation(self):
        self.session.deadline = time.monotonic() - 1
        with self.assertRaises(ValueError):
            self.session.dispatch("edit_file", {"name": "robot.xml", "old": "x", "new": "y"})

    def test_artifact_edit_requires_unique_match(self):
        file = self.session.candidate / "controller.py"
        file.write_text("repeat repeat")
        with self.assertRaises(ValueError):
            self.session.dispatch("edit_file", {"name": "controller.py", "old": "repeat", "new": "x"})
        self.assertEqual(file.read_text(), "repeat repeat")

    def test_no_verifier_path_in_agent_listing(self):
        listing = json.dumps(self.session.dispatch("list_files", {}))
        self.assertNotIn("harness/config.py", listing)
        self.assertNotIn("score_task.py", listing)
        self.assertNotIn("golden", listing)

    def test_preflight_failure_never_launches_agent(self):
        with patch.object(P, "preflight", return_value={"docker_running": False}), \
             patch.object(P, "launch") as launch, \
             patch.object(sys, "argv", ["task08_pilot.py", "run"]):
            with self.assertRaises(SystemExit):
                P.main()
            launch.assert_not_called()

    def test_rax_notes_persist_and_templates_are_rendered(self):
        session = P.Session(self.root / "rax", "deepseek-rax", 60, 1)
        session.start()
        self.assertNotIn("{%", session.prompt)
        session.dispatch("memory_write", {"notes": "Probe 0 tipped; lower the battery."})
        self.assertEqual(session.dispatch("memory_read", {})["notes"],
                         "Probe 0 tipped; lower the battery.")


if __name__ == "__main__":
    unittest.main()
