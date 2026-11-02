"""Check follow-up headroom measurement and unchanged final-scoring routing."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import task09_followup as F
import task09_followup_probe as W


class FollowupTests(unittest.TestCase):
    def test_headroom_distinguishes_unused_half_and_saturated_capacity(self):
        self.assertEqual(W.mean_servo_headroom(np.zeros(3)), 1.0)
        self.assertAlmostEqual(W.mean_servo_headroom([0.175, -0.175]), 0.5)
        self.assertEqual(W.mean_servo_headroom([0.35, -0.70]), 0.0)
        self.assertAlmostEqual(W.mean_servo_headroom([0.0, 0.175, -0.70]), 0.5)

    def test_hold_measures_clipped_applied_torque_for_each_joint_and_step(self):
        model = SimpleNamespace(opt=SimpleNamespace(timestep=0.1))
        data = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3), qfrc_applied=np.zeros(3))
        applied = []
        def step(_, state):
            applied.append(state.qfrc_applied.copy())
        with patch.object(W.mujoco, "MjData", return_value=data), \
             patch.object(W.mujoco, "mj_forward"), patch.object(W.mujoco, "mj_step", side_effect=step):
            measured, saturation, headroom = W.hold(model, np.zeros(3), np.array([0., .175, .70]), seconds=.3)
        self.assertEqual(len(applied), 3)
        np.testing.assert_array_equal(measured, np.zeros(3))
        np.testing.assert_array_equal(applied[0], np.array([0., .175, .35]))
        self.assertEqual(saturation, 1.0)
        self.assertAlmostEqual(headroom, .5)

    def test_final_uses_the_original_evaluator_without_public_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(F.Q, "evaluate", return_value={"reward": .5}) as original:
                result = F.evaluate(root / "bundle", root / "final", final=True, timeout=35)
            self.assertEqual(result["reward"], .5)
            original.assert_called_once_with(root / "bundle", root / "final", final=True, timeout=35)
            self.assertFalse((root / "final").exists())

    def test_development_mounts_new_public_worker_without_changing_live_worker(self):
        old_path = F.Q.WORKER
        old_bytes = old_path.read_bytes()
        report = {"probe_revision": "servo_headroom_v1", "variants": {}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            output = root / "probe"
            def checked(command, **kwargs):
                self.assertIn(f"{F.WORKER}:/pilot/probe.py:ro", command)
                self.assertNotIn(f"{old_path}:/pilot/probe.py:ro", command)
                self.assertIn(F.AGENT_IMAGE, command)
                self.assertNotIn(F.IMAGE, command)
                self.assertIn("none", command)
                self.assertIn("--read-only", command)
                (output / "probe.json").write_text(json.dumps(report))
                return SimpleNamespace(stdout="", stderr="")
            with patch.object(F.P, "checked", side_effect=checked), patch.object(F.Q, "evaluate") as original:
                self.assertEqual(F.evaluate(bundle, output), report)
                original.assert_not_called()
        self.assertEqual(F.Q.WORKER, old_path)
        self.assertEqual(old_path.read_bytes(), old_bytes)

    def test_session_routes_budgeted_probe_through_followup_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            starter = root / "starter"
            for variant in F.VARIANTS:
                directory = starter / variant
                directory.mkdir(parents=True)
                (directory / "lead.xml").write_text('<mujoco model="bare"/>')
                (directory / "trim.py").write_text(F.ZERO_TRIM)
            session = F.Session(root / "session", "astra", 60, 1, starter)
            session.start()
            with patch.object(F, "evaluate", return_value={"variants": {}}) as evaluate, \
                 patch.object(F.Q, "evaluate") as original:
                result = session.dispatch("test_candidate", {"hypothesis": "measure headroom"})
                self.assertEqual(result["version"], 0)
                self.assertEqual(evaluate.call_count, 1)
                original.assert_not_called()
                with self.assertRaises(ValueError):
                    session.dispatch("test_candidate", {"hypothesis": "exhausted"})
            self.assertIn("servo_headroom_mean", session.prompt)
            self.assertEqual(session.tests, 1)


if __name__ == "__main__":
    unittest.main()
