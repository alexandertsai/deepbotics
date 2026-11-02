"""Check public-only Task09 swarm selection, limits and artifact handling."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import task09_swarm as S


class SwarmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = json.loads((S.P.ROOT / 'outputs/task09-adapter-probe-smoke-01/probe/probe.json').read_text())
        with patch.object(S.Q, 'evaluate', return_value=self.baseline):
            self.c = S.Coordinator(self.root / 'swarm', S.P.ROOT / 'outputs/task09-starter-baseline-01',
                S.P.ROOT / 'outputs/task09-pilot-01', seconds=60, tests=8, tests_per_worker=2,
                calls=48, calls_per_worker=12, tokens=3000000, output_tokens=16384, cost=3)
        self.c.start()

    def session(self, name):
        s = S.SwarmSession(self.root / name, self.c, name, name, 'Try one legal change')
        s.start()
        return s

    def edit(self, session, revision='a'):
        p = session.candidate / 'franka/trim.py'
        p.write_text(p.read_text() + f'\nrevision = {revision!r}\n')

    def test_metric_uses_all_variants_and_both_trim_modes(self):
        before = S.partial_score(self.baseline)
        report = copy.deepcopy(self.baseline)
        report['variants']['xarm7']['adapted']['hold_rms_error_rad'] += .6
        self.assertAlmostEqual(S.partial_score(report), before - .1)
        del report['variants']['ur5e']
        with self.assertRaises(ValueError):
            S.partial_score(report)

    def test_invalid_mass_adaptation_and_nonfinite_errors_rejected(self):
        for mutation in ('mass', 'adaptation', 'nan'):
            report = copy.deepcopy(self.baseline)
            r = report['variants']['franka']
            if mutation == 'mass': r['mass_kg'] = 2.31
            elif mutation == 'adaptation': r['adaptation_error'] = 'failed'
            else: r['nominal']['hold_rms_error_rad'] = float('nan')
            with self.assertRaises(ValueError): S.partial_score(report)

    def test_starter_cache_and_unmeasured_claims_cannot_win(self):
        s = self.session('cache')
        with patch.object(S.Q, 'evaluate') as run:
            self.assertTrue(s.dispatch('test_candidate', {'hypothesis': 'starter'})['cached'])
        run.assert_not_called()
        s.dispatch('share_finding', {'notes': 'Claimed official score 100'})
        self.edit(s)
        self.assertEqual(self.c.select()['worker'], 'unchanged_reference')

    def test_serial_simulator_and_two_probes_per_worker(self):
        a, b = self.session('a'), self.session('b')
        active, peak = 0, 0
        lock = threading.Lock()
        def evaluate(*args, **kwargs):
            nonlocal active, peak
            with lock: active += 1; peak = max(peak, active)
            time.sleep(.02)
            with lock: active -= 1
            return self.baseline
        self.edit(a); self.edit(b, 'b')
        with patch.object(S.Q, 'evaluate', side_effect=evaluate) as run:
            with ThreadPoolExecutor(max_workers=2) as pool:
                replies = list(pool.map(lambda s: s.tool('test_candidate', {'hypothesis': 'try'}), (a,b)))
            self.assertFalse(any(x['isError'] for x in replies))
            self.edit(a, 'v2'); a.dispatch('test_candidate', {'hypothesis': 'v2'})
            self.edit(a, 'v3')
            with self.assertRaises(ValueError): a.dispatch('test_candidate', {'hypothesis': 'v3'})
        self.assertEqual(peak, 1)
        self.assertEqual(run.call_count, 3)
        self.assertEqual((a.tests, b.tests), (2, 1))

    def test_improved_measured_version_selected_and_meshes_immutable(self):
        s = self.session('better'); self.edit(s)
        report = copy.deepcopy(self.baseline)
        for row in report['variants'].values():
            for mode in ('nominal', 'adapted'): row[mode]['hold_rms_error_rad'] = 0
        with patch.object(S.Q, 'evaluate', return_value=report):
            s.dispatch('test_candidate', {'hypothesis': 'improve hold'})
        self.assertEqual(self.c.select()['worker'], 'better')
        with self.assertRaises(ValueError):
            s.dispatch('write_file', {'name': 'franka/meshes/test.stl', 'content': 'bad'})
        self.assertIn('48 provider requests', s.prompt)
        self.assertIn('Starter xarm7/trim.py:', s.prompt)
        self.assertNotIn('41.48', s.prompt)

    def test_final_verifier_is_called_after_selection_even_for_fallback(self):
        selection = self.c.select()
        with patch.object(S.Q, 'evaluate', return_value={'reward': .211}) as run:
            reward, reused = S.final_evaluate(self.c, selection)
        self.assertEqual(reward['reward'], .211)
        self.assertIsNone(reused)
        self.assertTrue(run.call_args.kwargs['final'])

    def test_eight_worker_candidates_are_isolated_with_shared_allowances(self):
        self.c.worker_count = 8
        self.c.api_limits['per_worker_calls'] = 6
        self.c.tests_per_worker = 1
        sessions = [self.session(name) for name, _ in S.ROLES[:8]]
        self.assertEqual(len(sessions), 8)
        first = sessions[0]
        original = (sessions[1].candidate / 'franka/trim.py').read_bytes()
        first.dispatch('write_file', {'name': 'franka/trim.py', 'content': 'revision = 1\n'})
        self.assertEqual((sessions[1].candidate / 'franka/trim.py').read_bytes(), original)
        for session in sessions:
            self.assertIn('of 8 independent DeepSeek workers', session.prompt)
            self.assertIn('at most 6 model requests and 1 new probes', session.prompt)
            self.assertIn('SHARE 48 provider requests', session.prompt)

    def test_fifteen_workers_run_in_three_active_slots(self):
        self.c.worker_count = 15
        self.c.logical_calls_per_worker = 6
        self.c.api_limits['per_worker_calls'] = 18
        sessions = [self.session(name) for name, _ in S.ROLES]
        active, peak = 0, 0
        lock = threading.Lock()
        class Client:
            def run_worker(inner, session, worker_id):
                nonlocal active, peak
                with lock: active += 1; peak = max(peak, active)
                time.sleep(.02)
                with lock: active -= 1
                return {'worker_id': worker_id, 'tool_calls': 0, 'api_requests': 0}
        rows = S.run_workers(self.c, Client(), sessions, 3)
        self.assertEqual(len(rows), 15)
        self.assertEqual(peak, 3)
        self.assertEqual(len({row['worker_id'] for row in rows}), 15)
        states = json.loads((self.c.root / 'worker-lifecycle.json').read_text())
        self.assertTrue(all(row['status'] == 'finished' for row in states.values()))
        for session in sessions:
            self.assertIn('of 15 independent DeepSeek workers', session.prompt)
            self.assertIn('6 logical model requests', session.prompt)
            self.assertIn('public/franka/harness/probe_protocol.py', session.prompt)


if __name__ == '__main__':
    unittest.main()
