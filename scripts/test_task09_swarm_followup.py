"""Check iterative budgets and public multi-metric follow-up selection."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import task09_swarm_followup as S


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = json.loads((S.P.ROOT / 'outputs/task09-followup-probe-smoke-01/probe/probe.json').read_text())
        with patch.object(S.Q, 'evaluate', return_value=self.baseline):
            self.c = S.Coordinator(self.root / 'swarm', S.P.ROOT / 'outputs/task09-starter-baseline-01',
                S.P.ROOT / 'outputs/task09-pilot-01', seconds=60, tests=45, tests_per_worker=3,
                calls=540, calls_per_worker=36, tokens=18000000, output_tokens=8192, cost=5, worker_count=15)
        self.c.logical_calls_per_worker = 12
        self.c.start()

    def session(self, name):
        session = S.SwarmSession(self.root / name, self.c, name, name, 'Improve public passive balance')
        session.start()
        return session

    def report(self, hold, residual, headroom):
        report = copy.deepcopy(self.baseline)
        for row in report['variants'].values():
            row['passive_residual_max_nm'] = residual
            for mode in ('nominal', 'adapted'):
                row[mode]['hold_rms_error_rad'] = hold
                row[mode]['servo_headroom_mean'] = headroom
        return report

    def test_zero_hold_error_does_not_override_passive_balance_and_headroom(self):
        bad = self.report(0, .47, .30)
        good = self.report(.001, .15, .70)
        self.assertGreater(S.hold_score(bad), S.hold_score(good))
        self.assertGreater(S.partial_score(good), S.partial_score(bad))

    def test_missing_headroom_and_invalid_physical_measurements_are_rejected(self):
        for change in ('missing', 'invalid_room', 'invalid_residual'):
            report = copy.deepcopy(self.baseline)
            row = report['variants']['franka']
            if change == 'missing': del row['nominal']['servo_headroom_mean']
            elif change == 'invalid_room': row['adapted']['servo_headroom_mean'] = 1.1
            else: row['passive_residual_max_nm'] = float('nan')
            with self.assertRaises((ValueError, KeyError)):
                S.partial_score(report)

    def test_headroom_changes_selection_when_hold_and_residual_are_equal(self):
        low = self.report(.001, .15, .20)
        high = self.report(.001, .15, .70)
        self.assertGreater(S.partial_score(high), S.partial_score(low))

    def test_three_probes_per_worker_and_twelve_logical_turns(self):
        session = self.session('iterate')
        file = session.candidate / 'franka/trim.py'
        with patch.object(S.Q, 'evaluate', return_value=self.baseline) as run:
            for i in range(4):
                file.write_text(file.read_text() + f'\nrevision = {i}\n')
                reply = session.tool('test_candidate', {'hypothesis': f'iteration {i}'})
                self.assertEqual(reply['isError'], i == 3)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(self.c.tests, 3)
        self.assertIn('12 logical model requests', session.prompt)
        self.assertIn('36 provider attempts including retries', session.prompt)
        self.assertIn('Plan three measured iterations', session.prompt)

    def test_worker_exit_rehearses_untested_frozen_proposal(self):
        session = self.session('exit')
        candidate = self.report(.001, .15, .70)
        class Client:
            def run_worker(inner, worker_session, worker_id):
                file = worker_session.candidate / 'franka/trim.py'
                file.write_text(file.read_text() + '\nrevision = 1\n')
                worker_session.dispatch('submit', {})
                return {'worker_id': worker_id, 'tool_calls': 1, 'api_requests': 1}
        with patch.object(S.Q, 'evaluate', return_value=candidate):
            S.run_workers(self.c, Client(), [session], 1)
        self.assertEqual(session.tests, 1)
        selection = self.c.select()
        self.assertEqual(selection['worker'], 'exit')
        self.assertFalse(selection['selection_used_final_scores'])

    def test_spring_roles_start_first_and_private_final_is_unchanged(self):
        self.assertEqual([name for name, _ in S.ROLES[:3]], ['spring_tuning', 'proximal_springs', 'passive_balance'])
        selection = self.c.select()
        with patch.object(S.Q, 'evaluate', return_value={'reward': .211}) as evaluate:
            S.final_evaluate(self.c, selection)
        self.assertTrue(evaluate.call_args.kwargs['final'])

    def test_shared_tool_status_is_bounded_but_full_public_logs_remain(self):
        key = self.c.manifest['sha256']
        for index in range(45):
            self.c.entries.append({'index': index, 'worker': 'peer', 'status': 'measured',
                'partial_score': self.c.baseline_score, 'sha256': key, 'report': {'large': 'x' * 50000}})
            self.c.findings.append({'worker': 'peer', 'notes': 'x' * 4000,
                                   'source': 'agent_statement_not_verified_measurement'})
        self.c.persist()
        status = self.c.status()
        self.assertEqual((len(status['probes']), len(status['findings'])), (12, 8))
        self.assertEqual(status['probes_total'], 45)
        self.assertLess(len(json.dumps(status)), 20000)
        disk = json.loads((self.c.root / 'probe-ledger.json').read_text())
        self.assertEqual(len(disk[0]['report']['large']), 50000)
        detail = self.c.read_probe(0)
        self.assertIn('tested_hold_poses', detail['report']['variants']['franka'])
        with self.assertRaises(ValueError): self.c.read_probe(1000)

    def test_adaptive_launch_flag_selects_controller_with_unchanged_cost_guard(self):
        args = ['task09_swarm_followup.py', '--output', str(self.root / 'launch'), '--paced', '--adaptive-pacing']
        summaries = [{'worker_id': name, 'api_requests': 0, 'tool_calls': 0} for name, _ in S.ROLES]
        with patch.object(S.sys, 'argv', args), patch.object(S.Q, 'preflight', return_value={'ready': True}), \
             patch.object(S.Q, 'evaluate', return_value=self.baseline), \
             patch.object(S, 'run_workers', return_value=summaries), \
             patch.object(S, 'final_evaluate', return_value=({'reward': .211}, None)), \
             patch('mac_evaluation.load_keys', return_value={'FIREWORKS_API_KEY': 'test-key'}), \
             patch('task09_adaptive_client.AdaptivePacedClient') as controller, redirect_stdout(io.StringIO()):
            controller.return_value.summary.return_value = {'requests': 0, 'estimated_cost_usd': 0}
            S.main()
        options = controller.call_args.kwargs
        self.assertEqual((options['max_cost'], options['max_concurrent'], options['max_workers']), (5, 3, 15))
        self.assertEqual(options['logical_requests_per_worker'], 12)
        self.assertEqual((options['ramp_interval_seconds'], options['reported_limit_fraction']), (120, .8))
        protocol = json.loads((self.root / 'launch/protocol.json').read_text())
        self.assertTrue(protocol['adaptive_pacing']['enabled'])


if __name__ == '__main__': unittest.main()
