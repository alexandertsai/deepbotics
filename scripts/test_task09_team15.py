import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import task09_team15 as T


class Team15Tests(unittest.TestCase):
    def report(self):
        row = {'mass_kg':1.,'mass_budget_kg':2.3,'passive_residual_max_nm':.2,
               'passive_drift_rad':.1,'spring_motion_rms_nm':.1}
        for mode in ('nominal','adapted','transfer_nominal','transfer_adapted'):
            row[mode] = {'hold_rms_error_rad':.01,'servo_headroom_mean':.5}
        return {'variants':{v:copy.deepcopy(row) for v in T.Q.VARIANTS}}

    def test_transfer_regression_is_penalized_even_with_perfect_nominal_hold(self):
        good = self.report()
        bad = copy.deepcopy(good)
        for row in bad['variants'].values():
            row['nominal']['hold_rms_error_rad'] = 0
            row['transfer_nominal']['hold_rms_error_rad'] = .8
            row['transfer_adapted']['hold_rms_error_rad'] = .8
        self.assertGreater(T.score(good),T.score(bad))
        del bad['variants']['franka']['transfer_adapted']
        with self.assertRaises(ValueError):T.score(bad)

    def test_analysis_exposes_only_entry_file_writes_and_no_private_or_network_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            s = object.__new__(T.Session)
            s.root = Path(directory)
            s.candidate = s.root/'candidate'
            s.analysis_count = 0
            s.deadline = time.monotonic()+60
            from contextlib import nullcontext
            class Pool:
                def owner(self,*args):return nullcontext()
                def __enter__(self):return self
                def __exit__(self,*args):pass
            class Host:simulator = Pool()
            s.coordinator = Host()
            s.worker_id = 'test'
            done = type('Completed',(),{'returncode':0,'stdout':'ok','stderr':''})()
            with patch.object(T.subprocess,'run',return_value=done) as run:
                s.analysis('print("ok")')
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--network')+1],'none')
            writes = [value for value in command if value.endswith(':rw')]
            self.assertEqual(len(writes),6)
            self.assertTrue(all(any(value.endswith('/'+entry+':rw') for entry in T.F.ENTRY_FILES) for value in writes))
            self.assertFalse(any('/tests' in value for value in command))

    def test_final_is_delegated_to_unchanged_verifier(self):
        with patch.object(T.Q,'evaluate',return_value={'reward':.5}) as evaluate:
            self.assertEqual(T.evaluate(Path('bundle'),Path('final'),final=True,timeout=20),{'reward':.5})
            self.assertEqual(evaluate.call_args.kwargs,{'final':True,'timeout':20})


if __name__ == '__main__':unittest.main()
