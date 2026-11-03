"""Task09 short-run wiring and final-feedback barrier without providers."""

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task09_swarm_short as W


class ShortTask09Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_baseline_is_locked_and_timeout_is_bounded_before_agents(self):
        args=SimpleNamespace(output=self.root/'run',simulation_lock=self.root/'sim.lock',
                             tests_per_worker=3,max_cost=10)
        events=[]
        class Lock:
            def __init__(self,path,deadline,log_path):
                self.deadline=deadline
                self.initial=deadline()
            def __enter__(self):
                events.append('locked')
                time.sleep(.002)
                self.assert_fixed=self.deadline()==self.initial
                if not self.assert_fixed:raise AssertionError('Sliding deadline')
            def __exit__(self,*exc):events.append('unlocked')
        def evaluate(*args,**kwargs):
            self.assertEqual(events,['locked'])
            self.assertGreater(kwargs['timeout'],0)
            self.assertLessEqual(kwargs['timeout'],2)
            return {'public':True}
        def constructor(*args,**kwargs):
            self.assertEqual(kwargs['worker_count'],6)
            self.assertEqual(kwargs['tests'],18)
            self.assertEqual(kwargs['cost'],10)
            W.F.Q.evaluate(self.root,self.root/'baseline')
            return 'coordinator'
        with patch.object(W,'simulation_lock',Lock),patch.object(W.F.Q,'evaluate',side_effect=evaluate) as original:
            with patch.object(W.F,'Coordinator',side_effect=constructor):
                self.assertEqual(W.setup_coordinator(args,time.time()+32,900),'coordinator')
            self.assertIs(W.F.Q.evaluate,original)
        self.assertEqual(events,['locked','unlocked'])

    def test_final_barrier_requires_fresh_both_condition_attestation(self):
        barrier=self.root/'barrier.json'
        started=time.time()
        for value in ({'released_at':started+1},
                      {'all_conditions_frozen_or_stopped':True,'released_at':started-1}):
            barrier.write_text(json.dumps(value))
            with self.assertRaises(ValueError):W.wait_final_release(barrier,time.time()+60,started)
        barrier.write_text(json.dumps({'all_conditions_frozen_or_stopped':True,'released_at':started+1}))
        W.wait_final_release(barrier,time.time()+60,started)
        barrier.unlink()
        with self.assertRaises(TimeoutError):W.wait_final_release(barrier,time.time(),started)

    def test_full_mocked_flow_closes_agents_before_private_final(self):
        output=self.root/'run';output.mkdir()
        barrier=self.root/'release.json';events=[];sessions=[];pool_options={}
        coordinator=SimpleNamespace(root=output,deadline=None,starter_probe_seconds=.1,tests=0,
            api_limits=dict(max_calls=216,per_worker_calls=36,max_tokens=6000000,max_output_tokens=8192,max_cost=10))
        coordinator.start=lambda:setattr(coordinator,'deadline',time.monotonic()+900)
        coordinator.rehearse_remaining=lambda s:events.append('public_rehearsal')
        coordinator.select=lambda:{'worker':'unchanged_reference'}
        class Session:
            def __init__(self,root,coord,worker,role,brief):
                self.worker_id,self.tests=worker,0;sessions.append(self)
        class Pool:
            def __init__(self,root,keys,**options):pool_options.update(options)
            def close(self):events.append('closed')
            def summary(self):return {'estimated_cost_usd':0}
        def workers(coord,client,workers,active):
            self.assertEqual((len(workers),active),(6,6));events.append('workers_stopped')
            return [{} for _ in workers]
        def evaluate(*args,**kwargs):
            self.assertIn('closed',events)
            self.assertIn('workers_stopped',events)
            self.assertTrue((output/'selection-frozen.json').exists())
            events.append('private_final');return {'reward':.5}
        def release(path,stop,not_before):
            self.assertEqual(events,['workers_stopped','closed','public_rehearsal'])
            events.append('both_conditions_frozen')
        class Lock:
            def __enter__(self):return self
            def __exit__(self,*exc):pass
        with patch.object(W,'simulation_lock',side_effect=lambda *args:Lock()),patch.object(W.F.Q,'preflight',return_value={'ok':True}),patch.object(W,'setup_coordinator',return_value=coordinator),patch.object(W,'Session',Session),patch.object(W,'AccountPool',Pool),patch.object(W,'primary_keys',return_value={'FIREWORKS_API_KEY':'mock'}),patch.object(W.F,'run_workers',side_effect=workers),patch.object(W,'wait_final_release',side_effect=release),patch.object(W.F.Q,'evaluate',side_effect=evaluate),patch.object(W,'hard_stop_timestamp',return_value=time.time()+3600):
            W.main(['--output',str(output),'--final-barrier',str(barrier),'--simulation-lock',str(self.root/'sim.lock')])
        self.assertEqual(pool_options['max_concurrent'],3)
        self.assertEqual(pool_options['generated_tpm'],60000)
        self.assertEqual(pool_options['logical_requests_per_worker'],12)
        self.assertEqual(pool_options['max_cost'],10)
        self.assertEqual(len(pool_options['worker_ids']),6)
        self.assertLess(events.index('both_conditions_frozen'),events.index('private_final'))
        protocol=json.loads((output/'protocol.json').read_text())
        self.assertEqual(protocol['new_development_probe_limit_total'],18)
        self.assertFalse(protocol['selection_used_private_results'])


if __name__=='__main__':unittest.main()
