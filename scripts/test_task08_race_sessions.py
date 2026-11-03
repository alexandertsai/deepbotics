"""Matched Task08 race limits, ownership, and probe timing without inference."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task08_race_sessions as W


class RaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.host=W.Coordinator(self.root/'race',W.P.ROOT/'outputs/task08-reference-baseline-01',
            W.P.ROOT/'outputs/task08-pilot-02',seconds=300,tests=24,calls=216,calls_per_worker=12)
        self.t0=time.monotonic();self.host.start(started_monotonic=self.t0,deadline=self.t0+300)

    def session(self,name,condition):
        session=W.Session(self.root/name,self.host,name,'same-role','same-brief',condition=condition)
        session.start();return session

    def test_prompt_information_and_rax_methods_match_across_conditions(self):
        ds=self.session('ds','deepseek-rax');astra=self.session('astra','astra')
        self.assertEqual(ds.prompt.replace('worker ID is ds','worker ID is WORKER'),
                         astra.prompt.replace('worker ID is astra','worker ID is WORKER'))
        self.assertIn('RAX adaptation:',astra.prompt)
        self.assertIn('second or third model response',ds.prompt)
        self.assertEqual(ds.deadline,astra.deadline)
        self.assertEqual(len(W.DS_ROLES),6)

    def test_probe_owner_and_monotonic_queue_events(self):
        session=self.session('ds','deepseek-rax')
        path=session.candidate/'controller.py';path.write_text(path.read_text()+'\n# change\n')
        events=[]
        class Shared:
            @contextmanager
            def bindowner(self,condition,worker):
                events.append(('owner',condition,worker));yield
            def __enter__(self):events.append(('enter',));return self
            def __exit__(self,*exc):events.append(('exit',))
        self.host.simulator=Shared()
        with patch.object(W.P,'docker_evaluate',return_value=copy.deepcopy(self.host.baseline_report)):
            session.dispatch('test_candidate',{'hypothesis':'Concrete change'})
        self.assertEqual(events[0],('owner','deepseek-rax','ds'))
        rows=[json.loads(x) for x in (self.host.root/'probe-events.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in rows],['requested','queued','started','measured'])
        self.assertGreaterEqual(rows[2]['queue_wait_seconds'],0)
        self.assertEqual([r['monotonic'] for r in rows],sorted(r['monotonic'] for r in rows))

    def test_four_probe_limit_and_no_rehearsal_after_deadline(self):
        session=self.session('ds','deepseek-rax')
        with patch.object(W.P,'docker_evaluate',return_value=copy.deepcopy(self.host.baseline_report)) as evaluate:
            for i in range(5):
                path=session.candidate/'controller.py';path.write_text(path.read_text()+f'\n# change{i}\n')
                reply=session.tool('test_candidate',{'hypothesis':'Concrete change'})
                self.assertEqual(reply['isError'],i==4)
        self.assertEqual(evaluate.call_count,4)
        self.host.deadline=time.monotonic()-1
        self.assertEqual(self.host.rehearse_remaining([session]),[])
        self.assertIs(W.Coordinator.select,W.S.Coordinator.select)
        with self.assertRaises(ValueError):self.host.start(started_monotonic=self.t0,deadline=self.t0+301)


if __name__=='__main__':unittest.main()
