"""Planner assignment gating and selector preservation without models or Docker."""

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task09_planner_short as W


class PlannerRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.assignment={"id":"hypothesis_1","title":"Independent proposal","hypothesis":"A testable change",
                         "first_test":"Measure changed candidate","success_criteria":"Improved public evidence"}

    def session(self):
        session=object.__new__(W.Session)
        session.assignment=self.assignment
        session.worker_id='hypothesis_1'
        session.first_assigned_probe_measured=False
        session.first_assigned_probe_origin=None
        session.deadline=time.monotonic()+60
        session.tests=0
        session.root=self.root
        return session

    def test_peer_access_and_submission_are_gated_before_assigned_test(self):
        session=self.session()
        for name,args in [('read_probe',{'index':0}),('share_finding',{'notes':'idea'}),('submit',{})]:
            with self.assertRaises(ValueError):session.dispatch(name,args)
        status=session.dispatch('status',{})
        self.assertFalse(status['peer_evidence_available'])
        self.assertNotIn('probes',status)
        with patch.object(W.F.SwarmSession,'dispatch') as dispatch:
            with self.assertRaises(ValueError):session.dispatch('test_candidate',{'hypothesis':'Unassigned test'})
            with self.assertRaises(ValueError):session.dispatch('test_candidate',{'hypothesis':'hypothesis_10'})
            dispatch.assert_not_called()

    def test_worker_first_test_preserves_assignment_in_forwarded_note(self):
        session=self.session()
        with patch.object(W.F.SwarmSession,'dispatch',return_value={'version':0}) as dispatch:
            session.dispatch('test_candidate',{'hypothesis':'hypothesis_1 measured proposal'})
        name,args=dispatch.call_args.args
        self.assertEqual(name,'test_candidate')
        self.assertIn(json.dumps(self.assignment),args['hypothesis'])

    def test_host_rehearsal_origin_is_explicit_and_selector_is_unchanged(self):
        coordinator=object.__new__(W.Coordinator)
        coordinator.root=self.root
        for name in W.F.ENTRY_FILES:
            for folder,value in [('starter','original'),('frozen','changed')]:
                path=self.root/folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
        session=self.session()
        with patch.object(W.F.Coordinator,'test',return_value={'version':1}) as test:
            coordinator.test(session,'Frozen branch rehearsal',source=self.root/'frozen')
        self.assertTrue(session.first_assigned_probe_measured)
        self.assertEqual(session.first_assigned_probe_origin,'host_rehearsal')
        self.assertIn(json.dumps(self.assignment),test.call_args.args[1])
        self.assertTrue(json.loads((self.root/'first-assigned-probe.json').read_text())[
            'implementation_semantics_not_automatically_verified'])
        self.assertIs(W.Coordinator.select,W.F.Coordinator.select)

    def test_unchanged_starter_cannot_unlock_first_test(self):
        coordinator=object.__new__(W.Coordinator);coordinator.root=self.root
        session=self.session();session.candidate=self.root/'candidate'
        for name in W.F.ENTRY_FILES:
            for folder in ('starter','candidate'):
                path=self.root/folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('same')
        with patch.object(W.F.Coordinator,'test') as test:
            with self.assertRaises(ValueError):coordinator.test(session,'hypothesis_1')
            test.assert_not_called()
        self.assertFalse(session.first_assigned_probe_measured)


if __name__=='__main__':unittest.main()
