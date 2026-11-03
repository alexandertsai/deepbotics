"""Public candidate exchange and lead selection without inference or Docker."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task09_team_sessions as W


class TeamTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        row={'mass_kg':1,'mass_budget_kg':2.3,'passive_residual_max_nm':.3,
             'nominal':{'hold_rms_error_rad':.2,'servo_headroom_mean':.5},
             'adapted':{'hold_rms_error_rad':.2,'servo_headroom_mean':.5}}
        report={'variants':{v:copy.deepcopy(row) for v in W.F.Q.VARIANTS}}
        reference=W.P.ROOT/'outputs/task09-starter-baseline-01'
        with patch.object(W.F.Q,'evaluate',return_value=report):
            self.host=W.Coordinator(self.root/'team',reference,reference)
        self.host.start()

    def session(self,name='lead',condition='astra',lead=True):
        session=W.Session(self.root/name,self.host,name,'engineer','Use public evidence',condition=condition,lead=lead)
        session.start();return session

    def test_read_only_verified_measured_entry_files(self):
        result=self.host.read_candidate(-1)
        self.assertEqual(set(result['entry_files']),set(W.F.ENTRY_FILES))
        self.assertNotIn('bundle',result)
        self.assertNotIn('manifest',result)
        self.host.entries.append({'status':'queued','sha256':'unmeasured'})
        with self.assertRaises(ValueError):self.host.read_candidate(0)
        path=self.host.root/'starter'/W.F.ENTRY_FILES[0]
        path.chmod(0o644);path.write_text(path.read_text()+'\n')
        with self.assertRaises(Exception):self.host.read_candidate(-1)

    def test_only_lead_can_choose_and_source_is_frozen_without_candidate_overwrite(self):
        explorer=self.session('explorer','deepseek-rax',False)
        with self.assertRaises(ValueError):self.host.choose_candidate(explorer,-1,'Use starter')
        lead=self.session()
        path=lead.candidate/W.F.ENTRY_FILES[-1];path.write_text(path.read_text()+'\n# local draft\n')
        draft=path.read_text()
        reply=lead.dispatch('choose_candidate',{'index':-1,'reason':'Retain the measured legal incumbent'})
        self.assertTrue(reply['lead_submitted'])
        self.assertEqual(path.read_text(),draft)
        self.assertEqual(lead.submitted['files'],self.host.manifest['files'])
        self.host.best={**self.host.best,'score':.99}
        selected=self.host.select()
        self.assertEqual(selected['selection_policy'],'astra_lead_measured_choice')
        self.assertEqual(selected['source_index'],-1)
        self.assertFalse(selected['selection_used_final_scores'])
        with self.assertRaises(ValueError):lead.dispatch('choose_candidate',{'index':-1,'reason':'Another choice'})

    def test_fallback_retains_public_incumbent_with_explicit_reason(self):
        result=self.host.select()
        self.assertEqual(result['worker'],'unchanged_reference')
        self.assertEqual(result['selection_policy'],'public_best_fallback')
        self.assertIn('did not freeze',result['fallback_reason'])

    def test_both_conditions_get_same_public_rax_information_and_eight_probes(self):
        lead=self.session()
        explorer=self.session('explorer','deepseek-rax',False)
        for session in (lead,explorer):
            self.assertIn('RAX adaptation:',session.prompt)
            self.assertIn('at most 8 new probes',session.prompt)
            self.assertIn('second or third model response',session.prompt)
            self.assertIn('Starter franka/lead.xml:',session.prompt)
            self.assertIn('read_candidate',[tool['name'] for tool in session.definitions()])
        self.assertIn('choose_candidate',[tool['name'] for tool in lead.definitions()])
        self.assertNotIn('choose_candidate',[tool['name'] for tool in explorer.definitions()])
        self.assertEqual(len(W.DS_ROLES),6)
        self.assertEqual(self.host.seconds,1800)


if __name__=='__main__':unittest.main()
