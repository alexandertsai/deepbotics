"""Planning flow and public input boundaries without provider calls."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task09_planning as W


def rows(count,prefix):
    return {"hypotheses":[{"id":f"hypothesis_{i}","title":f"{prefix} title {i}",
        "hypothesis":f"{prefix} distinct idea {i}","first_test":f"Test {i}",
        "success_criteria":f"Measure criterion {i}"} for i in range(1,count+1)]}


class Lane:
    def __init__(self,pool):
        self.pool=pool;self.condition=threading.Condition();self.deadline=time.monotonic()+999
        self.active_clients=set();self.registered={};self.calls=[]
    def register_worker(self,worker,max_calls):self.registered[worker]=max_calls
    def request(self,worker,body):
        self.calls.append((worker,body))
        if len(self.calls)==1:
            self.pool.barrier.wait(timeout=2)
            value=rows(3,worker)
        else:
            value=rows(6,'merged')
            if worker=='planner_b' and self.pool.invalid:
                value['hypotheses'][5]['hypothesis']=value['hypotheses'][0]['hypothesis']
        if len(self.calls)==2 and worker=='planner_b':
            self.pool.review_inputs=body
        response={"content":[{"type":"thinking","thinking":"internal"},
                             {"type":"text","text":json.dumps(value)}]}
        return 200,json.dumps(response).encode(),'application/json'


class Pool:
    def __init__(self,invalid=False):
        self.barrier=threading.Barrier(2);self.invalid=invalid
        self.lanes={worker:Lane(self) for worker in ('planner_a','planner_b')}
    def lane_for(self,worker):return self.lanes[worker]


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.starter=self.root/'starter';self.starter.mkdir()
        for name in W.Q.ENTRY_FILES:
            path=self.starter/name;path.parent.mkdir(exist_ok=True)
            path.write_text(f'Fresh public starter: {name}')
        (self.starter/'private-results.json').write_text('SECRET_EXTRA_FILE')
        self.task=self.root/'public-task';self.task.mkdir()
        (self.task/'instruction.md').write_text('Public task instruction without any experiment results.')
        self.patch=patch.object(W.Q,'TASK',self.task);self.patch.start();self.addCleanup(self.patch.stop)

    def test_independent_proposals_merge_review_and_exact_final_plan(self):
        pool=Pool();started=time.monotonic()
        plan=W.build_plan(pool,self.starter,self.root/'planning',started+500)
        self.assertEqual(len(plan['hypotheses']),6)
        self.assertEqual(set(plan['provenance']['input_sha256']),{'instruction.md',*W.Q.ENTRY_FILES})
        self.assertEqual(len(plan['provenance']['source_prompt_sha256']),4)
        for worker,lane in pool.lanes.items():
            self.assertEqual(lane.registered[worker],3)
            self.assertEqual(len(lane.calls),2)
            self.assertLessEqual(lane.deadline,started+180.1)
            initial=lane.calls[0][1]['messages'][0]['content']
            self.assertNotIn('SECRET_EXTRA_FILE',initial)
            self.assertNotIn('proposal_a',initial)
            self.assertEqual(lane.calls[0][1]['max_tokens'],8192)
        review=pool.review_inputs['messages'][0]['content']
        self.assertIn('proposal_a',review);self.assertIn('proposal_b',review);self.assertIn('Merged plan:',review)
        self.assertEqual(json.loads((self.root/'planning/plan.json').read_text()),plan)

    def test_invalid_duplicate_final_plan_fails_closed(self):
        pool=Pool(invalid=True)
        with self.assertRaisesRegex(W.PlanError,'distinct'):
            W.build_plan(pool,self.starter,self.root/'planning',time.monotonic()+10)
        self.assertFalse((self.root/'planning/plan.json').exists())
        self.assertTrue((self.root/'planning/planning-error.json').exists())
        self.assertTrue(all(lane.deadline<=time.monotonic() for lane in pool.lanes.values()))

    def test_symlink_escape_and_expired_deadline_make_no_requests(self):
        pool=Pool()
        with self.assertRaises(W.PlanError):
            W.build_plan(pool,self.starter,self.root/'expired',time.monotonic()-1)
        path=self.starter/W.Q.ENTRY_FILES[0];path.unlink()
        secret=self.root/'outside.txt';secret.write_text('PRIVATE')
        path.symlink_to(secret)
        with self.assertRaisesRegex(W.PlanError,'within'):
            W.build_plan(pool,self.starter,self.root/'escaped',time.monotonic()+10)
        self.assertFalse(any(lane.calls for lane in pool.lanes.values()))

    def test_strict_json_rejects_duplicate_keys_fences_and_nonfinite(self):
        for value in ('{"a":1,"a":2}','```json\n{}\n```','{"x":NaN}'):
            with self.assertRaises(W.PlanError):W.strict_json(value)
        with self.assertRaises(W.PlanError):W.validate(rows(3,'few'),6,final=True)


if __name__=='__main__':unittest.main()
