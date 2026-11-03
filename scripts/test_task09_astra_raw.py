import json
from pathlib import Path
import tempfile
import unittest

import task09_astra_raw as R


class RawUsageTests(unittest.TestCase):
    def test_request_records_are_not_duplicated_and_cache_is_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'agent.jsonl').write_text('{"type":"thread.started","thread_id":"raw-thread"}\n')
            path=root/'agent-home/sessions/rollout-raw-thread.jsonl'
            path.parent.mkdir(parents=True)
            usage={'input_tokens':1000,'cached_input_tokens':700,'output_tokens':200}
            row={'type':'event_msg','payload':{'type':'token_count','info':{
                'total_token_usage':usage,'last_token_usage':usage}}}
            path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
            result=R.usage_audit(root)
            self.assertEqual(len(result['request_usage_records']),1)
            self.assertAlmostEqual(result['request_level_api_equivalent_usd'],.0137)
            self.assertAlmostEqual(result['standard_credit_equivalent'],.3425)
            self.assertIsNone(result['subscription_incremental_charge_usd'])

    def test_long_context_tier_is_applied_per_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'agent-home/sessions/rollout.jsonl'
            path.parent.mkdir(parents=True)
            usage={'input_tokens':300000,'cached_input_tokens':200000,'output_tokens':1000}
            row={'type':'event_msg','payload':{'type':'token_count','info':{
                'total_token_usage':usage,'last_token_usage':usage}}}
            path.write_text(json.dumps(row)+'\n')
            result=R.usage_audit(root)
            self.assertTrue(result['request_usage_records'][0]['long_context'])
            self.assertAlmostEqual(result['request_level_api_equivalent_usd'],2.475)


if __name__=='__main__':unittest.main()
