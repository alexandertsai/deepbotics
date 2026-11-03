"""Reported pacing capacity and inherited guards without provider calls."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from throughput_paced_client import ThroughputPacedClient


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def client(self,**options):
        values=dict(root=Path(self.temp.name),keys={'FIREWORKS_API_KEY':'mock'},
            deadline=time.monotonic()+5,start_server=False,transport=self.success,
            start_stagger_seconds=0,max_output_tokens=100,_window_seconds=.03,
            base_backoff_seconds=.005,jitter_seconds=0)
        values.update(options);client=ThroughputPacedClient(**values);self.addCleanup(client.close)
        return client

    @staticmethod
    def success(*args):
        return 200,json.dumps({'usage':{'input_tokens':5,'output_tokens':3},'content':[]}).encode(),'application/json'

    @staticmethod
    def body():return {'messages':[{'role':'user','content':'Public task'}]}

    def test_no_headers_uses_future_fallbacks(self):
        client=self.client();client.register_worker('a');client.request('a',self.body())
        summary=client.summary()
        self.assertEqual(summary['max_concurrent'],3)
        self.assertEqual(summary['pacing']['effective_tpm'],{'prompt':500000,'uncached':250000,'generated':60000})

    def test_reported_limits_can_raise_or_lower_fallback_at_ninety_percent(self):
        client=self.client()
        client._apply_headers(200,{'x-ratelimit-limit-tokens-generated':'70312',
            'x-ratelimit-limit-tokens-prompt':'1000000','x-ratelimit-limit-tokens-uncached-prompt':'10000'},0)
        pacing=client.summary()['pacing']
        self.assertEqual(pacing['effective_tpm'],{'prompt':900000,'uncached':9000,'generated':63280})
        self.assertEqual(pacing['raw_reported_limit_tpm']['generated'],70312)
        client._apply_headers(200,{'x-ratelimit-limit-tokens-generated':'1000'},0)
        self.assertEqual(client.summary()['pacing']['effective_tpm']['generated'],900)

    def test_lower_output_header_clamps_later_request_and_records_raw_limit(self):
        seen=[]
        def transport(body,timeout):
            seen.append(json.loads(body)['max_tokens'])
            return (*self.success(),{'x-ratelimit-limit-tokens-generated':'2000'})
        client=self.client(transport=transport,max_output_tokens=4096)
        client.register_worker('a');client.request('a',self.body());client.request('a',self.body())
        self.assertEqual(seen,[4096,1800])
        self.assertEqual(client.records[0]['raw_reported_limit_tpm']['generated'],2000)

    def test_retry_after_backoff_and_unknown_billing_allowance_are_preserved(self):
        starts=[]
        def transport(body,timeout):
            starts.append(time.monotonic())
            if len(starts)==1:
                return 429,b'{}','application/json',{'Retry-After':'.01',
                    'x-ratelimit-limit-tokens-generated':'70312'}
            return self.success()
        client=self.client(transport=transport);client.register_worker('a')
        self.assertEqual(client.request('a',self.body())[0],200)
        self.assertGreaterEqual(starts[1]-starts[0],.009)
        self.assertEqual(client.calls,2)
        self.assertEqual(client.logical_requests,1)
        self.assertEqual(client.records[0]['applied_retry_delay_seconds'],.01)
        self.assertTrue(client.records[0]['rejected_billing_unknown'])
        self.assertGreater(client.summary()['unknown_usage_token_allowance'],0)
        self.assertLessEqual(client.summary()['pacing']['effective_tpm']['generated'],63280)


if __name__=='__main__':unittest.main()
