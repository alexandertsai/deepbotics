import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from short_pilot_pool import AccountPool


class FakeClient:
    def __init__(self, root, keys, **options):
        self.options = options
        self.max_concurrent = options['max_concurrent']
        self.max_workers = options['max_workers']
        self.closed = False

    def summary(self):
        return {'estimated_cost_usd': self.options['max_cost'], 'usage': {'input_tokens': 7},
                'requests': 1, 'active_requests': 0}

    def run_worker(self, session, worker_id, max_calls=None):
        return {'worker_id': worker_id}

    def close(self, **kwargs):
        self.closed = True


class PoolTests(unittest.TestCase):
    def make_pool(self, root):
        return AccountPool(root, {'FIREWORKS_API_KEY': 'primary-test'},
            secondary={'FIREWORKS_API_KEY': 'secondary-test'}, client_factory=FakeClient,
            third={'FIREWORKS_API_KEY':'third-test'},
            deadline=100, worker_ids=['a','b','c','d','e','f'], max_concurrent=3,
            max_workers=6, max_cost=6, max_tokens=101, max_calls=217)

    def test_aggregate_ceilings_and_usage(self):
        with tempfile.TemporaryDirectory() as root:
            pool = self.make_pool(Path(root))
            for key, total in [('max_cost',6),('max_tokens',101),('max_calls',217)]:
                self.assertEqual(sum(l.options[key] for l in pool.lanes), total)
            self.assertEqual(pool.summary()['usage']['input_tokens'],21)
            self.assertEqual(pool.summary()['estimated_cost_usd'],6)
            self.assertEqual(pool.summary()['accounts'],3)
            self.assertEqual(pool.summary()['max_concurrent'],9)

    def test_worker_keeps_account_across_calls(self):
        with tempfile.TemporaryDirectory() as root:
            pool = self.make_pool(Path(root))
            self.assertIs(pool.lane_for('a'),pool.lane_for('d'))
            self.assertIsNot(pool.lane_for('a'),pool.lane_for('b'))
            self.assertIsNot(pool.lane_for('b'),pool.lane_for('c'))
            self.assertEqual([pool.assignments[x] for x in 'abcdef'],[0,1,2,0,1,2])
            self.assertEqual(pool.run_worker(None,'a')['account_lane'],1)
            self.assertEqual(pool.run_worker(None,'a')['account_lane'],1)

    def test_all_accounts_are_closed(self):
        with tempfile.TemporaryDirectory() as root:
            pool = self.make_pool(Path(root))
            pool.close()
            self.assertTrue(all(l.closed for l in pool.lanes))

    def test_default_configuration_loads_all_three_accounts(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('short_pilot_pool.secondary_keys',return_value={'FIREWORKS_API_KEY':'second'}), \
                 patch('short_pilot_pool.third_keys',return_value={'FIREWORKS_API_KEY':'third'}):
                pool=AccountPool(Path(root),{'FIREWORKS_API_KEY':'first'},client_factory=FakeClient,
                    deadline=100,max_workers=6,worker_ids=list('abcdef'),max_cost=10)
            self.assertEqual(len(pool.lanes),3)
            self.assertEqual(sum(l.options['max_cost'] for l in pool.lanes),10)
            self.assertEqual([pool.run_worker(None,w)['account_lane'] for w in 'abcdef'],[1,2,3,1,2,3])

    def test_missing_third_account_fails_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('short_pilot_pool.secondary_keys',return_value={'FIREWORKS_API_KEY':'second'}), \
                 patch('short_pilot_pool.third_keys',side_effect=FileNotFoundError('Unavailable')), \
                 patch('short_pilot_pool.PacedClient') as client:
                with self.assertRaises(FileNotFoundError):
                    AccountPool(Path(root),{'FIREWORKS_API_KEY':'first'},client_factory=client,deadline=100)
                client.assert_not_called()


if __name__ == '__main__': unittest.main()
