"""Fixed worker routing across all three separately paced authorized accounts."""

from pathlib import Path
import json
import stat
import threading

from throughput_paced_client import ThroughputPacedClient as PacedClient
import task08_pilot as P

ACCOUNT_COUNT = 3


def stored_keys(filename):
    path = Path.home() / '.config/kimirax' / filename
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError('Credentials must be a regular owner-only file')
    value = json.loads(path.read_text())
    if not value.get('FIREWORKS_API_KEY'):
        raise ValueError('Configured Fireworks credentials unavailable')
    return {'FIREWORKS_API_KEY': value['FIREWORKS_API_KEY']}


def secondary_keys():
    return stored_keys('fireworks-secondary.json')


def third_keys():
    return stored_keys('fireworks-friend-20261102.json')


class AccountPool:
    def __init__(self, root, keys, *, worker_ids=None, secondary=None, third=None,
                 client_factory=PacedClient, **options):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.assignments = {name: i % ACCOUNT_COUNT for i, name in enumerate(worker_ids or [])}
        self.lanes = []
        self.deadline = options['deadline']
        self.max_cost = options.get('max_cost', 5.)
        self.max_workers = options.get('max_workers', 32)
        if len(self.assignments) > self.max_workers:
            raise ValueError('Too many assigned workers')
        other = secondary_keys() if secondary is None else secondary
        newest = third_keys() if third is None else third
        credentials = (keys, other, newest)
        values = [entry.get('FIREWORKS_API_KEY') for entry in credentials]
        if any(not isinstance(value,str) or not value for value in values) or len(set(values)) != ACCOUNT_COUNT:
            raise ValueError('Three distinct authorized account credentials are required')
        for name,default in (('max_calls',270),('max_tokens',6_000_000)):
            if options.get(name,default) < ACCOUNT_COUNT:
                raise ValueError('Combined request and token guards must allow all three accounts')
        cost_share=self.max_cost/ACCOUNT_COUNT
        cost_guards=[cost_share]*(ACCOUNT_COUNT-1)
        cost_guards.append(self.max_cost-sum(cost_guards))
        try:
            for index, account in enumerate(credentials):
                lane_options = dict(options)
                lane_options.setdefault('max_concurrent',3)
                for name, default in (('max_calls', 270), ('max_tokens', 6_000_000)):
                    total = options.get(name, default)
                    lane_options[name] = total // ACCOUNT_COUNT + (index < total % ACCOUNT_COUNT)
                lane_options['max_cost'] = cost_guards[index]
                lane_options['max_workers'] = options.get('max_workers', 32)
                self.lanes.append(client_factory(self.root / f'account-{index+1}',
                    {'FIREWORKS_API_KEY':account['FIREWORKS_API_KEY']}, **lane_options))
        except BaseException:
            for lane in self.lanes:
                lane.close()
            raise
        self._persist()

    def _persist(self):
        P.write_json(self.root / 'worker-accounts.json',
                     {name: index + 1 for name, index in self.assignments.items()})

    def lane_for(self, worker_id):
        with self.lock:
            if worker_id not in self.assignments:
                if len(self.assignments)>=self.max_workers:raise ValueError('Worker count limit reached')
                self.assignments[worker_id] = len(self.assignments) % ACCOUNT_COUNT
                self._persist()
            return self.lanes[self.assignments[worker_id]]

    def run_worker(self, session, worker_id, max_calls=None):
        result = self.lane_for(worker_id).run_worker(session, worker_id, max_calls=max_calls)
        result['account_lane'] = self.assignments[worker_id] + 1
        self.summary()
        return result

    def summary(self):
        with self.lock:
            rows = [lane.summary() for lane in self.lanes]
            sums = ('requests', 'reported_tokens', 'unknown_usage_token_allowance',
                    'estimated_cost_usd', 'errors', 'active_requests', 'queued_requests',
                    'inflight_token_reservations', 'inflight_estimated_cost_usd',
                    'aggregate_active_slot_seconds', 'logical_requests', 'provider_attempts',
                    'rate_limited_attempts', 'rejected_request_estimated_cost_allowance_usd',
                    'rejected_request_unknown_token_allowance')
            result = {name: sum(row.get(name, 0) for row in rows) for name in sums}
            result.update(accounts=len(self.lanes), per_account_max_concurrent=self.lanes[0].max_concurrent,
                          max_concurrent=sum(lane.max_concurrent for lane in self.lanes),
                          max_workers=self.lanes[0].max_workers,
                          estimated_cost_guard_usd=self.max_cost,
                          guard_allocation='Combined cost, token and request ceilings split across three accounts',
                          budget_exhausted=all(row.get('budget_exhausted', False) for row in rows),
                          peak_active_requests=sum(row.get('peak_active_requests', 0) for row in rows),
                          peak_active_is_sum_of_account_peaks=True,
                          usage={key: sum(row.get('usage', {}).get(key, 0) for row in rows)
                                 for key in set().union(*(row.get('usage', {}) for row in rows))},
                          worker_requests={key: value for row in rows for key, value in row.get('worker_requests', {}).items()},
                          worker_accounts={key: value + 1 for key, value in self.assignments.items()},
                          per_account=rows)
            P.write_json(self.root / 'api-summary.json', result)
            return result

    def close(self, **kwargs):
        failures = []
        for lane in self.lanes:
            try:
                lane.close(**kwargs)
            except Exception as error:
                failures.append(type(error).__name__)
        self.summary()
        if failures:
            raise RuntimeError('Account shutdown failed: ' + ', '.join(failures))
