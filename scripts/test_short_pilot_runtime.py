"""Observed-memory admission without Docker or provider calls."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import short_pilot_runtime as R


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_units_and_summed_docker_stats(self):
        self.assertEqual(R.memory_bytes('512MiB'),.5*R.GIB)
        self.assertEqual(R.memory_bytes('2.5 GiB'),2.5*R.GIB)
        self.assertEqual(R.memory_bytes('1000kB'),1_000_000)
        result=SimpleNamespace(stdout='{"MemUsage":"512MiB / 4GiB"}\n{"MemUsage":"2GiB / 4GiB"}\n')
        with patch.object(R.subprocess,'run',return_value=result) as run:
            self.assertEqual(R.docker_memory_bytes(timeout=.1),2.5*R.GIB)
        self.assertEqual(run.call_args.kwargs['timeout'],.1)

    def test_high_then_low_waits_and_logs_hold_and_memory_time(self):
        values=iter([7*R.GIB,8*R.GIB,6*R.GIB]);deadline=time.monotonic()+1
        log=self.root/'memory.jsonl'
        gate=R.MemoryGatedSimulationLock(self.root/'task.lock',lambda:deadline,log,
            poll_seconds=.005,memory_reader=lambda:next(values))
        with gate:
            time.sleep(.003)
        rows=[json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([r['event'] for r in rows],['acquired','released'])
        self.assertEqual(rows[-1]['stats_samples'],3)
        self.assertGreaterEqual(rows[-1]['memory_wait_seconds'],.009)
        self.assertGreater(rows[-1]['hold_seconds'],0)
        self.assertEqual(rows[-1]['peak_observed_memory_gib'],8)

    def test_stats_errors_fail_closed_until_fixed_deadline_and_unlock(self):
        calls=0
        def sliding_deadline():
            nonlocal calls
            calls+=1
            return time.monotonic()+.025
        def broken():
            raise subprocess.TimeoutExpired('docker stats',.01)
        path=self.root/'task.lock';log=self.root/'memory.jsonl'
        gate=R.MemoryGatedSimulationLock(path,sliding_deadline,log,poll_seconds=.003,memory_reader=broken)
        started=time.monotonic()
        with self.assertRaises(TimeoutError):
            with gate:self.fail('Stats failure admitted a simulation')
        self.assertLess(time.monotonic()-started,.25)
        self.assertEqual(calls,1)
        row=json.loads(log.read_text().splitlines()[-1])
        self.assertEqual(row['event'],'acquisition_failed')
        self.assertGreater(row['stats_errors'],0)
        self.assertEqual(row['hold_seconds'],0)
        with R.SimulationLock(path,lambda:time.monotonic()+1):pass


if __name__=='__main__':unittest.main()
