from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from task08_sim_pool import GIB, SimulationPool


class Task08SharedPool(unittest.TestCase):
    def pool(self, root, *, deadline=None, memory=lambda: 0, containers=lambda: set()):
        return SimulationPool(deadline or (lambda: time.monotonic() + 1), root / "pool.jsonl",
            memory_reader=memory, container_reader=containers, poll_seconds=.001)

    def test_both_conditions_share_four_slots_and_queue_fifth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deadline = time.monotonic() + 1
            pool = self.pool(root, deadline=lambda: deadline)
            release, all_four, fifth = threading.Event(), threading.Event(), threading.Event()
            count, mutex = [0], threading.Lock()
            def worker(index):
                with pool.owner("deepseek" if index % 2 else "astra", str(index)), pool:
                    with mutex:
                        count[0] += 1
                        if count[0] == 4:
                            all_four.set()
                    if index == 4:
                        fifth.set()
                    release.wait(.5)
            with ThreadPoolExecutor(max_workers=5) as executor:
                first = [executor.submit(worker, index) for index in range(4)]
                self.assertTrue(all_four.wait(.5))
                last = executor.submit(worker, 4)
                self.assertFalse(fifth.wait(.02))
                release.set()
                for future in [*first, last]:
                    future.result()
            self.assertEqual(pool.summary()["peak_active"], 4)
            self.assertEqual(pool.summary()["active"], 0)
            rows = [json.loads(line) for line in (root / "pool.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "acquired" for row in rows), 5)
            self.assertEqual({row["condition"] for row in rows}, {"deepseek", "astra"})

    def test_memory_block_permanently_reduces_capacity_and_records_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = iter((7 * GIB, GIB))
            pool = self.pool(root, memory=lambda: next(observations))
            with pool.owner("deepseek", "worker"), pool:
                self.assertEqual(pool.summary()["capacity"], 3)
            rows = [json.loads(line) for line in (root / "pool.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["event"], "capacity_reduced_for_memory")
            self.assertGreater(rows[1]["memory_wait_seconds"], 0)
            restored = self.pool(root)
            self.assertEqual(restored.summary()["capacity"], 3)

    def test_expired_memory_wait_uses_one_fixed_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def deadline():
                calls.append(True)
                return time.monotonic() + .02
            pool = self.pool(root, deadline=deadline, memory=lambda: 8 * GIB)
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                with pool:
                    self.fail("High-memory admission")
            self.assertEqual(len(calls), 1)
            self.assertLess(time.monotonic() - started, .3)
            self.assertEqual(pool.summary()["queued"], 0)

    def test_existing_task08_container_counts_against_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = self.pool(root, containers=lambda: {"existing-operator"})
            release, three = threading.Event(), threading.Event()
            count, lock = [0], threading.Lock()
            def worker():
                with pool:
                    with lock:
                        count[0] += 1
                        if count[0] == 3:
                            three.set()
                    release.wait(.1)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker) for _ in range(4)]
                self.assertTrue(three.wait(.5))
                time.sleep(.01)
                self.assertEqual(pool.summary()["active"], 3)
                release.set()
                for future in futures:
                    future.result()
            self.assertEqual(pool.summary()["peak_active"], 3)


if __name__ == "__main__":
    unittest.main()
