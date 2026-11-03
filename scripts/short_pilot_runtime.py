"""Task-scoped simulation locks with observed Docker memory admission."""

import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

from task08_swarm_short import SimulationLock

GIB = 1024 ** 3


def memory_bytes(value):
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTPE]?i?B)\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError("Unrecognized Docker memory unit")
    amount, unit = float(match[1]), match[2].lower()
    exponent = "bkmgtpe".index(unit[0]) if unit != "b" else 0
    return amount * ((1024 if "i" in unit else 1000) ** exponent)


def docker_memory_bytes(*, timeout=5.0):
    result = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}"],
        check=True, capture_output=True, text=True, timeout=timeout)
    total = 0.0
    for line in result.stdout.splitlines():
        if line.strip():
            row = json.loads(line)
            total += memory_bytes(row["MemUsage"].split("/", 1)[0])
    return total


class MemoryGatedSimulationLock:
    def __init__(self, path, deadline, log_path, *, memory_limit_gib=6.5,
                 poll_seconds=1.0, memory_reader=None):
        if not math.isfinite(memory_limit_gib) or memory_limit_gib <= 0:
            raise ValueError("Memory limit must be positive and finite")
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("Memory poll interval must be positive and finite")
        self.path, self.deadline, self.log_path = Path(path), deadline, Path(log_path)
        self.memory_limit_gib, self.poll_seconds = memory_limit_gib, poll_seconds
        self.memory_reader = memory_reader
        self.local = threading.local()

    def _log(self, state, event):
        now = time.monotonic()
        record = {key: value for key, value in state.items() if key not in ("lock", "started", "task_acquired", "admitted")}
        record.update(event=event, at=time.time(),
            queued_wait_seconds=max(0, state.get("task_acquired", now)-state["started"]),
            memory_wait_seconds=max(0, state.get("admitted", now)-state.get("task_acquired", now)),
            hold_seconds=max(0, now-state["admitted"]) if "admitted" in state else 0)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, allow_nan=False)+"\n")
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)

    def __enter__(self):
        fixed_deadline = float(self.deadline())
        if not math.isfinite(fixed_deadline):
            raise ValueError("Simulation deadline must be finite")
        state = {"acquisition": uuid.uuid4().hex, "pid": os.getpid(), "thread": threading.get_ident(),
            "path": str(self.path), "started": time.monotonic(), "deadline_monotonic": fixed_deadline,
            "memory_limit_gib": self.memory_limit_gib, "stats_samples": 0, "stats_errors": 0,
            "peak_observed_memory_gib": 0.0}
        lock = SimulationLock(self.path, lambda: fixed_deadline)
        acquired = False
        try:
            lock.__enter__()
            acquired = True
            state["task_acquired"] = time.monotonic()
            while True:
                remaining = fixed_deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Docker memory gate deadline reached")
                try:
                    observed = (docker_memory_bytes(timeout=min(5,remaining))
                                if self.memory_reader is None else self.memory_reader())
                    if not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed < 0:
                        raise ValueError("Invalid Docker memory observation")
                except Exception as error:
                    state["stats_errors"] += 1
                    state["last_stats_error"] = type(error).__name__
                else:
                    state["stats_samples"] += 1
                    state["peak_observed_memory_gib"] = max(state["peak_observed_memory_gib"],observed/GIB)
                    state["admission_memory_gib"] = observed/GIB
                    if observed <= self.memory_limit_gib*GIB and time.monotonic() < fixed_deadline:
                        state["admitted"] = time.monotonic()
                        state["lock"] = lock
                        self.local.state = state
                        self._log(state,"acquired")
                        return self
                time.sleep(min(self.poll_seconds,max(0,fixed_deadline-time.monotonic())))
        except BaseException as error:
            state["failure"] = type(error).__name__
            try:
                self._log(state,"acquisition_failed")
            finally:
                if acquired:
                    lock.__exit__(None,None,None)
                if hasattr(self.local,"state"):
                    del self.local.state
            raise

    def __exit__(self, *exc):
        state = self.local.state
        try:
            self._log(state,"released")
        finally:
            state["lock"].__exit__(*exc)
            del self.local.state
