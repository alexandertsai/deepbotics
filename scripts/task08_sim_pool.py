"""Shared bounded Task08 simulation admissions across both model conditions."""

from collections import deque
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

from short_pilot_runtime import GIB, docker_memory_bytes


def task08_container_ids(*, timeout=5):
    completed = subprocess.run(["docker", "ps", "--format", "{{json .}}"],
        capture_output=True, text=True, check=True, timeout=timeout)
    return {str(row["ID"]) for line in completed.stdout.splitlines() if line.strip()
        for row in [json.loads(line)] if re.search(r"task0?8(?:[^0-9]|$)", str(row))}


class SimulationPool:
    def __init__(self, deadline_callable, log_path, max_concurrent=4, memory_limit_gib=6.5,
                 *, memory_reader=None, container_reader=None, poll_seconds=.1):
        if type(max_concurrent) is not int or not 1 <= max_concurrent <= 4:
            raise ValueError("Simulation capacity must be between one and four")
        if not math.isfinite(memory_limit_gib) or memory_limit_gib <= 0 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("Memory threshold and polling interval must be positive and finite")
        self.deadline, self.log_path = deadline_callable, Path(log_path)
        self.state_path = self.log_path.with_suffix(self.log_path.suffix + ".state.json")
        self.max_concurrent, self.capacity = max_concurrent, max_concurrent
        self.memory_limit_gib, self.poll_seconds = memory_limit_gib, poll_seconds
        self.memory_reader, self.container_reader = memory_reader, container_reader
        self.condition, self.local = threading.Condition(threading.RLock()), threading.local()
        self.active, self.queue, self.peak_active = {}, deque(), 0
        self.external_ids = None
        self.memory_reduced = False
        if self.state_path.is_file():
            previous = json.loads(self.state_path.read_text())
            if previous.get("memory_reduced"):
                self.capacity, self.memory_reduced = min(3, self.capacity), True

    @contextmanager
    def owner(self, condition, worker_id):
        if not isinstance(condition, str) or not condition or not isinstance(worker_id, str) or not worker_id:
            raise ValueError("Condition and worker must be nonempty strings")
        previous = getattr(self.local, "owner", None)
        self.local.owner = {"condition": condition, "worker": worker_id}
        try:
            yield self
        finally:
            self.local.owner = previous

    bind_owner = owner

    def _persist(self):
        value = {"capacity": self.capacity, "configured_capacity": self.max_concurrent,
            "memory_reduced": self.memory_reduced, "memory_limit_gib": self.memory_limit_gib,
            "active": len(self.active), "peak_active": self.peak_active}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
        temporary.replace(self.state_path)

    def _log(self, state, event):
        now = time.monotonic()
        memory_wait = state["memory_wait_seconds"]
        if state.get("memory_blocked_since") is not None:
            memory_wait += now - state["memory_blocked_since"]
        admission = state.get("admitted", now)
        record = {key: value for key, value in state.items() if key not in ("started", "admitted", "memory_blocked_since")}
        record.update(event=event, at=time.time(), capacity=self.capacity,
            active=len(self.active), peak_active=self.peak_active,
            memory_wait_seconds=max(0, memory_wait),
            queued_wait_seconds=max(0, admission - state["started"] - memory_wait),
            hold_seconds=max(0, now - state["admitted"]) if "admitted" in state else 0)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    def summary(self):
        with self.condition:
            return {"capacity": self.capacity, "configured_capacity": self.max_concurrent,
                "memory_reduced": self.memory_reduced, "active": len(self.active),
                "queued": len(self.queue), "peak_active": self.peak_active,
                "external_containers_at_first_admission": sorted(self.external_ids or [])}

    def __enter__(self):
        if getattr(self.local, "state", None) is not None:
            raise RuntimeError("Nested simulation acquisitions are forbidden")
        deadline = float(self.deadline())
        if not math.isfinite(deadline):
            raise ValueError("Simulation deadline must be finite")
        state = {"acquisition": uuid.uuid4().hex, "started": time.monotonic(),
            "deadline_monotonic": deadline, "thread": threading.get_ident(),
            **(getattr(self.local, "owner", None) or {"condition": "unknown", "worker": "unknown"}),
            "memory_wait_seconds": 0.0, "memory_blocked_since": None,
            "stats_samples": 0, "stats_errors": 0}
        token = state["acquisition"]
        with self.condition:
            self.queue.append(token)
        try:
            while True:
                with self.condition:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Task08 simulation pool deadline reached")
                    if self.queue[0] != token or len(self.active) >= self.capacity:
                        self.condition.wait(timeout=min(self.poll_seconds, remaining))
                        continue
                try:
                    observed = (docker_memory_bytes(timeout=min(5, remaining)) if self.memory_reader is None
                                else self.memory_reader())
                    containers = (task08_container_ids(timeout=min(5, max(.001, deadline - time.monotonic())))
                                  if self.container_reader is None else self.container_reader())
                    if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed < 0:
                        raise ValueError("Invalid Docker memory observation")
                    containers = set(containers)
                except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError) as error:
                    state["stats_errors"] += 1
                    state["last_stats_error"] = type(error).__name__
                    with self.condition:
                        self.condition.wait(timeout=min(self.poll_seconds, max(0, deadline - time.monotonic())))
                    continue
                with self.condition:
                    state["stats_samples"] += 1
                    state["observed_memory_gib"] = observed / GIB
                    if self.external_ids is None:
                        self.external_ids = containers
                    external = len(containers & self.external_ids)
                    state["existing_external_task08_containers"] = external
                    if observed > self.memory_limit_gib * GIB:
                        if state["memory_blocked_since"] is None:
                            state["memory_blocked_since"] = time.monotonic()
                        if not self.memory_reduced:
                            self.capacity = min(3, self.capacity)
                            self.memory_reduced = True
                            self._persist()
                            self._log(state, "capacity_reduced_for_memory")
                            self.condition.notify_all()
                    else:
                        if state["memory_blocked_since"] is not None:
                            state["memory_wait_seconds"] += time.monotonic() - state["memory_blocked_since"]
                            state["memory_blocked_since"] = None
                        if len(self.active) + external < self.capacity and time.monotonic() < deadline:
                            self.queue.popleft()
                            state["admitted"] = time.monotonic()
                            self.active[token] = state
                            self.peak_active = max(self.peak_active, len(self.active))
                            self.local.state = state
                            self._persist()
                            self._log(state, "acquired")
                            self.condition.notify_all()
                            return self
                    self.condition.wait(timeout=min(self.poll_seconds, max(0, deadline - time.monotonic())))
        except BaseException as error:
            with self.condition:
                if token in self.queue:
                    self.queue.remove(token)
                if token in self.active:
                    self.active.pop(token)
                state["failure"] = type(error).__name__
                self._log(state, "acquisition_failed")
                self._persist()
                self.condition.notify_all()
            raise

    def __exit__(self, *exc):
        state = self.local.state
        with self.condition:
            try:
                self._log(state, "released")
            finally:
                self.active.pop(state["acquisition"])
                del self.local.state
                self._persist()
                self.condition.notify_all()


SharedHostSimulationPool = SimulationPool
