"""Matched public Task08 sessions for a 300-second paired race."""

from contextlib import nullcontext
import json
from pathlib import Path
import threading
import time

import task08_pilot as P
import task08_swarm as S

DS_ROLES = (*S.ROLES,
    ("payload_margin", "Improve payload-dependent timing and posture while preserving a complete usable controller for all three arms."),
    ("conservative_route", "Explore a conservative approach or settling improvement across shelf targets while preserving legal geometry."))
ROLES = DS_ROLES
ASTRA_ROLE = ("astra", "Improve the complete common-base design and controller across all three arms using measured public evidence.")


class ProbeLock:
    def __init__(self, coordinator, simulator, context):
        self.coordinator, self.simulator, self.context = coordinator, simulator, context

    def __enter__(self):
        queued = time.monotonic()
        self.coordinator.record(self.context,"queued",queue_wait_seconds=0)
        result = self.simulator.__enter__()
        self.context["queue_wait_seconds"] = time.monotonic()-queued
        self.coordinator.record(self.context,"started",queue_wait_seconds=self.context["queue_wait_seconds"])
        return result

    def __exit__(self,*exc):
        return self.simulator.__exit__(*exc)


class Coordinator(S.Coordinator):
    def __init__(self,*args,seconds=300,tests_per_worker=4,**kwargs):
        if not 1 <= seconds <= 300 or not 1 <= tests_per_worker <= 4:
            raise ValueError("Race allows at most 300 seconds and four probes per worker")
        self.probe_context = threading.local()
        self.event_lock = threading.Lock()
        super().__init__(*args,seconds=seconds,tests_per_worker=tests_per_worker,**kwargs)

    @property
    def simulator(self):
        context = getattr(self.probe_context,"current",None)
        return self.shared_simulator if context is None else ProbeLock(self,self.shared_simulator,context)

    @simulator.setter
    def simulator(self,value):
        self.shared_simulator = value

    def start(self,*,started_monotonic=None,deadline=None):
        self.started_monotonic = time.monotonic() if started_monotonic is None else started_monotonic
        self.deadline = self.started_monotonic+self.seconds if deadline is None else deadline
        if self.deadline > self.started_monotonic+300:
            raise ValueError("Race deadline exceeds common 300-second limit")

    def record(self,context,event,**fields):
        row = {key:context[key] for key in ("condition","worker_id","version")}
        row.update(event=event,monotonic=time.monotonic(),at=time.time(),**fields)
        with self.event_lock:
            with (self.root/"probe-events.jsonl").open("a") as handle:
                handle.write(json.dumps(row,allow_nan=False)+"\n")

    def test(self,session,hypothesis,source=None):
        context = {"condition":session.condition,"worker_id":session.worker_id,
            "version":session.version_count,"queue_wait_seconds":0}
        self.record(context,"requested")
        binder = getattr(self.shared_simulator,"bindowner",None) or getattr(self.shared_simulator,"bind_owner",None)
        ownership = nullcontext() if binder is None else binder(session.condition,session.worker_id)
        self.probe_context.current = context
        try:
            with ownership:
                result = super().test(session,hypothesis,source=source)
            self.record(context,"cached" if result.get("cached") else "measured",
                queue_wait_seconds=context["queue_wait_seconds"],partial_score=result["partial_score"])
            return result
        except Exception as error:
            self.record(context,"failed",error=type(error).__name__,queue_wait_seconds=context["queue_wait_seconds"])
            raise
        finally:
            del self.probe_context.current

    def rehearse_remaining(self,sessions):
        if self.deadline is None or time.monotonic() >= self.deadline or self.tests >= self.max_tests:
            return []
        return super().rehearse_remaining([s for s in sessions if s.tests < min(4,self.tests_per_worker)])


class Session(S.SwarmSession):
    def __init__(self,root,coordinator,worker_id,role,brief,*,condition="deepseek-rax"):
        if condition not in ("deepseek-rax","deepseek","astra"):
            raise ValueError("Unknown race condition")
        super().__init__(root,coordinator,worker_id,role,brief)
        self.condition = "deepseek-rax" if condition == "deepseek" else condition
        self.prompt = self.make_prompt()
        (self.root/"prompt.txt").write_text(self.prompt)

    def make_prompt(self):
        condition = self.condition
        try:
            self.condition = "deepseek-rax"
            text = P.Session.make_prompt(self)
        finally:
            self.condition = condition
        starter = self.coordinator.root/"starter"
        text += f"""
Paired Task08 race. Your worker ID is {self.worker_id}; your role is {self.role}: {self.brief}
Every worker has the same fixed 300-second development window, four new public probe opportunities,
and a target allowance of 12 logical model responses. Queue and simulator waits count against the window.
All complete starter files and cached public starter measurements are below. Use them directly.
Implement a concrete candidate and request its first test by your second or third model response.
Revise from measured evidence and test again. Avoid source re-reading and status polling.
Each worker owns its candidate. No additional agents may be spawned. Use only the provided task08 tools.
The host selects the highest measured Task08 public partial score and retains the unchanged starter on ties or regressions.
Only measured frozen candidates are eligible. Preserve a tested incumbent and submit before development ends.
Final evaluation begins only after both conditions stop and freeze selection; its results are unavailable during development.
The earlier prompt's total probe count is shared; your own branch may use at most four new probes.
Starter public development evidence:
{json.dumps(S.compact_report(self.coordinator.baseline_report),separators=(',',':'))}
Starter controller.py:
{(starter/'controller.py').read_text()}
Starter robot.xml:
{(starter/'robot.xml').read_text()}
"""
        return text
