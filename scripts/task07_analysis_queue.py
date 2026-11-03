"""Deferred operator-only Task07 frozen-worker analysis after the primary run."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import swarm_operator_analysis as A

SOURCE = A.P.ROOT / "outputs/task07-short-01"
OUTPUT = A.P.ROOT / "outputs/task07-frozen-worker-analysis-20261102"


class Task07AnalysisQueue(A.Sweep):
    def __init__(self, output, *, stop_at, primary_pid=17212, source=SOURCE,
                 operator_lock=A.GLOBAL_LOCK, poll_seconds=2):
        self.primary_pid, self.source = primary_pid, Path(source).resolve()
        super().__init__(output, stop_at=stop_at, operator_lock=operator_lock,
            runs=((7, self.source),), priority_pids=(primary_pid,), poll_seconds=poll_seconds)
        self.report["primary_pid"] = primary_pid
        self.report["kind"] = "deferred_task07_only_frozen_worker_operator_analysis"
        self.persist()

    def prepare(self, run):
        super().prepare(run)
        for row in run["workers"]:
            cached = self.cache.get((7, row["sha256"]))
            if row["status"] == "pending" and cached is not None:
                row.update(cached, cached=True,
                    status="scored" if cached["score"] is not None else "unscored")
        self.persist()

    def run(self):
        while time.monotonic() < self.deadline:
            eligible, error = A.run_is_eligible(self.source)
            primary_live = A.pid_alive(self.primary_pid)
            if eligible and not primary_live:
                break
            reasons = (["primary_task07_parent_still_running"] if primary_live else [])
            if not eligible:
                reasons.append(error)
            self.runs[0]["waiting_reason"] = "; ".join(reasons)
            self.wait_event(reasons)
            self.persist()
            time.sleep(min(self.poll_seconds, max(0, self.deadline - time.monotonic())))
        result = super().run()
        self.report["finished_at"] = time.time()
        self.persist()
        return result


def notify(report):
    run = report["runs"][0]
    selected = (run.get("selected") or {}).get("score")
    best = (run.get("best_available") or {}).get("score")
    text = (f"Task07 frozen-worker operator analysis: selected score={selected}; best AVAILABLE score={best}; "
        f"coverage={run['scored_workers']}/6 workers; partial={run['partial']}. "
        "Missing/deadline cases remain unscored; this is not a claim that all designs were evaluated. "
        f"No model/API calls. Report: {OUTPUT / 'report.json'}")
    try:
        subprocess.run(["agent-message", "claude", "investigator", text],
                       capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--operator-lock", type=Path, default=A.GLOBAL_LOCK)
    parser.add_argument("--primary-pid", type=int, default=17212)
    parser.add_argument("--stop-at", default="2026-11-02T14:59:30-08:00")
    args = parser.parse_args(argv)
    if args.output.resolve() != OUTPUT.resolve() or args.primary_pid <= 0:
        parser.error("Use the designated Task07 analysis output and a positive primary PID")
    stop = datetime.fromisoformat(args.stop_at)
    cutoff = datetime.now(ZoneInfo("America/Los_Angeles")).replace(hour=15, minute=0, second=0, microsecond=0).timestamp() - 30
    if stop.tzinfo is None or stop.timestamp() > cutoff:
        parser.error("Timezone-aware stop cannot exceed today14:59:30 Pacific")
    command = [str(Path(sys.executable).resolve()), "-B", str(Path(__file__).resolve()),
        "--output", str(args.output.resolve()), "--operator-lock", str(args.operator_lock),
        "--primary-pid", str(args.primary_pid), "--stop-at", args.stop_at]
    A.write(A.SETUP / "task07-frozen-analysis-process.json", {"pid": os.getpid(),
        "command": command, "started_at": time.time()})
    queue = Task07AnalysisQueue(args.output, stop_at=stop.timestamp(), primary_pid=args.primary_pid,
                               operator_lock=args.operator_lock)
    result = queue.run()
    notify(result)
    print(json.dumps({"status": result["status"], "coverage": result["runs"][0]["scored_workers"],
        "partial": result["runs"][0]["partial"], "model_calls": 0,
        "report": str(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
