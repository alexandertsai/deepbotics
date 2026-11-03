"""Private operator analysis after every worker has stopped and selection is frozen."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import task08_pilot as P
import task09_pilot as Q
from short_pilot_runtime import MemoryGatedSimulationLock
from services.controller.storage import verify


def _verify(bundle, manifest):
    if bundle.is_symlink():
        raise ValueError("Frozen bundle must not be a symlink")
    verify(bundle, manifest)
    expected = hashlib.sha256(json.dumps({key: value for key, value in manifest.items()
        if key != "sha256"}, sort_keys=True).encode()).hexdigest()
    if manifest.get("sha256") != expected:
        raise ValueError("Frozen manifest digest is invalid")


def _reward(value):
    reward = value.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Full verifier reward is missing or invalid")
    return float(reward)


def analyze_worker_finals(run_root, sessions, selected_manifest, stop_at, sim_lock_path):
    """Return operator-only results; never call from a live worker tool or prompt."""
    run_root = Path(run_root).resolve()
    sessions = list(sessions)
    if len(sessions) != 6 or len({session.worker_id for session in sessions}) != 6:
        raise ValueError("Operator analysis requires exactly six distinct workers")
    selection = json.loads((run_root / "selection.json").read_text())
    if selection.get("manifest") != selected_manifest:
        raise ValueError("Selected manifest does not match the frozen selection")
    _verify(run_root / "selected", selected_manifest)
    for session in sessions:
        root = Path(session.root)
        if root.is_symlink() or not root.resolve().is_relative_to(run_root):
            raise ValueError("Workers must belong to this run")
        if session.submitted is None or not (root / "agent-summary.json").is_file():
            raise ValueError("All workers must have stopped and frozen before operator analysis")
        if getattr(session, "process", None) is not None and session.process.poll() is None:
            raise ValueError("A worker process is still running")
    zone = ZoneInfo("America/Los_Angeles")
    cutoff = datetime.now(zone).replace(hour=15, minute=0, second=0, microsecond=0).timestamp() - 30
    stop_at = min(float(stop_at), cutoff)
    if not math.isfinite(stop_at):
        raise ValueError("Operator analysis deadline must be finite")
    deadline = time.monotonic() + stop_at - time.time()
    output = run_root / "operator-analysis"
    output.mkdir(mode=0o700, exist_ok=False)
    report = {"kind": "private_operator_analysis_after_all_workers_stopped",
        "selection_used_analysis_results": False, "feedback_available_to_workers": False,
        "started_at": time.time(), "stop_at": stop_at, "memory_admission_gib": 6.5,
        "simulation_lock": str(sim_lock_path), "workers": [], "unique_private_evaluations": 0}
    cache = {}
    selected_final = run_root / "final/reward.json"
    if selected_final.is_file():
        try:
            reward = _reward(json.loads(selected_final.read_text()))
            cache[selected_manifest["sha256"]] = {"reward": reward, "score": reward * 100,
                "reused_from": str(selected_final.relative_to(run_root))}
        except (OSError, ValueError, TypeError) as error:
            report["selected_final_error"] = f"{type(error).__name__}: {error}"

    def persist():
        P.write_json(output / "report.json", report)

    persist()
    lock = MemoryGatedSimulationLock(sim_lock_path, lambda: deadline, output / "simulation-waits.jsonl")
    for index, session in enumerate(sessions):
        manifest = session.submitted
        row = {"worker": session.worker_id, "sha256": manifest.get("sha256"), "score": None, "error": None}
        report["workers"].append(row)
        started = time.monotonic()
        evaluation_started = False
        try:
            bundle = Path(session.root).resolve() / "frozen"
            _verify(bundle, manifest)
            key = manifest["sha256"]
            if key in cache:
                row.update(cache[key], cached=True)
            else:
                with lock:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Operator analysis hard deadline reached")
                    report["unique_private_evaluations"] += 1
                    evaluation = output / "evaluations" / f"{index:02d}"
                    evaluation_started = True
                    reward = _reward(Q.evaluate(bundle, evaluation, final=True, timeout=min(900, remaining)))
                    _verify(bundle, manifest)
                result = {"reward": reward, "score": reward * 100,
                    "reused_from": str((evaluation / "reward.json").relative_to(run_root))}
                cache[key] = result
                row.update(result, cached=False)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"[:2000]
            if evaluation_started:
                cache[key] = {"score": None, "error": row["error"],
                    "reused_from": str(evaluation.relative_to(run_root))}
        row["analysis_seconds"] = time.monotonic() - started
        persist()
    report["finished_at"] = time.time()
    report["status"] = "completed" if all(row["error"] is None for row in report["workers"]) else "completed_with_failures"
    persist()
    return report
