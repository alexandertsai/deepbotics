"""Local private scoring of completed swarms; never exposes feedback to workers."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

import task08_pilot as P
import task09_pilot as Q
import task07_adapter as T
from short_pilot_runtime import MemoryGatedSimulationLock
from services.controller.storage import snapshot, verify

OUTPUT = P.ROOT / "outputs/swarm-operator-analysis-20261102"
SETUP = P.ROOT / "outputs/short-pilots-20261102"
GLOBAL_LOCK = Path("/tmp/kimirax-swarm-operator-analysis.lock")
PRIORITY_PIDS = (30911, 17212)
TASK07_WORKERS = ("depth_targeting", "magnet_contact", "safe_transit", "singulation", "nest_release", "conservative_cycle")
RUNS = ((9, P.ROOT / "outputs/task09-swarm-06-short-01"),
        (8, P.ROOT / "outputs/task08-swarm-06-short-02"),
        (7, P.ROOT / "outputs/task07-short-01"))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    P.write_json(temporary, value)
    temporary.replace(path)


def manifest_verify(bundle, manifest):
    if bundle.is_symlink():
        raise ValueError("Frozen directory cannot be a symlink")
    verify(bundle, manifest)
    digest = hashlib.sha256(json.dumps({key: value for key, value in manifest.items()
        if key != "sha256"}, sort_keys=True).encode()).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError("Invalid manifest digest")


def reward_value(value):
    reward = value.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("No complete valid reward")
    return float(reward)


def registered_runs():
    rows = []
    for path in SETUP.glob("*-process.json"):
        try:
            value = read(path)
            command = value.get("command", [])
            if "--output" not in command:
                continue
            output = Path(command[command.index("--output") + 1])
            if not output.is_absolute():
                output = P.ROOT / output
            rows.append({"pid": int(value["pid"]), "output": output.resolve()})
        except (OSError, ValueError, TypeError, IndexError):
            continue
    return rows


def pid_alive(pid):
    completed = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                               capture_output=True, text=True, timeout=3, check=False)
    status = completed.stdout.strip()
    return bool(status and not status.startswith("Z"))


def run_is_eligible(root):
    if not (root / "summary.json").is_file():
        return False, "Source summary is not complete"
    try:
        if not isinstance(read(root / "summary.json"), dict):
            raise ValueError("Invalid summary")
    except (OSError, ValueError):
        return False, "Source summary is not valid JSON"
    parents = [row for row in registered_runs() if row["output"] == root.resolve()]
    if not parents:
        return False, "Source parent registration is missing"
    if any(pid_alive(row["pid"]) for row in parents):
        return False, "Source parent is still running"
    return True, None


def runtime_blockers(task, priority_pids=PRIORITY_PIDS):
    rows = registered_runs()
    reasons = []
    registered_pids = {row["pid"] for row in rows}
    for row in rows:
        match = re.search(r"task0?([789])", row["output"].name)
        if match and int(match[1]) == task and row["pid"] != os.getpid() and pid_alive(row["pid"]):
            reasons.append(f"unfinished_live_parent:{row['pid']}")
    for pid in priority_pids:
        known_task = {30911: 9, 17212: 7}.get(pid)
        if known_task == task and pid != os.getpid() and pid not in registered_pids and pid_alive(pid):
            reasons.append(f"known_live_parent:{pid}")
    try:
        completed = subprocess.run(["docker", "ps", "--format", "{{json .}}"],
            check=True, capture_output=True, text=True, timeout=5)
        containers = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        if len(containers) >= 3:
            reasons.append(f"container_admission_limit:{len(containers)}")
        for container in containers:
            if re.search(rf"task0?{task}(?:[^0-9]|$)", str(container)):
                reasons.append("same_task_live_container:" + str(container.get("Names", "unknown")))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        reasons.append("container_inventory_unavailable:" + type(error).__name__)
    return reasons


def evaluate(task, bundle, output, remaining):
    if remaining <= 0:
        raise TimeoutError("Operator hard deadline reached")
    if task == 8:
        return P.docker_evaluate(bundle, output, final=True, timeout=min(900, remaining))
    if task == 9:
        return Q.evaluate(bundle, output, final=True, timeout=min(900, remaining))
    if task == 7:
        return T.evaluate(bundle, output, final=True, timeout=min(3600, remaining))
    raise ValueError("Unsupported task")


class Sweep:
    def __init__(self, output, *, stop_at, operator_lock=GLOBAL_LOCK, runs=RUNS,
                 priority_pids=PRIORITY_PIDS, poll_seconds=2):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.stop_at, self.operator_lock = float(stop_at), Path(operator_lock)
        self.deadline = time.monotonic() + self.stop_at - time.time()
        self.priority_pids, self.poll_seconds = tuple(priority_pids), poll_seconds
        self.runs, self.cache = [], {}
        self.report = {"kind": "free_operator_only_completed_swarm_analysis", "model_calls": 0,
            "local_simulation_compute_only": True, "selection_was_fixed_before_analysis": True,
            "feedback_available_to_workers": False, "started_at": time.time(), "stop_at": stop_at,
            "operator_lock": str(operator_lock), "memory_admission_gib": 6.5, "runs": self.runs}
        for task, root in runs:
            root = Path(root).resolve()
            names, mapping_error = [], None
            try:
                names = list(TASK07_WORKERS) if task == 7 else read(root / "protocol.json")["workers"]
                if not isinstance(names, list) or len(names) != 6 or len(set(names)) != 6:
                    raise ValueError("Expected six distinct protocol workers")
                if any(not isinstance(name, str) or P.safe_relative(name).parts != (name,) for name in names):
                    raise ValueError("Invalid worker role mapping")
            except (OSError, ValueError, KeyError, TypeError) as error:
                mapping_error = f"{type(error).__name__}: {error}"
            self.runs.append({"task": task, "source": str(root), "source_state": "waiting",
                "mapping_error": mapping_error, "workers": [{"worker": name, "role": name,
                    "sha256": None, "score": None, "error": None, "status": "waiting_source"} for name in names],
                "selected": None, "best_available": None, "comparison_scope": "best AVAILABLE; analysis may be partial"})
        self.persist()

    def persist(self):
        for run in self.runs:
            available = [row for row in run["workers"] if row["score"] is not None]
            run["best_available"] = max(available, key=lambda row: row["score"]) if available else None
            run["scored_workers"] = len(available)
            run["partial"] = bool(run["mapping_error"] or len(available) < 6)
            selected = run["selected"]
            run["selected_vs_best_available"] = None if selected is None or selected.get("score") is None or not available else {
                "selected_score": selected["score"], "best_available_worker": run["best_available"]["worker"],
                "best_available_score": run["best_available"]["score"],
                "gap_best_available_minus_selected": run["best_available"]["score"] - selected["score"],
                "partial": run["partial"]}
        write(self.output / "report.json", self.report)

    def wait_event(self, reasons):
        path = self.output / "priority-waits.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps({"at": time.time(), "blockers": reasons}) + "\n")

    def prepare(self, run):
        root, task = Path(run["source"]), run["task"]
        condition = root / "deepseek" if task == 7 else root
        destination = self.output / root.name
        destination.mkdir()
        protocol = read(root / "protocol.json")
        run["task_simulation_lock"] = protocol.get("simulation_lock") or str(Path(f"/tmp/kimirax-short-task{task:02d}-sim.lock"))
        try:
            selection = read(condition / "selection.json")
            manifest = selection["manifest"]
            manifest_verify(condition / "selected", manifest)
            selected = {"worker": selection["worker"], "sha256": manifest["sha256"], "score": None, "error": None}
            run["selected"] = selected
            reward = reward_value(read(condition / "final/reward.json"))
            result = {"score": reward * 100, "reward": reward, "error": None,
                "evaluation_source": str(condition / "final/reward.json"), "reused_complete_final": True}
            self.cache[(task, manifest["sha256"])] = result
            selected.update(result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            if run["selected"] is None:
                run["selected"] = {"score": None, "error": f"{type(error).__name__}: {error}"}
            else:
                run["selected"]["error"] = f"{type(error).__name__}: {error}"
        for row in run["workers"]:
            worker_root = condition / row["worker"]
            try:
                manifest = read(worker_root / "frozen-manifest.json")
                frozen = worker_root / "frozen"
                manifest_verify(frozen, manifest)
                target = destination / "bundles" / row["worker"]
                target.parent.mkdir(exist_ok=True)
                copied = snapshot(frozen, target, max_bytes=256 * 1024**2,
                                  workspace_path=manifest.get("workspace_path"))
                if copied != manifest:
                    raise ValueError("Copied bundle manifest differs")
                row.update(sha256=manifest["sha256"], bundle=str(target.relative_to(self.output)), status="pending")
                write(destination / f"{row['worker']}-manifest.json", copied)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status="unscored", error=f"{type(error).__name__}: {error}")
        run["source_state"] = "finished_parent_stopped_artifacts_copied"
        self.persist()

    def score_row(self, run, row):
        key = (run["task"], row["sha256"])
        if key in self.cache:
            row.update(self.cache[key], status="scored" if self.cache[key]["score"] is not None else "unscored", cached=True)
            return True
        reasons = runtime_blockers(run["task"], self.priority_pids)
        if reasons:
            self.wait_event(reasons)
            return False
        lock = MemoryGatedSimulationLock(self.operator_lock, lambda: self.deadline, self.output / "simulation-waits.jsonl")
        started = time.monotonic()
        attempted = False
        task_handle = None
        try:
            with lock:
                reasons = runtime_blockers(run["task"], self.priority_pids)
                if reasons:
                    self.wait_event(reasons)
                    return False
                task_lock = Path(run["task_simulation_lock"])
                task_lock.parent.mkdir(parents=True, exist_ok=True)
                task_handle = task_lock.open("a")
                try:
                    fcntl.flock(task_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.wait_event(["source_task_lock_busy:" + str(task_lock)])
                    return False
                reasons = runtime_blockers(run["task"], self.priority_pids)
                if reasons:
                    self.wait_event(reasons)
                    return False
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Operator hard deadline reached")
                bundle = self.output / row["bundle"]
                manifest = read(self.output / Path(run["source"]).name / f"{row['worker']}-manifest.json")
                manifest_verify(bundle, manifest)
                output = self.output / Path(run["source"]).name / "evaluations" / row["sha256"]
                attempted = True
                reward = reward_value(evaluate(run["task"], bundle, output, remaining))
                manifest_verify(bundle, manifest)
                result = {"score": reward * 100, "reward": reward, "error": None,
                    "evaluation_source": str((output / "reward.json").relative_to(self.output)),
                    "reused_complete_final": False}
                self.cache[key] = result
                row.update(result, status="scored", cached=False)
        except Exception as error:
            result = {"score": None, "error": f"{type(error).__name__}: {error}"[:2000]}
            row.update(result, status="unscored")
            if attempted:
                self.cache[key] = result
        finally:
            if task_handle is not None:
                fcntl.flock(task_handle, fcntl.LOCK_UN)
                task_handle.close()
        row["operator_seconds"] = time.monotonic() - started
        return True

    def run(self):
        while time.monotonic() < self.deadline:
            pending = False
            made_progress = False
            recheck_task_priority = False
            for run in self.runs:
                if run["mapping_error"]:
                    continue
                if run["source_state"] == "waiting":
                    eligible, error = run_is_eligible(Path(run["source"]))
                    if not eligible:
                        run["waiting_reason"] = error
                        pending = True
                        continue
                    try:
                        self.prepare(run)
                    except Exception as error:
                        run["mapping_error"] = f"{type(error).__name__}: {error}"[:2000]
                        for row in run["workers"]:
                            row.update(status="unscored", error=run["mapping_error"])
                        continue
                    made_progress = True
                for row in run["workers"]:
                    if row["status"] != "pending":
                        continue
                    pending = True
                    if time.monotonic() >= self.deadline:
                        break
                    if self.score_row(run, row):
                        made_progress = True
                        self.persist()
                        recheck_task_priority = True
                        break
                    else:
                        break
                if recheck_task_priority:
                    break
            self.persist()
            if not pending:
                break
            if not made_progress:
                time.sleep(min(self.poll_seconds, max(0, self.deadline - time.monotonic())))
        for run in self.runs:
            for row in run["workers"]:
                if row["status"] in ("pending", "waiting_source"):
                    row.update(status="unscored", error="Hard deadline reached before eligible private evaluation")
        self.report.update(finished_at=time.time(), status="completed" if all(not run["partial"] for run in self.runs) else "partial_best_available")
        self.persist()
        return self.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--operator-lock", type=Path, default=GLOBAL_LOCK)
    parser.add_argument("--stop-at", default="2026-11-02T14:59:30-08:00")
    args = parser.parse_args(argv)
    if args.output.resolve() != OUTPUT.resolve():
        parser.error("Output must be outputs/swarm-operator-analysis-20261102/")
    cutoff = datetime.now(ZoneInfo("America/Los_Angeles")).replace(hour=15, minute=0, second=0, microsecond=0).timestamp() - 30
    stop = datetime.fromisoformat(args.stop_at)
    if stop.tzinfo is None or stop.timestamp() > cutoff:
        parser.error("Stop timestamp must include a timezone and cannot exceed14:59:30 Pacific today")
    report = Sweep(args.output, stop_at=stop.timestamp(), operator_lock=args.operator_lock).run()
    print(json.dumps({"status": report["status"], "model_calls": 0,
                      "report": str(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
