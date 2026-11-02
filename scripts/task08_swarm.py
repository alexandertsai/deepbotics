"""Four DeepSeek workers sharing a Task 08 search budget and RAX artifacts."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET

import task08_pilot as P

ROLES = (
    ("unfold_timing", "Improve loaded unfolding and settling without losing the 30-second deadline. Keep geometry unchanged unless essential."),
    ("shelf_approach", "Explore base approach and shelf-entry trajectories. Preserve shelf clearance throughout the motion."),
    ("arm_planning", "Improve arm posture and inverse-kinematics plan selection across all three arms. Keep an entire tested controller usable."),
    ("base_stability", "Explore small, legal common-base geometry or battery-placement changes while preserving structural connectivity and the fixed wheels."),
)
WEIGHTS = {
    "S1.load": .03, "S1.inertia": .03, "S1.structure": .03, "S1.mecanum": .03,
    "S1.equilibrium": .03, "S3.still_valid": .03, "S3.resource_constraints": .02,
    "S3.design_efficiency": .22, "S3.reach_preserved": .025, "S3.reach_beyond": .025,
    "S3.battery_onboard": .03, "S3.static_margin": .15, "S5.testB": .04,
    "S7.pick_compat": .10, "S7.payload_margin": .05,
}


def partial_score(report):
    arms = report.get("arms", {})
    if set(arms) != {"panda", "ur5e", "xarm7"}:
        raise ValueError("Development probe did not measure all three arms")
    values = {}
    for key in WEIGHTS:
        rows = [float(arm["checkpoints"][key]) for arm in arms.values()]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in rows):
            raise ValueError("Invalid measured checkpoint")
        values[key] = min(rows)
    score = sum(WEIGHTS[k] * v for k, v in values.items())
    if values["S1.load"] < 1 or values["S1.structure"] < 1:
        score = min(score, .15)
    return score


def compact_report(report):
    return {name: {
        "checkpoints": row.get("checkpoints", {}),
        "shelf": [{k: trial.get(k) for k in ("ok", "reason", "target_name", "payload_kg")}
                  for trial in row.get("shelf", [])],
    } for name, row in report.get("arms", {}).items()}


class Coordinator:
    def __init__(self, root, reference_root, previous_root, seconds=900, tests=3,
                 calls=12, calls_per_worker=3, tokens=250000, output_tokens=8192,
                 cost=2.0, tests_per_worker=None):
        from services.controller.storage import snapshot, verify
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.reference_root, self.previous_root = reference_root.resolve(), previous_root.resolve()
        original = json.loads((self.reference_root / "manifest.json").read_text())
        verify(self.reference_root / "frozen", original)
        self.manifest = snapshot(self.reference_root / "frozen", self.root / "starter",
                                 max_bytes=5 * 1024**2, entrypoint="controller.py", workspace_path="code/task08")
        baseline_manifest = json.loads((self.previous_root / "deepseek-rax/versions/0.json").read_text())["manifest"]
        if self.manifest["files"] != baseline_manifest["files"]:
            raise ValueError("Cached development probe does not match the starter")
        self.baseline_report = json.loads((self.previous_root / "deepseek-rax/probes/0/probe.json").read_text())
        self.baseline_score = partial_score(self.baseline_report)
        self.max_tests, self.seconds = tests, seconds
        self.tests_per_worker = tests_per_worker
        self.api_limits = {"max_calls": calls, "per_worker_calls": calls_per_worker,
            "max_tokens": tokens, "max_output_tokens": output_tokens, "max_cost": cost}
        self.tests = 0
        self.deadline = None
        self.client = None
        self.lock, self.simulator = threading.RLock(), threading.Lock()
        self.entries, self.findings = [], []
        self.inflight, self.probe_errors = {}, {}
        self.cache = {self.manifest["sha256"]: {"score": self.baseline_score, "report": self.baseline_report,
                      "bundle": self.root / "starter", "manifest": self.manifest, "worker": "unchanged_reference"}}
        self.best = dict(self.cache[self.manifest["sha256"]])
        P.write_json(self.root / "starter-manifest.json", self.manifest)

    def start(self):
        self.deadline = time.monotonic() + self.seconds

    def persist(self):
        P.write_json(self.root / "probe-ledger.json", self.entries)
        P.write_json(self.root / "findings.json", self.findings)

    def status(self):
        with self.lock:
            result = {"seconds_remaining": max(0, round(self.deadline - time.monotonic())),
                "new_probes_used": self.tests, "new_probes_remaining": self.max_tests - self.tests,
                "new_probes_per_worker_limit": self.tests_per_worker,
                "partial_score_maximum": sum(WEIGHTS.values()), "starter_partial_score": self.baseline_score,
                "best_partial_score": self.best["score"], "best_worker": self.best["worker"],
                "probes": self.entries, "findings": self.findings}
        if self.client:
            result["api_budget"] = self.client.summary()
        return result

    def test(self, session, hypothesis, source=None):
        if time.monotonic() >= self.deadline:
            raise ValueError("Shared development time has ended")
        version = session.version_count
        session.version_count += 1
        bundle = session.root / "versions" / str(version)
        bundle.parent.mkdir(exist_ok=True)
        if source is None:
            manifest = session.freeze(bundle)
        else:
            from services.controller.storage import snapshot
            manifest = snapshot(source, bundle, max_bytes=5 * 1024**2,
                                entrypoint="controller.py", workspace_path="code/task08")
        P.write_json(session.root / "versions" / f"{version}.json",
                     {"manifest": manifest, "hypothesis": hypothesis})
        with self.lock:
            key = manifest["sha256"]
            if key in self.cache:
                cached = self.cache[key]
                return {"version": version, "cached": True, "partial_score": cached["score"],
                        "report": cached["report"], "new_probe_charged": False}
            pending = self.inflight.get(key)
            if pending is not None:
                owner = False
            elif self.tests_per_worker is not None and session.tests >= self.tests_per_worker:
                raise ValueError("This worker's development probe allowance is exhausted")
            elif self.tests >= self.max_tests:
                raise ValueError("The ensemble's development probe allowance is exhausted")
            else:
                owner = True
                pending = threading.Event()
                self.inflight[key] = pending
                index = self.tests
                self.tests += 1
                session.tests += 1
                entry = {"index": index, "worker": session.worker_id, "version": version,
                         "sha256": key, "hypothesis": str(hypothesis)[:4000], "status": "queued"}
                self.entries.append(entry)
                self.persist()
        if not owner:
            if not pending.wait(max(0, self.deadline - time.monotonic())):
                raise TimeoutError("Shared development time expired waiting for the identical candidate")
            with self.lock:
                if key not in self.cache:
                    raise RuntimeError(self.probe_errors.get(key, "The identical candidate probe failed"))
                cached = self.cache[key]
                return {"version": version, "cached": True, "partial_score": cached["score"],
                        "report": cached["report"], "new_probe_charged": False}
        with self.simulator:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                with self.lock:
                    entry.update(status="time_exhausted")
                    self.probe_errors[key] = "Shared development time expired before simulation"
                    self.inflight.pop(key).set()
                    self.persist()
                raise ValueError("Shared development time expired while this probe was queued")
            with self.lock:
                entry["status"] = "running"
                self.persist()
            print(json.dumps({"event": "probe_started", "worker": session.worker_id, "index": index}), flush=True)
            try:
                report = P.docker_evaluate(bundle, self.root / "probes" / str(index), timeout=min(900, remaining))
                score = partial_score(report)
                result = {"score": score, "report": report, "bundle": bundle,
                          "manifest": manifest, "worker": session.worker_id}
                with self.lock:
                    self.cache[key] = result
                    self.inflight.pop(key).set()
                    if score > self.best["score"] + 1e-10:
                        self.best = result
                    entry.update(status="measured", partial_score=score, report=compact_report(report))
                    self.persist()
                print(json.dumps({"event": "probe_completed", "worker": session.worker_id,
                                  "index": index, "partial_score": score}), flush=True)
                return {"version": version, "partial_score": score, "report": report,
                        "shared_status": self.status()}
            except Exception as error:
                with self.lock:
                    entry.update(status="error", error=str(error))
                    self.probe_errors[key] = str(error)
                    pending = self.inflight.pop(key, None)
                    if pending is not None:
                        pending.set()
                    self.persist()
                raise

    def rehearse_remaining(self, sessions):
        from services.controller.storage import verify
        outcomes = []
        for session in sessions:
            if self.tests >= self.max_tests or time.monotonic() >= self.deadline:
                break
            if self.tests_per_worker is not None and session.tests >= self.tests_per_worker:
                continue
            source = session.root / "frozen"
            if not any((source / name).read_bytes() != (self.root / "starter" / name).read_bytes()
                       for name in ("robot.xml", "controller.py")):
                continue
            entry = {"worker": session.worker_id}
            try:
                verify(source, session.submitted)
                ET.parse(source / "robot.xml")
                ast.parse((source / "controller.py").read_text())
                result = self.test(session, "Coordinator rehearsal of the worker's final candidate using remaining shared budget", source=source)
                entry.update(partial_score=result["partial_score"], cached=result.get("cached", False))
            except Exception as error:
                entry["error"] = str(error)
            outcomes.append(entry)
            P.write_json(self.root / "coordinator-rehearsals.json", outcomes)
        return outcomes

    def select(self):
        from services.controller.storage import snapshot, verify
        with self.lock:
            chosen = dict(self.best)
        verify(chosen["bundle"], chosen["manifest"])
        manifest = snapshot(chosen["bundle"], self.root / "selected",
                            max_bytes=5 * 1024**2, entrypoint="controller.py", workspace_path="code/task08")
        P.write_json(self.root / "selected-manifest.json", manifest)
        selection = {"worker": chosen["worker"], "partial_score": chosen["score"],
                     "starter_partial_score": self.baseline_score,
                     "selection_used_final_scores": False, "manifest": manifest}
        P.write_json(self.root / "selection.json", selection)
        return selection


class SwarmSession(P.Session):
    def __init__(self, root, coordinator, worker_id, role, brief):
        self.coordinator, self.worker_id = coordinator, worker_id
        self.role, self.brief, self.version_count = role, brief, 0
        super().__init__(root, "deepseek-rax", coordinator.seconds, coordinator.max_tests)
        for name in ("robot.xml", "controller.py"):
            shutil.copyfile(coordinator.root / "starter" / name, self.candidate / name)

    def make_prompt(self):
        text = super().make_prompt()
        controller = (self.coordinator.root / "starter/controller.py").read_text()
        robot = (self.coordinator.root / "starter/robot.xml").read_text()
        limits = self.coordinator.api_limits
        text += f"""
You are worker {self.worker_id} of four independent DeepSeek workers managed by a host coordinator.
Your role is {self.role}: {self.brief}
The four workers SHARE {limits['max_calls']} provider requests, a {limits['max_tokens']} reported-token stop threshold,
{self.coordinator.max_tests} NEW development probes, and {self.coordinator.seconds} seconds.
You may use at most {limits['per_worker_calls']} model requests and {self.coordinator.tests_per_worker or self.coordinator.max_tests} new probes.
Be efficient with tool round trips. Both complete starter files are supplied below for exact edits.
Implement a small concrete change and request a test within your first four model requests.
Do not spend the early requests on status checks or re-reading files already present in this prompt.
Record the evidence after a test, then revise if time and probes remain. Keep the 30-second target episode limit.
Each worker has its own candidate files. You cannot overwrite a peer. No additional agents may be spawned.
The public starter development probe from the earlier experiment is reused below; no hidden final results are provided.
Read swarm_status for peer measurements, and share_finding for concise evidence or hypotheses.
The host selects the highest measured partial development score, retaining the starter on a tie or regression.
Only measured frozen versions can be selected. Your submit call freezes your branch and ends your own turn.
Do not expect untested edits to be submitted automatically as the ensemble winner.
After all workers stop, the coordinator may rehearse untested final candidates using remaining shared time and probe allowances.
The earlier prompt's probe count is the ensemble count, not a separate allowance for your branch.
Starter public development evidence:
{json.dumps(compact_report(self.coordinator.baseline_report), separators=(',', ':'))}
Starter controller.py:
{controller}
Starter robot.xml:
{robot}
"""
        return text

    def start(self):
        self.deadline = self.coordinator.deadline

    def definitions(self):
        definitions = super().definitions()
        definitions += [
            {"name": "swarm_status", "description": "Read shared probe measurements, findings and aggregate budgets.",
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"name": "share_finding", "description": "Share a concise observation or hypothesis. Claims are not used as measured scores.",
             "inputSchema": {"type": "object", "properties": {"notes": {"type": "string"}},
                             "required": ["notes"], "additionalProperties": False}},
        ]
        return definitions

    def dispatch(self, name, arguments):
        if name in ("status", "swarm_status"):
            return {**self.coordinator.status(), "worker": self.worker_id, "submitted": self.submitted is not None}
        if self.submitted is not None or time.monotonic() >= self.deadline:
            raise ValueError("This worker's development has ended")
        if name == "test_candidate":
            return self.coordinator.test(self, arguments["hypothesis"])
        if name == "share_finding":
            notes = arguments["notes"]
            if not isinstance(notes, str) or len(notes) > 4000:
                raise ValueError("Use at most 4000 characters")
            with self.coordinator.lock:
                self.coordinator.findings.append({"worker": self.worker_id, "notes": notes,
                                                  "source": "agent_statement_not_verified_measurement"})
                self.coordinator.persist()
            return {"shared": True}
        return super().dispatch(name, arguments)


def final_evaluate(coordinator, selection):
    reference_protocol = json.loads((coordinator.reference_root / "protocol.json").read_text())
    current_image = P.checked(["docker", "image", "inspect", P.IMAGE, "--format", "{{.Id}}"]).stdout.strip()
    same_files = selection["manifest"]["files"] == coordinator.manifest["files"]
    same_image = reference_protocol["verifier_image_id"] == current_image
    if same_files and same_image:
        shutil.copytree(coordinator.reference_root / "final", coordinator.root / "final")
        reward = json.loads((coordinator.root / "final/reward.json").read_text())
        return reward, str(coordinator.reference_root / "final")
    reward = P.docker_evaluate(coordinator.root / "selected", coordinator.root / "final", final=True, timeout=5400)
    return reward, None


def main():
    from task08_swarm_client import SharedClient
    import mac_evaluation as M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=P.ROOT / "outputs/task08-swarm-01")
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--calls-per-worker", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=250000)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--tests", type=int, default=3)
    parser.add_argument("--tests-per-worker", type=int)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600:
        parser.error("seconds must be 1..3600")
    if any(v < 1 for v in (args.max_calls, args.calls_per_worker, args.max_tokens, args.tests)):
        parser.error("Request, token and test limits must be positive")
    if args.tests_per_worker is not None and args.tests_per_worker < 1:
        parser.error("tests-per-worker must be positive")
    if not 1 <= args.max_output_tokens <= 32768 or not math.isfinite(args.max_cost) or args.max_cost <= 0:
        parser.error("Output tokens must be 1..32768 and estimated cost must be positive and finite")
    checks = P.preflight()
    if not all(checks.values()):
        raise SystemExit("Preflight failed; no inference requests made: " + json.dumps(checks))
    coordinator = Coordinator(args.output, P.ROOT / "outputs/task08-reference-baseline-01",
                              P.ROOT / "outputs/task08-pilot-02", seconds=args.seconds, tests=args.tests,
                              calls=args.max_calls, calls_per_worker=args.calls_per_worker,
                              tokens=args.max_tokens, output_tokens=args.max_output_tokens,
                              cost=args.max_cost, tests_per_worker=args.tests_per_worker)
    sessions = [SwarmSession(coordinator.root / name, coordinator, name, name, brief) for name, brief in ROLES]
    protocol = {"task": "Task08", "kind": "four_worker_RAX_adaptation_reference_starter_pilot",
        "workers": [name for name, _ in ROLES], "seconds_total": args.seconds,
        "max_provider_requests_total": args.max_calls, "max_provider_requests_per_worker": args.calls_per_worker,
        "reported_token_stop_threshold_total": args.max_tokens, "max_output_tokens_per_request": args.max_output_tokens,
        "estimated_spend_guard_usd_total": args.max_cost, "new_development_probe_limit_total": args.tests,
        "new_development_probes_per_worker_limit": args.tests_per_worker,
        "coordinator_rehearsal_of_untested_frozen_candidates": True,
        "simulator_concurrency": 1, "simulator_cpus": 4, "simulator_memory_gb": 4,
        "api_concurrency": 1,
        "cached_development_probe": str(coordinator.previous_root / "deepseek-rax/probes/0/probe.json"),
        "selection": "best measured partial-development score including unchanged starter; final scores unavailable during selection",
        "comparison_limits": ["Historical single agent hit token stop early and froze untested edits",
            "Selection policy now retains a tested incumbent; changed policy confounds worker-count attribution",
            "Starter public development measurement reused without another simulation",
            "Workers receive full starter files and cached public development feedback upfront; earlier single read them through tools",
            "Expanded budget differs from historical single and first swarm run",
            "Four CLI host processes are additional overhead outside the simulator container cap",
            "RAX bridge, method instructions and artifact storage adapted; not a full upstream backend"],
        "upstream_commit": P.checked(["git", "rev-parse", "HEAD"], cwd=P.REPO).stdout.strip(),
        "started_at": time.time()}
    P.write_json(coordinator.root / "protocol.json", protocol)
    coordinator.start()
    client = SharedClient(coordinator.root / "shared-api", M.load_keys(), deadline=coordinator.deadline,
                          **coordinator.api_limits)
    coordinator.client = client
    try:
        print(json.dumps({"event": "four_workers_started", "workers": protocol["workers"]}), flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(client.run_worker, session, session.worker_id) for session in sessions]
            summaries = []
            for session, future in zip(sessions, futures):
                try:
                    summaries.append(future.result())
                except Exception as error:
                    summaries.append({"worker": session.worker_id, "error": str(error)})
            P.write_json(coordinator.root / "worker-summaries.json", summaries)
    finally:
        client.close()
        P.write_json(coordinator.root / "api-summary.json", client.summary())
    coordinator.rehearse_remaining(sessions)
    for session, summary in zip(sessions, summaries):
        summary["development_probes_including_coordinator_rehearsal"] = session.tests
    P.write_json(coordinator.root / "worker-summaries.json", summaries)
    selection = coordinator.select()
    print(json.dumps({"event": "selection_frozen", "worker": selection["worker"],
                      "partial_score": selection["partial_score"]}), flush=True)
    reward, reused = final_evaluate(coordinator, selection)
    reference = json.loads((coordinator.reference_root / "final/reward.json").read_text())["reward"]
    previous = json.loads((coordinator.previous_root / "deepseek-rax/final/reward.json").read_text())["reward"]
    changed_workers = sum(any((session.candidate / name).read_bytes() !=
        (coordinator.root / "starter" / name).read_bytes() for name in ("robot.xml", "controller.py"))
        for session in sessions)
    result = {"official_reward": reward, "selected_worker": selection["worker"],
        "status": "completed" if changed_workers else "no_new_candidates",
        "workers_with_artifact_changes": changed_workers,
        "score": round(reward["reward"] * 100, 2), "unchanged_reference_score": round(reference * 100, 2),
        "historical_single_deepseek_score": round(previous * 100, 2),
        "delta_from_reference": round((reward["reward"] - reference) * 100, 2),
        "delta_from_historical_single": round((reward["reward"] - previous) * 100, 2),
        "new_development_probes": coordinator.tests, "api": client.summary(),
        "workers_with_model_requests": sum(int(row.get("api_requests", 0) > 0) for row in summaries),
        "workers_with_tool_activity": sum(int(row.get("tool_calls", 0) > 0) for row in summaries),
        "final_evaluation_reused_from": reused, "comparison_limits": protocol["comparison_limits"]}
    if not changed_workers:
        result["interpretation"] = "No worker changed the starter. The reported score is a reused unchanged-reference fallback, not a swarm-generated design."
    P.write_json(coordinator.root / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
