"""Prepared Task09 follow-up with iterative probes and a public multi-metric proxy."""

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
import task09_followup as Q

ROLES = (
    ("spring_tuning", "Explore conservative legal joint spring stiffness and reference pose choices across the public workspace, without changing fixed geometry."),
    ("proximal_springs", "Explore compact legal spring compensation on proximal joints across all three variants. Preserve passive backdrivability and useful workspace."),
    ("passive_balance", "Explore legal passive gravity compensation in lead.xml while preserving buildability, workspace and the mass budget. Keep every variant usable."),
    ("robust_codesign", "Explore conservative hardware/software co-design across all three variants. Preserve a tested incumbent and report public evidence."),
    ("distal_mass", "Explore legal reductions in distal printed-link mass and corresponding model-based trim. Preserve servo, trigger and structural geometry contracts."),
    ("mass_distribution", "Explore compact, legal changes to printed density and mass distribution while preserving stock geometry, continuous structure and the device mass budget."),
    ("servo_saturation", "Explore combined passive assistance and feedforward that reduces servo saturation across the whole public workspace, within legal hardware constraints."),
    ("torque_headroom", "Improve trim behavior near the 0.35 Nm servo limit while retaining passive balance and legal mass across every variant."),
    ("nominal_trim", "Improve nominal gravity trim for all variants using the public model, without hidden parameters or XML-path assumptions."),
    ("adaptive_trim", "Improve plan_probe/adapt and adapted trim using at most 16 hold measurements. Preserve nominal trim and legal torques."),
    ("payload_adaptation", "Improve estimation of unknown handle payload from the 16 permitted hold measurements without assuming the hidden payload or XML path."),
    ("probe_design", "Explore calibration pose selection for identifying gravity errors from the fixed 16-measurement budget, using all three variants."),
    ("workspace_coverage", "Explore gravity compensation across the public workspace rather than only the home pose. Preserve robustness to a nominal model supplied at runtime."),
    ("calibration_regularization", "Explore conservative regularized adaptation to avoid overfitting the small probe set while retaining a usable nominal trim."),
    ("conservative_baseline", "Develop one minimal, conservative improvement to nominal and adapted gravity compensation, preserving all three variants and avoiding unnecessary hardware changes."),
)
ENTRY_FILES = Q.ENTRY_FILES


def hold_score(report):
    variants = report.get("variants", {})
    if set(variants) != set(Q.VARIANTS):
        raise ValueError("Development probe must measure all three variants")
    errors = []
    for row in variants.values():
        if row.get("error") or row.get("adaptation_error"):
            raise ValueError("Public probe returned a variant or adaptation error")
        mass, budget = float(row["mass_kg"]), float(row["mass_budget_kg"])
        if not math.isfinite(mass) or not 0 < mass <= budget or budget != 2.30:
            raise ValueError("Public model exceeds the mass budget or has invalid mass")
        for mode in ("nominal", "adapted"):
            value = float(row[mode]["hold_rms_error_rad"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid public hold error")
            errors.append(value)
    return -sum(errors) / len(errors)


SELECTION_WEIGHTS = {"hold": .35, "passive_residual": .45, "servo_headroom": .20}
SELECTION_SCALES = {"hold_error_rad": .10, "passive_residual_nm": .35}
SELECTION_DESCRIPTION = "public multi-metric proxy: 35% hold quality, 45% passive residual quality, 20% mean servo headroom; not an official score"


def metric_components(report):
    hold_score(report)
    hold, passive, headroom = [], [], []
    for row in report["variants"].values():
        residual = float(row["passive_residual_max_nm"])
        if not math.isfinite(residual) or residual < 0:
            raise ValueError("Invalid public passive residual maximum")
        passive.append(1 / (1 + residual / SELECTION_SCALES["passive_residual_nm"]))
        for mode in ("nominal", "adapted"):
            error = float(row[mode]["hold_rms_error_rad"])
            room = float(row[mode]["servo_headroom_mean"])
            if not math.isfinite(room) or not 0 <= room <= 1:
                raise ValueError("Invalid public servo headroom")
            hold.append(1 / (1 + error / SELECTION_SCALES["hold_error_rad"]))
            headroom.append(room)
    return {"hold": sum(hold) / len(hold), "passive_residual": sum(passive) / len(passive),
            "servo_headroom": sum(headroom) / len(headroom)}


def partial_score(report):
    return sum(SELECTION_WEIGHTS[key] * value for key, value in metric_components(report).items())


def compact_report(report):
    fields = ("mass_kg", "mass_budget_kg", "passive_residual_rms_nm", "passive_residual_max_nm",
              "probe_measurements", "error", "adaptation_error", "nominal", "adapted")
    return {name: {key: row[key] for key in fields if key in row}
            for name, row in report.get("variants", {}).items()}


class Coordinator:
    def __init__(self, root, reference_root, previous_root, seconds=900, tests=3,
                 calls=12, calls_per_worker=3, tokens=250000, output_tokens=8192,
                 cost=2.0, tests_per_worker=None, worker_count=4):
        from services.controller.storage import snapshot, verify
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.reference_root, self.previous_root = reference_root.resolve(), previous_root.resolve()
        original = json.loads((self.reference_root / "manifest.json").read_text())
        verify(self.reference_root / "frozen", original)
        self.manifest = snapshot(self.reference_root / "frozen", self.root / "starter",
                                 max_bytes=256 * 1024**2, workspace_path="code/task09")
        baseline_started = time.monotonic()
        self.baseline_report = Q.evaluate(self.root / "starter", self.root / "starter-probe")
        self.starter_probe_seconds = time.monotonic() - baseline_started
        self.baseline_score = partial_score(self.baseline_report)
        self.max_tests, self.seconds = tests, seconds
        self.worker_count = worker_count
        self.tests_per_worker = tests_per_worker
        self.api_limits = {"max_calls": calls, "per_worker_calls": calls_per_worker,
            "max_tokens": tokens, "max_output_tokens": output_tokens, "max_cost": cost}
        self.tests = 0
        self.deadline = None
        self.client = None
        self.lock, self.simulator = threading.RLock(), threading.Lock()
        self.entries, self.findings, self.rehearsals = [], [], []
        self.inflight, self.probe_errors = {}, {}
        self.cache = {self.manifest["sha256"]: {"score": self.baseline_score, "report": self.baseline_report,
                      "bundle": self.root / "starter", "manifest": self.manifest, "worker": "unchanged_reference"}}
        self.best = dict(self.cache[self.manifest["sha256"]])
        P.write_json(self.root / "starter-manifest.json", self.manifest)

    def start(self):
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + self.seconds

    def persist(self):
        P.write_json(self.root / "probe-ledger.json", self.entries)
        P.write_json(self.root / "findings.json", self.findings)

    def status(self):
        with self.lock:
            probe_summaries = [{key: row[key] for key in ("index", "worker", "status", "partial_score", "selection_components")
                               if key in row} for row in self.entries[-12:]]
            findings = [{**row, "notes": row.get("notes", "")[:1200]} for row in self.findings[-8:]]
            result = {"seconds_remaining": max(0, round(self.deadline - time.monotonic())),
                "new_probes_used": self.tests, "new_probes_remaining": self.max_tests - self.tests,
                "new_probes_per_worker_limit": self.tests_per_worker,
                "partial_score_maximum": 1.0, "selection_metric": SELECTION_DESCRIPTION, "starter_partial_score": self.baseline_score,
                "best_partial_score": self.best["score"], "best_worker": self.best["worker"],
                "starter_selection_components": metric_components(self.baseline_report),
                "best_selection_components": metric_components(self.best["report"]),
                "probes": probe_summaries, "probes_total": len(self.entries),
                "findings": findings, "findings_total": len(self.findings),
                "best_development_metrics": compact_report(self.best["report"])}
        if self.client:
            result["api_budget"] = self.client.summary()
        return result

    def read_probe(self, index):
        with self.lock:
            if type(index) is not int or not 0 <= index < len(self.entries):
                raise ValueError("Use the index of a measured public development probe")
            entry = self.entries[index]
            if entry["status"] != "measured":
                raise ValueError("This probe has no completed measurement")
            return {"index": index, "worker": entry["worker"], "partial_score": entry["partial_score"],
                    "report": self.cache[entry["sha256"]]["report"]}

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
            manifest = snapshot(source, bundle, max_bytes=256 * 1024**2, workspace_path="code/task09")
        P.write_json(session.root / "versions" / f"{version}.json",
                     {"manifest": manifest, "hypothesis": hypothesis})
        with self.lock:
            key = manifest["sha256"]
            if key in self.cache:
                cached = self.cache[key]
                return {"version": version, "cached": True, "partial_score": cached["score"],
                        "report": {"variants": compact_report(cached["report"])}, "new_probe_charged": False}
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
                        "report": {"variants": compact_report(cached["report"])}, "new_probe_charged": False}
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
                probe_started = time.monotonic()
                report = Q.evaluate(bundle, self.root / "probes" / str(index), timeout=min(900, remaining))
                score = partial_score(report)
                result = {"score": score, "report": report, "bundle": bundle,
                          "manifest": manifest, "worker": session.worker_id}
                with self.lock:
                    self.cache[key] = result
                    self.inflight.pop(key).set()
                    if score > self.best["score"] + 1e-10:
                        self.best = result
                    entry.update(status="measured", partial_score=score, selection_components=metric_components(report), report=report.get("variants", {}),
                                 evaluation_seconds=time.monotonic() - probe_started)
                    self.persist()
                print(json.dumps({"event": "probe_completed", "worker": session.worker_id,
                                  "index": index, "partial_score": score}), flush=True)
                return {"version": version, "partial_score": score, "selection_components": metric_components(report), "report": {"variants": compact_report(report)},
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
            entry = {"worker": session.worker_id}
            try:
                if not source.is_dir():
                    raise ValueError("Worker did not produce a frozen candidate")
                if not any((source / name).read_bytes() != (self.root / "starter" / name).read_bytes()
                           for name in ENTRY_FILES):
                    continue
                verify(source, session.submitted)
                for variant in Q.VARIANTS:
                    ET.parse(source / variant / "lead.xml")
                    ast.parse((source / variant / "trim.py").read_text())
                result = self.test(session, "Coordinator rehearsal of the frozen final proposal within remaining shared and worker probe allowances", source=source)
                entry.update(partial_score=result["partial_score"], cached=result.get("cached", False))
            except Exception as error:
                entry["error"] = str(error)
            outcomes.append(entry)
            with self.lock:
                self.rehearsals.append(entry)
                P.write_json(self.root / "coordinator-rehearsals.json", self.rehearsals)
        return outcomes

    def select(self):
        from services.controller.storage import snapshot, verify
        with self.lock:
            chosen = dict(self.best)
        verify(chosen["bundle"], chosen["manifest"])
        manifest = snapshot(chosen["bundle"], self.root / "selected",
                            max_bytes=256 * 1024**2, workspace_path="code/task09")
        P.write_json(self.root / "selected-manifest.json", manifest)
        selection = {"worker": chosen["worker"], "partial_score": chosen["score"],
                     "selection_components": metric_components(chosen["report"]),
                     "starter_partial_score": self.baseline_score,
                     "selection_used_final_scores": False, "manifest": manifest}
        P.write_json(self.root / "selection.json", selection)
        return selection


class SwarmSession(Q.Session):
    mcp_namespace = "task09"

    def __init__(self, root, coordinator, worker_id, role, brief):
        self.coordinator, self.worker_id = coordinator, worker_id
        self.role, self.brief, self.version_count = role, brief, 0
        super().__init__(root, "deepseek-rax", coordinator.seconds, coordinator.max_tests, coordinator.root / "starter")
        for name in ENTRY_FILES:
            (self.candidate / name).chmod(0o644)

    def make_prompt(self):
        text = super().make_prompt().replace("after both agents stop", "after all workers stop")
        files = "\n".join(f"Starter {name}:\n{(self.coordinator.root / 'starter' / name).read_text()}" for name in ENTRY_FILES)
        limits = self.coordinator.api_limits
        logical_limit = getattr(self.coordinator, "logical_calls_per_worker", None)
        allowance = (f"{logical_limit} logical model requests, with at most {limits['per_worker_calls']} provider attempts including retries"
                     if logical_limit else f"{limits['per_worker_calls']} model requests")
        text += f"""
You are worker {self.worker_id} of {self.coordinator.worker_count} independent DeepSeek workers managed by a host coordinator.
Your role is {self.role}: {self.brief}
The workers SHARE {limits['max_calls']} provider requests, a {limits['max_tokens']} reported-token stop threshold,
{self.coordinator.max_tests} NEW development probes, and {self.coordinator.seconds} seconds.
You may use at most {allowance} and {self.coordinator.tests_per_worker or self.coordinator.max_tests} new probes.
Be efficient with tool round trips. All six complete starter entry files are supplied below for exact edits.
Implement a small concrete change and request a test within your first four model requests.
Read shared measured findings once on starting so later worker waves can use earlier spring experiments. Avoid re-reading complete starter files already supplied.
Shared status contains bounded summaries. Use read_probe for one promising measured probe's full public details when needed.
Public helper paths include public/franka/harness/probe_protocol.py, public/franka/harness/torques.py,
and public/franka/harness/scenarios.py. Substitute ur5e or xarm7 for the other variants.
Plan three measured iterations: propose and test, revise from evidence and retest, then reserve the third probe for the final regression check.
Do not submit after one probe while logical requests and probe allowances remain unless you have a concrete blocker. Record each test and use measured findings from peers. Keep legal servo torque limits and the 16-measurement adaptation protocol.
Each worker has its own candidate files. You cannot overwrite a peer. No additional agents may be spawned.
A freshly measured unchanged-starter public probe is provided below; no hidden final results are provided.
Read swarm_status for peer measurements, and share_finding for concise evidence or hypotheses.
The host selects a public proxy averaged across all variants:
35% hold quality + 45% passive residual quality + 20% mean servo headroom.
Hold quality is the average of 1/(1+hold RMS error/0.10 rad) in nominal and adapted modes.
Passive quality is the average of 1/(1+passive residual maximum/0.35 Nm).
Headroom is mean(1-abs(applied servo torque)/0.35 Nm) over public hold steps/joints, then averaged across both modes.
Lower residual maximum and higher headroom matter even after hold error approaches zero. These are transparent proxy scales, not private scoring thresholds.
This public-only proxy is not the official score and omits private robustness, buildability and hardware scoring.
A measured error or mass-budget failure cannot become the incumbent. Starter wins ties or regressions.
Only measured frozen versions can be selected. Your submit call freezes your branch and ends your own turn.
Do not expect untested edits to be submitted automatically as the ensemble winner.
After all workers stop, the coordinator may rehearse untested final candidates using remaining shared time and probe allowances.
The earlier prompt's probe count is the ensemble count, not a separate allowance for your branch.
Starter public development evidence:
{json.dumps(compact_report(self.coordinator.baseline_report), separators=(',', ':'))}
{files}
"""
        return text

    def start(self):
        self.deadline = self.coordinator.deadline

    def definitions(self):
        definitions = super().definitions()
        definitions += [
            {"name": "read_probe", "description": "Fetch one completed public development probe by index. No private verifier feedback is available.",
             "inputSchema": {"type": "object", "properties": {"index": {"type": "integer", "minimum": 0}},
                             "required": ["index"], "additionalProperties": False}},
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
        if name == "read_probe":
            return self.coordinator.read_probe(arguments["index"])
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
    return Q.evaluate(coordinator.root / "selected", coordinator.root / "final", final=True, timeout=3600), None


def run_workers(coordinator, client, sessions, active_workers):
    lifecycle = {s.worker_id: {"status": "queued"} for s in sessions}
    def run(session):
        started = time.monotonic()
        with coordinator.lock:
            lifecycle[session.worker_id] = {"status": "running", "started_at": time.time()}
            P.write_json(coordinator.root / "worker-lifecycle.json", lifecycle)
        print(json.dumps({"event": "worker_started", "worker": session.worker_id}), flush=True)
        try:
            if started >= coordinator.deadline:
                session.submitted = session.freeze(session.root / "frozen")
                P.write_json(session.root / "frozen-manifest.json", session.submitted)
                result = {"worker_id": session.worker_id, "stop_reason": "not_started_before_deadline",
                          "tool_calls": 0, "api_requests": 0, "agent_seconds": 0}
            else:
                result = client.run_worker(session, session.worker_id)
                coordinator.rehearse_remaining([session])
            result["queued_seconds"] = max(0, started - coordinator.started_monotonic)
            return result
        except Exception as error:
            return {"worker_id": session.worker_id, "error": str(error)}
        finally:
            with coordinator.lock:
                lifecycle[session.worker_id].update(status="finished", completed_at=time.time())
                P.write_json(coordinator.root / "worker-lifecycle.json", lifecycle)
    with coordinator.lock:
        P.write_json(coordinator.root / "worker-lifecycle.json", lifecycle)
    with ThreadPoolExecutor(max_workers=active_workers) as pool:
        return list(pool.map(run, sessions))


def main():
    from task08_swarm_client import SharedClient
    import mac_evaluation as M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=P.ROOT / "outputs/task09-swarm-15-followup-01")
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--max-calls", type=int, default=540)
    parser.add_argument("--calls-per-worker", type=int, default=36)
    parser.add_argument("--max-tokens", type=int, default=18000000)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument("--tests", type=int, default=45)
    parser.add_argument("--tests-per-worker", type=int, default=3)
    parser.add_argument("--workers", type=int, choices=(4, 8, 15), default=15)
    parser.add_argument("--api-concurrency", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--active-workers", type=int, default=3)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--logical-calls-per-worker", type=int, default=12)
    parser.add_argument("--prompt-tpm", type=int, default=120000)
    parser.add_argument("--uncached-prompt-tpm", type=int, default=120000)
    parser.add_argument("--output-tpm", type=int, default=30000)
    parser.add_argument("--prompt-bytes-per-token", type=float, default=2.9)
    parser.add_argument("--adaptive-pacing", action="store_true")
    parser.add_argument("--pacing-ramp-seconds", type=float, default=120)
    parser.add_argument("--pacing-ramp-fraction", type=float, default=.10)
    parser.add_argument("--reported-limit-fraction", type=float, default=.80)
    parser.add_argument("--max-prompt-tpm", type=int, default=240000)
    parser.add_argument("--max-uncached-prompt-tpm", type=int, default=240000)
    parser.add_argument("--max-generated-tpm", type=int, default=60000)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 10800:
        parser.error("seconds must be 1..10800")
    if any(v < 1 for v in (args.max_calls, args.calls_per_worker, args.max_tokens, args.tests)):
        parser.error("Request, token and test limits must be positive")
    if args.tests_per_worker is not None and args.tests_per_worker < 1:
        parser.error("tests-per-worker must be positive")
    if args.api_concurrency > args.workers:
        parser.error("api-concurrency cannot exceed workers")
    active_workers = args.active_workers or args.workers
    if not 1 <= active_workers <= args.workers:
        parser.error("active-workers must be between 1 and workers")
    if any(v < 1 for v in (args.logical_calls_per_worker, args.prompt_tpm, args.uncached_prompt_tpm, args.output_tpm)):
        parser.error("Logical request and pacing limits must be positive")
    if not 1 <= args.max_output_tokens <= 32768 or not math.isfinite(args.max_cost) or args.max_cost <= 0:
        parser.error("Output tokens must be 1..32768 and estimated cost must be positive and finite")
    if not math.isfinite(args.prompt_bytes_per_token) or args.prompt_bytes_per_token <= 0:
        parser.error("prompt-bytes-per-token must be positive and finite")
    if args.adaptive_pacing and not args.paced:
        parser.error("adaptive-pacing requires paced")
    if (not math.isfinite(args.pacing_ramp_seconds) or args.pacing_ramp_seconds <= 0
            or not math.isfinite(args.pacing_ramp_fraction) or not 0 < args.pacing_ramp_fraction <= .5
            or not math.isfinite(args.reported_limit_fraction) or not 0 < args.reported_limit_fraction < 1):
        parser.error("Invalid adaptive pacing interval or fraction")
    if any(cap < start for cap, start in ((args.max_prompt_tpm, args.prompt_tpm),
            (args.max_uncached_prompt_tpm, args.uncached_prompt_tpm), (args.max_generated_tpm, args.output_tpm))):
        parser.error("Adaptive pacing maxima cannot be below initial rates")
    checks = Q.preflight()
    if not all(checks.values()):
        raise SystemExit("Preflight failed; no inference requests made: " + json.dumps(checks))
    coordinator = Coordinator(args.output, P.ROOT / "outputs/task09-starter-baseline-01",
                              P.ROOT / "outputs/task09-pilot-01", seconds=args.seconds, tests=args.tests,
                              calls=args.max_calls, calls_per_worker=args.calls_per_worker,
                              tokens=args.max_tokens, output_tokens=args.max_output_tokens,
                              cost=args.max_cost, tests_per_worker=args.tests_per_worker, worker_count=args.workers)
    coordinator.logical_calls_per_worker = args.logical_calls_per_worker if args.paced else None
    roles = ROLES[:args.workers]
    sessions = [SwarmSession(coordinator.root / name, coordinator, name, name, brief) for name, brief in roles]
    protocol = {"task": "Task09", "kind": "multi_worker_RAX_adaptation_public_starter_pilot",
        "workers": [name for name, _ in roles], "seconds_total": args.seconds,
        "max_provider_requests_total": args.max_calls, "max_provider_requests_per_worker": args.calls_per_worker,
        "reported_token_stop_threshold_total": args.max_tokens, "max_output_tokens_per_request": args.max_output_tokens,
        "estimated_spend_guard_usd_total": args.max_cost, "new_development_probe_limit_total": args.tests,
        "new_development_probes_per_worker_limit": args.tests_per_worker,
        "coordinator_rehearsal_of_untested_frozen_candidates": True,
        "simulator_concurrency": 1, "simulator_cpus": 4, "simulator_memory_gb": 4,
        "api_concurrency": args.api_concurrency,
        "active_worker_limit": active_workers,
        "token_pacing_enabled": args.paced,
        "logical_model_requests_per_worker": args.logical_calls_per_worker if args.paced else None,
        "initial_pacing_tpm": {"total_prompt": args.prompt_tpm, "uncached_prompt": args.uncached_prompt_tpm,
                               "output": args.output_tpm} if args.paced else None,
        "max_retries_per_logical_request": 2 if args.paced else 0,
        "pacing_start_stagger_seconds": 2.0 if args.paced else None,
        "parallel_inflight_reservations": args.api_concurrency > 1,
        "post_submit_provider_admission_blocked": True,
        "starter_development_probe": str(coordinator.root / "starter-probe/probe.json"),
        "starter_probe_outside_candidate_probe_allowance": True,
        "selection_metric": SELECTION_DESCRIPTION,
        "selection_weights": SELECTION_WEIGHTS, "selection_scales": SELECTION_SCALES,
        "public_proxy_does_not_cover_private_validity_or_robustness": True,
        "worker_role_order": "spring and passive compensation roles first",
        "coordinator_rehearsal_on_worker_exit": True,
        "bounded_tool_status": {"probe_summaries": 12, "findings": 8, "finding_characters": 1200,
                                "full_measured_probe_available_by_index": True},
        "prompt_bytes_per_token_estimate": args.prompt_bytes_per_token,
        "adaptive_pacing": {"enabled": args.adaptive_pacing, "ramp_interval_seconds": args.pacing_ramp_seconds,
                            "ramp_fraction": args.pacing_ramp_fraction, "reported_limit_fraction": args.reported_limit_fraction,
                            "min_clean_successes": 3, "429_target_backoff_fraction": .25,
                            "maximum_tpm": {"prompt": args.max_prompt_tpm, "uncached": args.max_uncached_prompt_tpm,
                                            "generated": args.max_generated_tpm},
                            "backoff_floor": "initial rates, plus inherited temporary cooldown and throughput reduction"},
        "verifier_image_id": P.checked(["docker", "image", "inspect", Q.IMAGE, "--format", "{{.Id}}"]).stdout.strip(),
        "public_image_id": P.checked(["docker", "image", "inspect", Q.AGENT_IMAGE, "--format", "{{.Id}}"]).stdout.strip(),
        "selection": "best measured partial-development score including unchanged starter; final scores unavailable during selection",
        "comparison_limits": ["Historical single agent hit token stop early and froze untested edits",
            "Selection policy now retains a tested incumbent; changed policy confounds worker-count attribution",
            "Starter public development measurement newly simulated before the development clock",
            "Workers receive full starter files and public development feedback upfront; earlier single read them through tools",
            "Expanded budget differs from historical single",
            "Worker count, API concurrency, per-worker allowances and submit admission policy can differ from earlier swarm runs",
            "Public helper path hints are included; paced conditions add retry recovery and bounded active worker slots",
            "Public selection proxy revised after exploratory prior-run results; not independently validated against private performance",
            "Probe allowance, logical turns, role order and time allowance differ from earlier pilots",
            "Adaptive pacing, when enabled, changes throughput targets based on clean traffic and rate-limit feedback",
            "Parallel requests use conservative in-flight token and estimated-cost reservations; serial runs use between-request token thresholds",
            "CLI host processes are additional overhead outside the simulator container cap",
            "RAX bridge, method instructions and artifact storage adapted; not a full upstream backend"],
        "upstream_commit": P.checked(["git", "rev-parse", "HEAD"], cwd=P.REPO).stdout.strip(),
        "started_at": time.time()}
    P.write_json(coordinator.root / "protocol.json", protocol)
    coordinator.start()
    development_started = time.monotonic()
    coordinator.started_monotonic = development_started
    client_options = dict(deadline=coordinator.deadline, max_concurrent=args.api_concurrency,
                          max_workers=args.workers, **coordinator.api_limits)
    client_class = SharedClient
    if args.paced:
        from task09_paced_client import PacedClient
        client_class = PacedClient
        client_options.update(logical_requests_per_worker=args.logical_calls_per_worker,
                              prompt_tpm=args.prompt_tpm, uncached_prompt_tpm=args.uncached_prompt_tpm,
                              generated_tpm=args.output_tpm, max_retries=2, prompt_bytes_per_token=args.prompt_bytes_per_token)
        if args.adaptive_pacing:
            from task09_adaptive_client import AdaptivePacedClient
            client_class = AdaptivePacedClient
            client_options.update(ramp_interval_seconds=args.pacing_ramp_seconds, ramp_fraction=args.pacing_ramp_fraction,
                                  reported_limit_fraction=args.reported_limit_fraction,
                                  max_prompt_tpm=args.max_prompt_tpm, max_uncached_prompt_tpm=args.max_uncached_prompt_tpm,
                                  max_generated_tpm=args.max_generated_tpm)
    client = client_class(coordinator.root / "shared-api", M.load_keys(), **client_options)
    coordinator.client = client
    try:
        print(json.dumps({"event": "workers_started", "workers": protocol["workers"],
                          "api_concurrency": args.api_concurrency}), flush=True)
        summaries = run_workers(coordinator, client, sessions, active_workers)
        P.write_json(coordinator.root / "worker-summaries.json", summaries)
    finally:
        client.close()
        P.write_json(coordinator.root / "api-summary.json", client.summary())
    coordinator.rehearse_remaining(sessions)
    for session, summary in zip(sessions, summaries):
        summary["development_probes_including_coordinator_rehearsal"] = session.tests
    P.write_json(coordinator.root / "worker-summaries.json", summaries)
    development_seconds = time.monotonic() - development_started
    selection = coordinator.select()
    print(json.dumps({"event": "selection_frozen", "worker": selection["worker"],
                      "partial_score": selection["partial_score"]}), flush=True)
    final_started = time.monotonic()
    reward, reused = final_evaluate(coordinator, selection)
    final_seconds = time.monotonic() - final_started
    reference = json.loads((coordinator.reference_root / "final/reward.json").read_text())["reward"]
    previous = json.loads((coordinator.previous_root / "deepseek-rax/final/reward.json").read_text())["reward"]
    changed_workers = sum(any((session.candidate / name).read_bytes() !=
        (coordinator.root / "starter" / name).read_bytes() for name in ENTRY_FILES)
        for session in sessions)
    result = {"official_reward": reward, "selected_worker": selection["worker"],
        "status": "completed" if changed_workers else "no_new_candidates",
        "workers_with_artifact_changes": changed_workers,
        "development_seconds": development_seconds, "final_verification_seconds": final_seconds,
        "starter_probe_seconds": coordinator.starter_probe_seconds,
        "development_and_final_seconds": development_seconds + final_seconds,
        "total_seconds_including_starter_probe": coordinator.starter_probe_seconds + development_seconds + final_seconds,
        "cost_scope": "DeepSeek API estimate only; local simulation and subscription costs excluded; failed-request usage may be unknown",
        "score": round(reward["reward"] * 100, 2), "unchanged_reference_score": round(reference * 100, 2),
        "historical_single_deepseek_score": round(previous * 100, 2),
        "delta_from_reference": round((reward["reward"] - reference) * 100, 2),
        "delta_from_historical_single": round((reward["reward"] - previous) * 100, 2),
        "new_development_probes": coordinator.tests, "api": client.summary(),
        "workers_with_model_requests": sum(int(row.get("api_requests", 0) > 0) for row in summaries),
        "workers_with_tool_activity": sum(int(row.get("tool_calls", 0) > 0) for row in summaries),
        "final_evaluation_reused_from": reused, "comparison_limits": protocol["comparison_limits"]}
    if selection["worker"] == "unchanged_reference":
        result["interpretation"] = "Public proxy selected the unchanged starter; this is a freshly verified fallback, not a swarm-generated improvement."
    P.write_json(coordinator.root / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
