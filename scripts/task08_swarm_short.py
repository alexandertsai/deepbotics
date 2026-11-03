"""Six-worker paced Task08 pilot with a bounded development window."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as daytime
import json
import fcntl
import math
from pathlib import Path
import shutil
import threading
import time
from zoneinfo import ZoneInfo

import task08_pilot as P
import task08_swarm as S

ROLES = (*S.ROLES,
    ("payload_margin", "Improve payload-dependent reach timing and settled posture. Preserve all three arms and a complete usable controller."),
    ("conservative_route", "Explore a small conservative approach or settling improvement across shelf targets. Preserve legal geometry and a measured incumbent."))


def hard_stop_timestamp(value=None, *, now=None):
    zone = ZoneInfo("America/Los_Angeles")
    current = datetime.now(zone) if now is None else datetime.fromtimestamp(now, zone)
    stop = datetime.fromisoformat(value) if value else datetime.combine(current.date(), daytime(15), zone)
    if stop.tzinfo is None:
        raise ValueError("Hard stop must include a UTC offset")
    return stop.timestamp()


def primary_keys():
    import mac_evaluation as M
    key = M.load_keys().get("FIREWORKS_API_KEY")
    if not key:
        raise ValueError("Primary Fireworks key is not configured")
    return {"FIREWORKS_API_KEY": key}


class Coordinator(S.Coordinator):
    def status(self):
        with self.lock:
            result = {"workers": 6, "active_worker_limit": getattr(self, "active_worker_limit", 3),
                "seconds_remaining": max(0, round(self.deadline - time.monotonic())),
                "new_probes_used": self.tests, "new_probes_remaining": self.max_tests - self.tests,
                "new_probes_per_worker_limit": self.tests_per_worker,
                "starter_partial_score": self.baseline_score, "best_partial_score": self.best["score"],
                "best_worker": self.best["worker"],
                "probes": [{k: row[k] for k in ("index", "worker", "status", "partial_score") if k in row}
                           for row in self.entries],
                "recent_findings": [{"worker": row["worker"], "notes": row["notes"][:600]}
                                    for row in self.findings[-6:]]}
        if self.client:
            budget = self.client.summary()
            result["api_budget"] = {k: budget[k] for k in ("requests", "logical_requests", "reported_tokens",
                "estimated_cost_usd", "active_requests", "limits", "worker_logical_requests") if k in budget}
            result["api_budget"]["limits"] = dict(self.api_limits)
            result["api_budget"]["worker_logical_requests"] = budget.get("worker_logical_requests", {
                worker: count for lane in budget.get("per_account", [])
                for worker, count in lane.get("worker_logical_requests", {}).items()})
        return result

    def test(self, session, hypothesis, source=None):
        started = time.monotonic()
        result = super().test(session, hypothesis, source=source)
        result["report"] = S.compact_report(result["report"])
        result["evaluation_seconds"] = time.monotonic() - started
        return result


class Session(S.SwarmSession):
    def make_prompt(self):
        text = super().make_prompt()
        text = text.replace("of four independent DeepSeek workers", "of 6 independent DeepSeek workers")
        text = text.replace("The four workers SHARE", "The 6 workers SHARE")
        attempts = self.coordinator.api_limits["per_worker_calls"]
        text = text.replace(f"at most {attempts} model requests", f"at most {self.coordinator.logical_calls} logical model requests")
        active = getattr(self.coordinator, "active_worker_limit", 3)
        text += (f"\nThis short run has 6 workers and at most {active} active workers/API calls. "
            f"You have {self.coordinator.logical_calls} logical requests; 429 recovery attempts are handled by the host. "
            "Implement and request the first development test within four logical responses. "
            "Use a second probe for a concrete revision if the shared simulator and time permit. "
            "Avoid status polling and repeated source reads. Submit the most useful frozen candidate before your time ends.\n")
        return text


class SimulationLock:
    def __init__(self, path, deadline):
        self.path, self.deadline = Path(path), deadline
        self.local = threading.local()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a")
        try:
            while time.monotonic() < self.deadline():
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.local.handle = handle
                    return self
                except BlockingIOError:
                    time.sleep(.05)
            raise TimeoutError("Shared simulator lock deadline reached")
        except BaseException:
            handle.close()
            raise

    def __exit__(self, *exc):
        handle = self.local.handle
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        del self.local.handle


def run_workers(coordinator, client, sessions, active_workers=3):
    lifecycle, lock = {}, threading.Lock()
    def run(session):
        started = time.monotonic()
        with lock:
            lifecycle[session.worker_id] = {"status": "running", "started_at": time.time()}
            P.write_json(coordinator.root / "worker-lifecycle.json", lifecycle)
        try:
            if started >= coordinator.deadline:
                with session.lock:
                    session.submitted = session.freeze(session.root / "frozen")
                return {"worker_id": session.worker_id, "api_requests": 0,
                        "stop_reason": "not_started_before_deadline"}
            return client.run_worker(session, session.worker_id)
        except Exception as error:
            return {"worker_id": session.worker_id, "error": str(error)}
        finally:
            with lock:
                lifecycle[session.worker_id].update(status="finished", completed_at=time.time())
                P.write_json(coordinator.root / "worker-lifecycle.json", lifecycle)
    with ThreadPoolExecutor(max_workers=active_workers) as pool:
        return list(pool.map(run, sessions))


def final_evaluate(coordinator, selection, hard_stop, timeout):
    remaining = hard_stop - time.time() - 30
    if remaining <= 0:
        raise TimeoutError("Hard stop cleanup reserve reached before final evaluation")
    reference = json.loads((coordinator.reference_root / "protocol.json").read_text())
    image = P.checked(["docker", "image", "inspect", P.IMAGE, "--format", "{{.Id}}"], timeout=min(10, remaining)).stdout.strip()
    if selection["manifest"]["files"] == coordinator.manifest["files"] and image == reference["verifier_image_id"]:
        shutil.copytree(coordinator.reference_root / "final", coordinator.root / "final")
        return json.loads((coordinator.root / "final/reward.json").read_text()), str(coordinator.reference_root / "final")
    remaining = hard_stop - time.time() - 30
    if remaining <= 0:
        raise TimeoutError("Hard stop cleanup reserve reached before final evaluation")
    simulator = coordinator.simulator
    if isinstance(simulator, SimulationLock):
        final_deadline = time.monotonic() + remaining
        simulator = SimulationLock(simulator.path, lambda: final_deadline)
    with simulator:
        remaining = hard_stop - time.time() - 30
        if remaining <= 0:
            raise TimeoutError("Hard stop cleanup reserve reached before final simulation")
        return P.docker_evaluate(coordinator.root / "selected", coordinator.root / "final", final=True,
                                 timeout=min(timeout, remaining)), None


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=P.ROOT / "outputs/task08-swarm-06-short-01")
    for flag, default in (("seconds", 900), ("tests", 12), ("tests-per-worker", 2), ("workers", 6),
            ("active-workers", 6), ("api-concurrency", 3), ("max-calls", 216), ("calls-per-worker", 36),
            ("logical-calls-per-worker", 12), ("max-tokens", 6000000), ("max-output-tokens", 8192),
            ("prompt-tpm", 500000), ("uncached-prompt-tpm", 250000), ("output-tpm", 60000),
            ("final-timeout", 900)):
        result.add_argument("--" + flag, type=int, default=default)
    result.add_argument("--max-cost", type=float, default=6.0)
    result.add_argument("--prompt-bytes-per-token", type=float, default=2.9)
    result.add_argument("--start-stagger-seconds", type=float, default=2.0)
    result.add_argument("--hard-stop-at")
    result.add_argument("--two-accounts", action="store_true", default=True,
                        help="Legacy flag; all three authorized accounts are used by default")
    result.add_argument("--simulation-lock", type=Path)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    max_active = 6
    if args.workers != 6 or not 1 <= args.active_workers <= max_active or not 1 <= args.api_concurrency <= 3:
        cli.error("Use six workers, at most three API slots per account, and at most three active workers per account")
    integer_limits = (args.seconds, args.tests, args.tests_per_worker, args.max_calls, args.calls_per_worker,
        args.logical_calls_per_worker, args.max_tokens, args.max_output_tokens, args.prompt_tpm,
        args.uncached_prompt_tpm, args.output_tpm, args.final_timeout)
    if any(value < 1 for value in integer_limits) or args.max_output_tokens > 32768 or args.seconds > 900:
        cli.error("Limits must be positive; development is at most900 seconds and output at most32768 tokens")
    cost_cap = 6
    if not math.isfinite(args.max_cost) or not 0 < args.max_cost <= cost_cap:
        cli.error(f"Estimated cost guard must be positive and at most {cost_cap} dollars")
    if not math.isfinite(args.prompt_bytes_per_token) or args.prompt_bytes_per_token <= 0:
        cli.error("Prompt byte/token estimate must be positive and finite")
    if not math.isfinite(args.start_stagger_seconds) or args.start_stagger_seconds < 0:
        cli.error("Start stagger must be finite and nonnegative")
    stop = hard_stop_timestamp(args.hard_stop_at)
    if stop > hard_stop_timestamp():
        cli.error("Hard stop cannot be later than today at15:00 Pacific")
    effective_seconds = min(args.seconds, int(stop - time.time() - 30))
    if effective_seconds < 1:
        cli.error("Today's15:00 Pacific hard stop cleanup reserve has begun")
    keys = primary_keys()
    checks = P.preflight()
    if not all(checks.values()):
        raise SystemExit("Preflight failed; no inference requests made: " + json.dumps(checks))
    coordinator = Coordinator(args.output, P.ROOT / "outputs/task08-reference-baseline-01",
        P.ROOT / "outputs/task08-pilot-02", seconds=effective_seconds, tests=args.tests,
        calls=args.max_calls, calls_per_worker=args.calls_per_worker, tokens=args.max_tokens,
        output_tokens=args.max_output_tokens, cost=args.max_cost, tests_per_worker=args.tests_per_worker)
    coordinator.logical_calls = args.logical_calls_per_worker
    coordinator.active_worker_limit = args.active_workers
    if args.simulation_lock:
        coordinator.simulator = SimulationLock(args.simulation_lock, lambda: coordinator.deadline)
    sessions = [Session(coordinator.root / name, coordinator, name, name, brief) for name, brief in ROLES]
    coordinator.start()
    coordinator.deadline = min(coordinator.deadline, time.monotonic() + max(0, stop - time.time() - 30))
    for session in sessions:
        session.start()
    protocol = {"task": "Task08", "kind": "six_worker_paced_short_RAX_reference_starter_pilot",
        "workers": [name for name, _ in ROLES], "active_workers": args.active_workers,
        "api_concurrency": args.api_concurrency * 3,
        "authorized_accounts": 3,
        "api_concurrency_per_account": args.api_concurrency, "development_seconds": effective_seconds,
        "hard_stop_at": datetime.fromtimestamp(stop, ZoneInfo("America/Los_Angeles")).isoformat(),
        "hard_stop_cleanup_reserve_seconds": 30,
        "credential_routing": "three_accounts_fixed_worker_assignment",
        "simulation_lock": str(args.simulation_lock) if args.simulation_lock else None,
        "logical_calls_per_worker": args.logical_calls_per_worker, "provider_attempt_limits": coordinator.api_limits,
        "new_probe_limit": args.tests, "new_probes_per_worker_limit": args.tests_per_worker,
        "probe_allowance_is_not_guaranteed_completion": True, "simulator_concurrency": 1,
        "simulator_cpus": 4, "simulator_memory_gb": 4, "final_timeout_seconds": args.final_timeout,
        "pacing_tpm": {"prompt": args.prompt_tpm, "uncached_prompt": args.uncached_prompt_tpm, "generated": args.output_tpm},
        "prompt_bytes_per_token": args.prompt_bytes_per_token,
        "selection": "highest measured public partial score, starter retained on tie; no final scores during selection",
        "comparison_limits": ["Six workers and expanded allowances differ from historical single-agent conditions",
            "CLI processes add host overhead outside the simulator container cap",
            "Cached starter development measurement and full starter source are supplied upfront"],
        "started_at": time.time()}
    P.write_json(coordinator.root / "protocol.json", protocol)
    from short_pilot_pool import AccountPool as Client
    extra_options = {"worker_ids": [session.worker_id for session in sessions]}
    client = Client(coordinator.root / "shared-api", keys, deadline=coordinator.deadline,
        max_workers=6, max_concurrent=args.api_concurrency, logical_requests_per_worker=args.logical_calls_per_worker,
        prompt_tpm=args.prompt_tpm, uncached_prompt_tpm=args.uncached_prompt_tpm, generated_tpm=args.output_tpm,
        prompt_bytes_per_token=args.prompt_bytes_per_token, start_stagger_seconds=args.start_stagger_seconds,
        **coordinator.api_limits, **extra_options)
    coordinator.client = client
    started = time.monotonic()
    try:
        summaries = run_workers(coordinator, client, sessions, args.active_workers)
        P.write_json(coordinator.root / "worker-summaries.json", summaries)
    finally:
        client.close()
        P.write_json(coordinator.root / "api-summary.json", client.summary())
    coordinator.rehearse_remaining(sessions)
    development_seconds = time.monotonic() - started
    selection = coordinator.select()
    final_started = time.monotonic()
    reward, reused, final_error = None, None, None
    try:
        reward, reused = final_evaluate(coordinator, selection, stop, args.final_timeout)
    except Exception as error:
        final_error = str(error)
    changed = sum(any((session.candidate / name).read_bytes() != (coordinator.root / "starter" / name).read_bytes()
                      for name in ("robot.xml", "controller.py")) for session in sessions)
    result = {"task": "Task08", "status": "completed" if reward is not None else "development_completed_final_unavailable",
        "workers": 6, "workers_with_artifact_changes": changed, "selected_worker": selection["worker"],
        "selected_partial_score": selection["partial_score"], "official_reward": reward,
        "score": round(reward["reward"] * 100, 2) if reward is not None else None,
        "final_evaluation_reused_from": reused, "final_error": final_error,
        "new_development_probes": coordinator.tests, "api": client.summary(),
        "development_seconds": development_seconds, "final_verification_seconds": time.monotonic() - final_started,
        "total_seconds": time.monotonic() - started, "hard_stop_at": protocol["hard_stop_at"],
        "comparison_limits": protocol["comparison_limits"]}
    if not changed:
        result["interpretation"] = "No new candidate artifacts; any reused reference reward is not a swarm improvement."
    P.write_json(coordinator.root / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
