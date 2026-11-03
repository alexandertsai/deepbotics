"""Six DeepSeek branches and one fresh Astra branch on public Task07 episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from zoneinfo import ZoneInfo

import task07_adapter as Q
import task08_pilot as P
from task08_swarm_short import SimulationLock, primary_keys

ROLES = (
    ("depth_targeting", "Use public RGB-D calibration to choose a reachable exposed steel surface."),
    ("magnet_contact", "Build reliable magnet approach, descent, contact and lift behavior."),
    ("safe_transit", "Build a safe pick and transit route avoiding the bin rim and walls."),
    ("singulation", "Use force sensing and motion to separate attached brackets before placement."),
    ("nest_release", "Prioritize a complete pick, nest placement and reliable release cycle."),
    ("conservative_cycle", "Build the simplest complete depth-guided pick-and-place cycle with recovery."),
)


def safe_artifact(base, value):
    base = base.resolve()
    path = P.safe_relative(value)
    if path.suffix not in (".py", ".json", ".txt"):
        raise ValueError("Candidate files must be Python, JSON or text")
    target = base / path
    descendants = [target]
    parent = target.parent
    while parent != base and parent != parent.parent:
        descendants.append(parent)
        parent = parent.parent
    if not target.resolve().is_relative_to(base) or any(p.is_symlink() for p in descendants):
        raise ValueError("Unsafe candidate path")
    return target


def freeze(source, destination):
    from services.controller.storage import snapshot
    if not (source / "policy.py").is_file():
        raise ValueError("policy.py is required")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Candidate symlinks are forbidden")
    return snapshot(source.resolve(), destination.resolve(), max_bytes=16 * 1024**2,
                    entrypoint="policy.py", workspace_path="code/task07")


def agent_validity(sessions):
    return {session.worker_id: {"tool_calls": len(session.history),
        "changed": any(path.read_bytes() != (session.coordinator.root / "starter" / path.relative_to(session.candidate)).read_bytes()
            if (session.coordinator.root / "starter" / path.relative_to(session.candidate)).is_file() else True
            for path in session.candidate.rglob("*") if path.is_file())}
        for session in sessions}


class Coordinator:
    def __init__(self, root, starter, starter_report, *, seconds, tests_per_worker, simulator, stop_at):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = freeze(starter, self.root / "starter")
        self.seconds, self.tests_per_worker = seconds, tests_per_worker
        self.baseline_report = starter_report
        self.tests, self.entries, self.findings = 0, [], []
        self.lock, self.simulator = threading.RLock(), simulator
        self.deadline, self.stop_at, self.client = None, stop_at, None
        self.best = {"score": Q.partial_score(starter_report), "report": starter_report,
                     "bundle": self.root / "starter", "manifest": self.manifest, "worker": "starter"}
        self.cache = {self.manifest["sha256"]: self.best}

    def status(self):
        with self.lock:
            return {"seconds_remaining": max(0, round(self.deadline - time.monotonic())),
                "probes_per_worker": self.tests_per_worker, "new_probes_used": self.tests,
                "best_public_proxy": self.best["score"], "best_worker": self.best["worker"],
                "probe_ledger": self.entries, "findings": self.findings[-6:]}

    def persist(self):
        P.write_json(self.root / "probe-ledger.json", self.entries)
        P.write_json(self.root / "findings.json", self.findings)

    def test(self, session, hypothesis):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Development has ended")
        index = session.version_count
        session.version_count += 1
        bundle = session.root / "versions" / str(index)
        bundle.parent.mkdir(exist_ok=True)
        manifest = freeze(session.candidate, bundle)
        P.write_json(session.root / "versions" / f"{index}.json", manifest)
        with self.lock:
            cached = self.cache.get(manifest["sha256"])
            if cached:
                return {"cached": True, "public_proxy": cached["score"], "report": cached["report"]}
            if session.tests >= self.tests_per_worker:
                raise ValueError("Worker public probe allowance exhausted")
            session.tests += 1
            probe_index = self.tests
            self.tests += 1
            entry = {"index": probe_index, "worker": session.worker_id, "version": index,
                "sha256": manifest["sha256"], "hypothesis": str(hypothesis)[:2000], "status": "queued"}
            self.entries.append(entry)
            self.persist()
        started = time.monotonic()
        try:
            with self.simulator:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Development ended while queued for public simulation")
                with self.lock:
                    entry["status"] = "running"
                    self.persist()
                report = Q.evaluate(bundle, self.root / "probes" / str(probe_index),
                                    timeout=remaining, stop_at=self.stop_at - 30)
            score = Q.partial_score(report)
            result = {"score": score, "report": report, "bundle": bundle,
                      "manifest": manifest, "worker": session.worker_id}
            with self.lock:
                self.cache[manifest["sha256"]] = result
                if score > self.best["score"] + 1e-10:
                    self.best = result
                entry.update(status="measured", public_proxy=score,
                             evaluation_seconds=time.monotonic() - started)
                self.persist()
            return {"public_proxy": score, "report": report, "shared_status": self.status()}
        except Exception as error:
            with self.lock:
                entry.update(status="error", error=str(error), elapsed_seconds=time.monotonic() - started)
                self.persist()
            raise

    def select(self):
        from services.controller.storage import verify
        verify(self.best["bundle"], self.best["manifest"])
        manifest = freeze(self.best["bundle"], self.root / "selected")
        selection = {"worker": self.best["worker"], "public_proxy": self.best["score"],
                     "manifest": manifest, "selection_used_final_scores": False}
        P.write_json(self.root / "selection.json", selection)
        return selection


class Session(P.Session):
    mcp_namespace = "task07"

    def __init__(self, root, coordinator, worker_id, brief, *, astra=False):
        self.root, self.coordinator = root.resolve(), coordinator
        self.root.mkdir(parents=True)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.candidate = self.root / "candidate"
        shutil.copytree(coordinator.root / "starter", self.candidate)
        for path in self.candidate.rglob("*"):
            if path.is_file():
                path.chmod(0o644)
        self.worker_id, self.brief = worker_id, brief
        self.condition = "astra" if astra else "deepseek-rax"
        self.tests, self.version_count, self.history = 0, 0, []
        self.submitted, self.deadline = None, None
        self.lock = threading.RLock()
        self.prompt = self.make_prompt()
        (self.root / "prompt.txt").write_text(self.prompt)

    def start(self):
        self.deadline = self.coordinator.deadline

    def make_prompt(self):
        prompt = f"""Task07 shortened public-starter pilot, not a leaderboard replication.
You have {self.coordinator.seconds} shared wall seconds and {self.coordinator.tests_per_worker} new public probes for your own branch.
Use only task07 MCP tools. No shell, external filesystem, network, apps or additional agents.
All workers start from the same deliberately simple hold-position policy below. Author a self-contained CPU policy.
Your role: {self.brief}
Only candidate/ files are editable. read_file accepts public/ or candidate/. Public simulator source is available.
test_candidate freezes the policy and runs published seeds {Q.PUBLIC_SEEDS}, {Q.PUBLIC_SECONDS:g} simulated seconds each.
These short episodes cannot establish full-task success. Rendering and physics may make probes slow; request the first within four responses.
DeepSeek workers have at most 12 logical model responses. Use a second changed version if time permits. Avoid repeated status polling.
Each worker has a separate branch. share_finding records an unverified statement. Peer measured reports are visible in status.
Selection uses only measured frozen public versions: mean(clear_frac - .03*floor_drops - .05*damage_events - .05*bin_hits).
This proxy is not the official reward. The unchanged starter wins ties or regressions. Untested edits cannot become the winner.
submit freezes your own branch. All branches freeze before any hidden final evaluation. No hidden scores or seeds are available.
No emojis. Keep code comments minimal. Report observed facts only.
Starter policy.py:\n{Q.ZERO_POLICY}
Starter public evidence:\n{json.dumps(self.coordinator.baseline_report, separators=(',', ':'))}
Official instructions follow; their longer budget is replaced by this pilot:\n{(Q.TASK / 'instruction.md').read_text()}
"""
        if self.condition == "deepseek-rax":
            from jinja2 import Environment, StrictUndefined
            renderer = Environment(undefined=StrictUndefined)
            prompt += "\nRAX adaptation: form explicit hypotheses, test frozen versions, save cumulative evidence, and preserve a measured incumbent.\n"
            for name in ("evidence-and-regression", "change-discipline"):
                text = (P.ROOT / "rax/agent_skills/method" / name / "SKILL.md").read_text()
                prompt += renderer.from_string(text.split("---", 2)[-1]).render(agent_formal=False)
            prompt += "\nTool reports and version manifests replace archive paths; memory_write/read stores persistent notes.\n"
        return prompt

    def definitions(self):
        def definition(name, description, properties=None, required=()):
            return {"name": name, "description": description, "inputSchema": {"type": "object",
                "properties": properties or {}, "required": list(required), "additionalProperties": False}}
        string = {"type": "string"}
        tools = [definition("list_files", "List public inputs and branch artifacts."),
            definition("read_file", "Read a public/ or candidate/ text file.",
                {"path": string, "offset": {"type": "integer", "minimum": 0},
                 "length": {"type": "integer", "minimum": 1, "maximum": 60000}}, ("path",)),
            definition("write_file", "Replace a relative candidate file, usually policy.py.", {"name": string, "content": string}, ("name", "content")),
            definition("edit_file", "Replace one unique exact string in a candidate file.", {"name": string, "old": string, "new": string}, ("name", "old", "new")),
            definition("test_candidate", "Freeze and run one bounded public probe.", {"hypothesis": string}, ("hypothesis",)),
            definition("status", "Read time, measured peer results and probe allowances."),
            definition("share_finding", "Share a short observation or hypothesis; not a measured result.", {"notes": string}, ("notes",)),
            definition("submit", "Freeze this branch and stop development.")]
        if self.condition == "deepseek-rax":
            tools += [definition("memory_write", "Save cumulative evidence notes.", {"notes": string}, ("notes",)),
                      definition("memory_read", "Read saved evidence notes.")]
        return tools

    def freeze(self, destination):
        return freeze(self.candidate, destination)

    def dispatch(self, name, arguments):
        if name == "status":
            return {**self.coordinator.status(), "worker": self.worker_id, "own_probes_used": self.tests,
                    "own_probes_remaining": self.coordinator.tests_per_worker - self.tests,
                    "submitted": self.submitted is not None}
        if self.submitted is not None or time.monotonic() >= self.deadline:
            raise ValueError("Development has ended")
        if name == "list_files":
            return {"candidate": [str(p.relative_to(self.candidate)) for p in self.candidate.rglob("*") if p.is_file()],
                "public": [str(p.relative_to(Q.PUBLIC)) for p in Q.PUBLIC.rglob("*") if p.is_file()]}
        if name == "read_file":
            path = P.safe_relative(arguments["path"])
            base = {"public": Q.PUBLIC, "candidate": self.candidate}.get(path.parts[0])
            if base is None or len(path.parts) < 2:
                raise ValueError("Use public/ or candidate/ paths")
            target = base.joinpath(*path.parts[1:])
            if target.is_symlink() or not target.resolve().is_relative_to(base.resolve()):
                raise ValueError("Unsafe read path")
            if target.suffix not in (".py", ".xml", ".json", ".txt", ".md"):
                raise ValueError("Only public text assets may be read")
            offset, length = arguments.get("offset", 0), arguments.get("length", 30000)
            if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 60000:
                raise ValueError("Invalid text range")
            text = target.read_text()
            return {"text": text[offset:offset+length], "total_characters": len(text)}
        if name in ("write_file", "edit_file"):
            target = safe_artifact(self.candidate, arguments["name"])
            content = arguments.get("content")
            if name == "edit_file":
                original, old = target.read_text(), arguments["old"]
                if not old or original.count(old) != 1:
                    raise ValueError("old must match exactly once")
                content = original.replace(old, arguments["new"], 1)
            if not isinstance(content, str) or len(content.encode()) > 2 * 1024**2:
                raise ValueError("Candidate file exceeds size limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return {"saved": True}
        if name == "test_candidate":
            return self.coordinator.test(self, arguments["hypothesis"])
        if name == "share_finding":
            notes = arguments["notes"]
            if not isinstance(notes, str) or len(notes) > 3000:
                raise ValueError("Use at most 3000 characters")
            with self.coordinator.lock:
                self.coordinator.findings.append({"worker": self.worker_id, "notes": notes,
                    "source": "agent_statement_not_verified_measurement"})
                self.coordinator.persist()
            return {"shared": True}
        if name.startswith("memory_") and self.condition == "deepseek-rax":
            path = self.root / "memory.txt"
            if name == "memory_write":
                notes = arguments["notes"]
                if not isinstance(notes, str) or len(notes) > 20000:
                    raise ValueError("Use at most 20000 characters")
                path.write_text(notes)
                return {"saved": True}
            if name == "memory_read":
                return {"notes": path.read_text() if path.exists() else "No saved notes."}
        if name == "submit":
            self.submitted = self.freeze(self.root / "frozen")
            P.write_json(self.root / "frozen-manifest.json", self.submitted)
            return {"submitted": True, "sha256": self.submitted["sha256"]}
        raise ValueError("Unknown tool")


def launch_astra(session, admission=None):
    import claude_mac_evaluation as C
    from harness.codex_cli.configuration import write_config
    from task08_swarm_client import SharedClient
    from codex_usage import write as write_usage
    home = session.root / "agent-home"
    home.mkdir()
    auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    if not auth.is_file():
        raise ValueError("Codex login is unavailable")
    (home / "auth.json").symlink_to(auth)
    started, process = time.monotonic(), None
    with tempfile.TemporaryDirectory(prefix="t07-", dir="/tmp") as temporary:
        socket = Path(temporary) / "mcp.sock"
        server = C.mcp_server(session, socket)
        try:
            write_config(home / "config.toml", {"model": "gpt-6-astra", "model_reasoning_effort": "high",
                "approval_policy": "never", "sandbox_mode": "read-only", "web_search": "disabled",
                "features": {name: False for name in ("shell_tool", "multi_agent", "multi_agent_v2", "apps", "plugins",
                    "skill_search", "browser_use", "computer_use", "image_generation")},
                "mcp_servers": {"task07": {"command": str(P.ROOT / ".venv/bin/python"),
                    "args": ["-I", str(P.ROOT / "rax/harness/mcp_bridge.py"), str(socket)], "required": True,
                    "tool_timeout_sec": 950, "default_tools_approval_mode": "approve"}}})
            env = {k: os.environ[k] for k in ("PATH", "LANG", "USER", "LOGNAME", "TMPDIR") if k in os.environ}
            env.update(HOME=str(home), CODEX_HOME=str(home), PYTHONUNBUFFERED="1")
            command = [shutil.which("codex"), "exec", "--skip-git-repo-check", "--json", "-C", str(session.workspace), "--", session.prompt]
            P.write_json(session.root / "launch.json", {"command": command, "fireworks_key_exposed_to_agent": False})
            if admission is not None and time.time() >= admission:
                raise TimeoutError("Astra did not start before the 14:00 Pacific Task07 cutoff")
            with (session.root / "agent.jsonl").open("w") as out, (session.root / "agent.stderr").open("w") as err:
                process = subprocess.Popen(command, cwd=session.workspace, env=env,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                submitted_at = None
                while process.poll() is None and time.monotonic() < session.deadline:
                    if session.submitted is not None:
                        submitted_at = submitted_at or time.monotonic()
                        if time.monotonic() - submitted_at >= 10:
                            break
                    time.sleep(.1)
        finally:
            if process is not None:
                SharedClient._stop(process)
            server.shutdown()
            server.server_close()
            with session.lock:
                if session.submitted is None:
                    session.submitted = session.freeze(session.root / "frozen")
                    P.write_json(session.root / "frozen-manifest.json", session.submitted)
            result = {"worker_id": session.worker_id, "agent_seconds": time.monotonic() - started,
                "development_probes": session.tests, "tool_calls": len(session.history),
                "frozen_sha256": session.submitted["sha256"], "subscription_usage": write_usage(session.root)}
            P.write_json(session.root / "agent-summary.json", result)
    return result


def notify(message):
    try:
        subprocess.run(["agent-message", "claude", "investigator", message],
                       capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def launch_deepseek(client, session, admission):
    if time.time() >= admission:
        raise TimeoutError("DeepSeek worker did not start before the 14:00 Pacific Task07 cutoff")
    return client.run_worker(session, session.worker_id)


def main(argv=None):
    run_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--tests-per-worker", type=int, default=2)
    parser.add_argument("--api-concurrency", type=int, default=3)
    parser.add_argument("--output-tpm", type=int, default=60000)
    parser.add_argument("--simulation-lock", type=Path, default=P.ROOT / "outputs/short-pilots-20261102/task07-simulator.lock")
    parser.add_argument("--final-timeout", type=int, default=3600)
    args = parser.parse_args(argv)
    if (not 1 <= args.seconds <= 900 or args.tests_per_worker != 2 or args.final_timeout < 1
            or not 1 <= args.api_concurrency <= 3 or not 1 <= args.output_tpm <= 30000):
        parser.error("Use at most 900 development seconds and exactly two probes per worker")
    from short_pilot_runtime import MemoryGatedSimulationLock
    admission, stop = Q.local_deadlines()
    admission_deadline = time.monotonic() + admission - time.time()
    hard_deadline = time.monotonic() + stop - time.time() - 30
    if time.time() >= admission:
        raise SystemExit("Task07 dropped: launch admission closed at 14:00 Pacific")
    checks = Q.preflight()
    if not all(checks.values()):
        raise SystemExit("Preflight failed without inference: " + json.dumps(checks))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    registration = P.ROOT / "outputs/short-pilots-20261102/task07-process.json"
    P.write_json(registration, {"pid": os.getpid(), "command": [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output)],
        "output": str(args.output), "started_at": time.time()})
    Q.make_starter(args.output / "starter")
    baseline_started = time.monotonic()
    baseline_lock = MemoryGatedSimulationLock(args.simulation_lock, lambda: admission_deadline,
                                             args.output / "starter-simulation-wait.jsonl")
    with baseline_lock:
        baseline = Q.evaluate(args.output / "starter", args.output / "starter-probe",
                              timeout=min(240, admission - time.time()), stop_at=admission)
    Q.partial_score(baseline)
    if time.time() >= admission:
        raise SystemExit("Task07 dropped: baseline setup missed 14:00 Pacific admission")
    development_deadline = min(time.monotonic() + args.seconds, hard_deadline)
    deep = Coordinator(args.output / "deepseek", args.output / "starter", baseline,
        seconds=args.seconds, tests_per_worker=2, simulator=None, stop_at=stop)
    astra = Coordinator(args.output / "astra", args.output / "starter", baseline,
        seconds=args.seconds, tests_per_worker=2, simulator=None, stop_at=stop)
    for coordinator in (deep, astra):
        coordinator.deadline = development_deadline
        coordinator.simulator = MemoryGatedSimulationLock(args.simulation_lock, lambda: development_deadline,
            coordinator.root / "simulation-wait.jsonl")
    workers = [Session(deep.root / name, deep, name, brief) for name, brief in ROLES]
    astra_session = Session(astra.root / "agent", astra, "astra", "Develop the strongest complete policy within the common budget.", astra=True)
    for session in [*workers, astra_session]:
        session.start()
    protocol = {"task": "Task07", "kind": "short_public_starter_pilot", "development_seconds": args.seconds,
        "workers": 6, "astra_workers": 1, "public_seeds": Q.PUBLIC_SEEDS, "public_episode_seconds": Q.PUBLIC_SECONDS,
        "probes_per_worker": 2, "probe_completion_not_guaranteed": True, "simulator_cpus": Q.SIMULATOR_CPUS,
        "simulator_memory": Q.SIMULATOR_MEMORY, "simulator_concurrency": 1,
        "api_accounts": 3, "api_concurrency_per_account": args.api_concurrency,
        "output_tpm_per_account": args.output_tpm, "prompt_tpm_per_account": 120000,
        "estimated_api_cost_guard_usd": 10., "simulator_memory_gate_gib": 6.5,
        "queue_wait_counts_against_fixed_900s_dev_window": True,
        "simulation_lock": str(args.simulation_lock),
        "simulation_wait_logs": [str(c.root / "simulation-wait.jsonl") for c in (deep, astra)],
        "logical_requests_per_worker": 12, "total_reported_token_guard": 6_000_000,
        "selection": "measured frozen public proxy; unchanged starter wins ties; all agents frozen before any private final",
        "development_started_at": time.time(), "hard_stop_at": datetime.fromtimestamp(stop, ZoneInfo("America/Los_Angeles")).isoformat(),
        "baseline_public_seconds": time.monotonic() - baseline_started,
        "comparison_limits": ["Six DeepSeek branches have twelve total new probes; one Astra branch has two",
            "Single-public-seed 10-second episodes omit most of the official 120-second horizon and do not establish cross-seed robustness",
            "Host CLI overhead is outside Docker resource caps", "Final private evaluation may hit the hard stop"]}
    P.write_json(args.output / "protocol.json", protocol)
    if time.time() >= admission:
        raise SystemExit("Task07 dropped: agent launch missed 14:00 Pacific admission")
    from short_pilot_pool import AccountPool
    client = AccountPool(deep.root / "shared-api", primary_keys(), worker_ids=[s.worker_id for s in workers],
        deadline=development_deadline, max_concurrent=args.api_concurrency, max_workers=6, logical_requests_per_worker=12,
        max_calls=216, per_worker_calls=36, max_tokens=6_000_000, max_output_tokens=8192, max_cost=10.,
        prompt_tpm=500000, uncached_prompt_tpm=250000, generated_tpm=args.output_tpm, prompt_bytes_per_token=2.9)
    deep.client = client
    started = time.monotonic()
    summaries = {}
    try:
        with ThreadPoolExecutor(max_workers=7) as pool:
            futures = {pool.submit(launch_deepseek, client, s, admission): s for s in workers}
            futures[pool.submit(launch_astra, astra_session, admission)] = astra_session
            for future in as_completed(futures):
                session = futures[future]
                try:
                    summaries[session.worker_id] = future.result()
                except Exception as error:
                    summaries[session.worker_id] = {"error": str(error)}
                P.write_json(args.output / "worker-summaries.json", summaries)
    finally:
        client.close()
        P.write_json(deep.root / "api-summary.json", client.summary())
    for session in [*workers, astra_session]:
        with session.lock:
            if session.submitted is None:
                session.submitted = session.freeze(session.root / "frozen")
                P.write_json(session.root / "frozen-manifest.json", session.submitted)
    selections = {"deepseek": deep.select(), "astra": astra.select()}
    P.write_json(args.output / "all-frozen.json", {"at": time.time(), "selections": selections})
    development_seconds = time.monotonic() - started
    P.write_json(args.output / "development-summary.json", {"development_seconds": development_seconds,
        "selections": selections, "api": client.summary(), "all_conditions_frozen": True})
    validity = agent_validity([*workers, astra_session])
    P.write_json(args.output / "agent-validity.json", validity)
    finals = {}
    final_cache = {}
    for name, coordinator in (("deepseek", deep), ("astra", astra)):
        final_started = time.monotonic()
        try:
            condition_sessions = workers if name == "deepseek" else [astra_session]
            if not any(validity[session.worker_id]["tool_calls"] for session in condition_sessions):
                raise RuntimeError("Zero-tool condition: no valid agent run; final evaluation skipped")
            key = selections[name]["manifest"]["sha256"]
            if key in final_cache:
                previous = final_cache[key]
                shutil.copytree(previous / "final", coordinator.root / "final")
                reward = json.loads((coordinator.root / "final/reward.json").read_text())
                reused = str(previous / "final")
            else:
                final_lock = MemoryGatedSimulationLock(args.simulation_lock, lambda: hard_deadline,
                    coordinator.root / "final-simulation-wait.jsonl")
                with final_lock:
                    reward = Q.evaluate(coordinator.root / "selected", coordinator.root / "final", final=True,
                        timeout=args.final_timeout, stop_at=stop - 30)
                final_cache[key] = coordinator.root
                reused = None
            finals[name] = {"official_reward": reward, "score": reward["reward"] * 100,
                           "final_evaluation_reused_from_identical_frozen_policy": reused}
        except Exception as error:
            finals[name] = {"official_reward": None, "error": str(error)}
        finals[name]["final_verification_seconds"] = time.monotonic() - final_started
        P.write_json(args.output / "final-results.json", finals)
        notify(f"Task07 {name}: selected {selections[name]['worker']}, public proxy={selections[name]['public_proxy']:.4f}; "
            f"official score={finals[name].get('score')}, final error={finals[name].get('error')}; "
            "single public seed, 10-second development episodes; exploratory pilot. All agents froze before private evaluation.")
    result = {"task": "Task07", "status": "completed" if all(row.get("official_reward") is not None for row in finals.values()) else "development_completed_final_unavailable",
        "score": None, "conditions": {name: {**finals[name], "selection": selections[name]} for name in finals},
        "selections": selections, "finals": finals,
        "agent_validity": validity, "zero_tool_agent_failure": any(not row["tool_calls"] for row in validity.values()),
        "development_seconds": development_seconds, "total_seconds": time.monotonic() - run_started,
        "api": client.summary(), "astra_subscription_usage": summaries.get("astra", {}).get("subscription_usage"),
        "new_development_probes": {"deepseek": deep.tests, "astra": astra.tests},
        "comparison_limits": protocol["comparison_limits"]}
    P.write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
