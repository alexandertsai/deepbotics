"""Short public-starter GELLO pilot with an explicit unchanged baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
from types import SimpleNamespace
import subprocess
import sys
import threading
import time

import task08_pilot as P

ROOT, REPO = P.ROOT, P.REPO
TASK = REPO / "tasks/task09"
PUBLIC = TASK / "environment/assets"
IMAGE = "rlebench-task09-verifier:dev"
AGENT_IMAGE = "rlebench-task09-agent:dev"
WORKER = ROOT / "scripts/task09_pilot_worker.py"
MODEL = P.MODEL
write_json = P.write_json
VARIANTS = ("franka", "ur5e", "xarm7")
ENTRY_FILES = tuple(f"{v}/{name}" for v in VARIANTS for name in ("lead.xml", "trim.py"))
ZERO_TRIM = '''import numpy as np

def make_trim(model):
    return lambda q: np.zeros(model.nv)

def plan_probe(model, sample, lower, upper):
    return np.tile(np.clip(sample[0], lower, upper), (15, 1))

def adapt(model, probe):
    return None

def make_trim_adapted(model, params):
    return make_trim(model)
'''


def make_starter(destination):
    destination.mkdir(parents=True, exist_ok=False)
    for variant in VARIANTS:
        target = destination / variant
        target.mkdir()
        shutil.copyfile(PUBLIC / variant / "lead.xml", target / "lead.xml")
        shutil.copytree(PUBLIC / variant / "meshes", target / "meshes")
        (target / "trim.py").write_text(ZERO_TRIM)


def evaluate(bundle, output, *, final=False, timeout=900):
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    import hashlib
    name = "task09-pilot-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
        "--cpus", "4", "--memory", "4g", "--pids-limit", "128", "--tmpfs", "/tmp:rw,nosuid,size=1g",
        "--security-opt", "no-new-privileges", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
        "-e", f"MUJOCO_GL={'osmesa' if final else 'disable'}", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{bundle.resolve()}:/submission:ro"]
    if final:
        command += ["-e", "PYTHONPATH=/tests", "-e", "RLEBENCH_ARTIFACTS=/submission",
            "-v", f"{output.resolve()}:/logs/verifier", IMAGE, "python3", "/tests/score_task.py"]
    else:
        command += ["-v", f"{output.resolve()}:/logs", "-v", f"{WORKER}:/pilot/probe.py:ro",
            AGENT_IMAGE, "python3", "/pilot/probe.py"]
    try:
        completed = P.checked(command, timeout=timeout)
        (output / "container.log").write_text(completed.stdout + completed.stderr)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
        raise TimeoutError("Evaluation wall-time budget exhausted") from None
    except subprocess.CalledProcessError as error:
        (output / "container.log").write_text((error.stdout or "") + (error.stderr or ""))
        raise RuntimeError("Evaluation container failed; operator log records the error") from None
    return json.loads((output / ("reward.json" if final else "probe.json")).read_text())


class Session(P.Session):
    def __init__(self, root, condition, seconds, tests, starter):
        self.root, self.condition = root.resolve(), condition
        self.workspace = self.root / "workspace"
        self.workspace.mkdir(parents=True)
        self.candidate = self.root / "candidate"
        self.initial = starter
        shutil.copytree(starter, self.candidate)
        self.history, self.tests, self.max_tests = [], 0, tests
        self.max_seconds, self.deadline, self.submitted = seconds, None, None
        self.lock = threading.RLock()
        self.prompt = self.make_prompt()
        (self.root / "prompt.txt").write_text(self.prompt)

    def make_prompt(self):
        prompt = f'''Task 09 public-starter pilot. Improve the three supplied bare GELLO models and zero-torque trim stubs.
This is a shortened pilot, not a full three-hour leaderboard replication. No oracle or reference solution is supplied.
You have {self.max_seconds} wall seconds and {self.max_tests} development probe invocations across all variants.
Only pilot MCP tools may be used. No shell, external filesystem, network, web, other agents or other apps.
Candidate artifacts are franka/lead.xml, franka/trim.py, ur5e/lead.xml, ur5e/trim.py, xarm7/lead.xml, xarm7/trim.py.
Public files use the public/ prefix and candidate files use candidate/. list_files lists both.
Use edit_file for exact replacements or write_file for complete lead.xml/trim.py files. Meshes are supplied and immutable.
test_candidate freezes all artifacts and tests every variant with public nominal helpers: loading, mass,
passive residual, nominal/adapted hold errors and servo saturation at three public poses. It also exercises
the 16-measurement plan_probe/adapt protocol on the nominal model. This is not the private verifier score.
The final frozen files are evaluated once on the unchanged private Task09 verifier after both agents stop.
Call submit before stopping. If the budget expires, the supervisor freezes the current files. Preserve a usable candidate.
No emojis. Minimize code comments. Report measured facts and distinguish guesses. Public helpers are readable with read_file.
Do not rely on a local XML path, hidden parameters or task-specific payload priors in trim.py.
The original task instructions follow; its three-hour allowance is replaced by this pilot's shorter budget.
{(TASK / 'instruction.md').read_text()}
'''
        if self.condition == "deepseek-rax":
            from jinja2 import Environment, StrictUndefined
            renderer = Environment(undefined=StrictUndefined)
            prompt += "\nRAX adaptation: use hypotheses, frozen tested versions, evidence notes and a frozen submission.\n"
            for name in ("evidence-and-regression", "change-discipline"):
                text = (ROOT / "rax/agent_skills/method" / name / "SKILL.md").read_text().split("---", 2)[-1]
                prompt += renderer.from_string(text).render(agent_formal=False)
            prompt += "\nHere, probe reports and version manifests replace archive paths. Use memory_write and memory_read for cumulative evidence.\n"
        return prompt

    def definitions(self):
        definitions = super().definitions()
        for tool in definitions:
            name = tool["name"]
            if name in ("write_file", "edit_file"):
                tool["inputSchema"]["properties"]["name"]["enum"] = list(ENTRY_FILES)
                tool["description"] = ("Replace a candidate entry file completely." if name == "write_file"
                    else "Replace exactly one unique string in a candidate entry file.")
            elif name == "test_candidate":
                tool["description"] = "Freeze all variants and spend one public nominal development probe."
        return definitions

    def freeze(self, destination):
        from services.controller.storage import snapshot
        return snapshot(self.candidate, destination, max_bytes=256 * 1024**2, workspace_path="code/task09")

    def dispatch(self, name, arguments):
        if name == "status":
            return super().dispatch(name, arguments)
        if self.submitted is not None or time.monotonic() >= self.deadline:
            raise ValueError("Development has ended")
        if name == "list_files":
            return {"candidate": list(ENTRY_FILES), "public": [str(p.relative_to(PUBLIC))
                for p in PUBLIC.rglob("*") if p.is_file()]}
        if name == "read_file":
            path = P.safe_relative(arguments["path"])
            base = {"public": PUBLIC, "candidate": self.candidate}.get(path.parts[0])
            if base is None or len(path.parts) < 2:
                raise ValueError("Use public/ or candidate/ paths")
            file = base.joinpath(*path.parts[1:])
            if not file.resolve().is_relative_to(base.resolve()) or file.is_symlink():
                raise ValueError("Unsafe file path")
            if file.suffix not in (".xml", ".json", ".py", ".txt", ".md"):
                raise ValueError("Only text assets can be read")
            offset, length = arguments.get("offset", 0), arguments.get("length", 30000)
            if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 60000:
                raise ValueError("Invalid text range")
            text = file.read_text()
            return {"text": text[offset:offset + length], "total_characters": len(text)}
        if name in ("write_file", "edit_file"):
            if arguments["name"] not in ENTRY_FILES:
                raise ValueError("Only variant lead.xml and trim.py may be changed")
            file = self.candidate / arguments["name"]
            if name == "edit_file":
                old, original = arguments["old"], file.read_text()
                if not old or original.count(old) != 1:
                    raise ValueError("old must match exactly once")
                content = original.replace(old, arguments["new"], 1)
            else:
                content = arguments["content"]
            if len(content.encode()) > 1024**2:
                raise ValueError("Entry file exceeds pilot limit")
            file.write_text(content)
            return {"saved": True}
        if name == "test_candidate":
            if self.tests >= self.max_tests:
                raise ValueError("Development probe budget exhausted")
            index = self.tests
            self.tests += 1
            bundle = self.root / "versions" / str(index)
            bundle.parent.mkdir(exist_ok=True)
            manifest = self.freeze(bundle)
            P.write_json(self.root / "versions" / f"{index}.json",
                {"manifest": manifest, "hypothesis": arguments["hypothesis"]})
            result = evaluate(bundle, self.root / "probes" / str(index),
                timeout=max(1, min(900, self.deadline - time.monotonic())))
            return {"version": index, "manifest": manifest["sha256"], "report": result}
        return super().dispatch(name, arguments)


def preflight(images=True):
    checks = P.preflight(images=False)
    checks["assets_staged"] = all((PUBLIC / v / "lead.xml").is_file() for v in VARIANTS)
    if images:
        for image, key in ((IMAGE, "verifier_image_built"), (AGENT_IMAGE, "public_image_built")):
            try:
                P.checked(["docker", "image", "inspect", image], timeout=10)
                checks[key] = True
            except (OSError, subprocess.SubprocessError):
                checks[key] = False
    return checks


def prepare_assets():
    image = "kimirax-task09-assets:dev"
    dockerfile = ("FROM python:3.12-slim\n"
        "RUN pip install --no-cache-dir mujoco==3.5.0 numpy==2.5.1 scipy==1.18.0\n")
    subprocess.run(["docker", "build", "-t", image, "-"], input=dockerfile, text=True, check=True)
    subprocess.run(["docker", "run", "--rm", "--network", "none", "-e", "MUJOCO_GL=disable",
        "-e", "OPENBLAS_NUM_THREADS=1", "-v", f"{REPO}:/repo", "-w", "/repo", image,
        "python3", "tasks/task09/build_assets.py"], check=True)


def launch(session, max_calls):
    import claude_mac_evaluation as C
    import mac_evaluation as M
    from harness.codex_cli.configuration import write_config
    from replicate_deepseek import Relay

    keys = M.load_keys()
    process = None
    relay = server = None
    with tempfile.TemporaryDirectory(prefix="t09-", dir="/tmp") as temporary:
        socket = Path(temporary) / "mcp.sock"
        server = C.mcp_server(session, socket)
        bridge = ["-I", str(ROOT / "rax/harness/mcp_bridge.py"), str(socket)]
        home = session.root / "agent-home"
        home.mkdir()
        env = {k: os.environ[k] for k in ("PATH", "LANG", "USER", "LOGNAME", "TMPDIR") if k in os.environ}
        env.update(HOME=str(home), PYTHONUNBUFFERED="1")
        if session.condition == "astra":
            auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
            if not auth.is_file():
                raise ValueError("Codex auth.json unavailable; run codex login first")
            (home / "auth.json").symlink_to(auth)
            write_config(home / "config.toml", {
                "model": "gpt-6-astra", "model_reasoning_effort": "high", "approval_policy": "never",
                "sandbox_mode": "read-only", "web_search": "disabled",
                "features": {"shell_tool": False, "multi_agent": False, "multi_agent_v2": False,
                             "apps": False, "plugins": False, "skill_search": False,
                             "browser_use": False, "computer_use": False, "image_generation": False},
                "mcp_servers": {"task09": {"command": str(ROOT / ".venv/bin/python"), "args": bridge,
                                          "required": True, "tool_timeout_sec": 950,
                                          "default_tools_approval_mode": "approve"}},
            })
            env["CODEX_HOME"] = str(home)
            command = [shutil.which("codex"), "exec", "--skip-git-repo-check", "--json",
                       "-C", str(session.workspace), "--", session.prompt]
        else:
            args = SimpleNamespace(max_calls=max_calls, max_tokens=250000,
                                   max_output_tokens=8192, max_cost=2.0)
            meter = Relay(session.root, keys, args)
            relay = meter.server()
            write_json(home / "mcp.json", {"mcpServers": {"task09": {"type": "stdio",
                       "command": str(ROOT / ".venv/bin/python"), "args": bridge}}})
            write_json(home / "settings.json", {"permissions": {"allow": ["mcp__task09__*"],
                       "deny": ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "Agent", "Task", "WebFetch", "WebSearch"]}})
            write_json(home / ".claude.json", {"hasCompletedOnboarding": True})
            env.update(CLAUDE_CONFIG_DIR=str(home), ANTHROPIC_BASE_URL=f"http://127.0.0.1:{relay.server_port}",
                       ANTHROPIC_API_KEY=meter.token, ANTHROPIC_DEFAULT_OPUS_MODEL=MODEL,
                       ANTHROPIC_DEFAULT_SONNET_MODEL=MODEL, ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL,
                       CLAUDE_CODE_MAX_OUTPUT_TOKENS="8192", CLAUDE_CODE_MAX_RETRIES="0",
                       DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1", DISABLE_AUTOUPDATER="1",
                       CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", ENABLE_TOOL_SEARCH="false")
            command = [shutil.which("claude"), "-p", "--bare", "--model", MODEL, "--effort", "max",
                       "--max-turns", str(max_calls), "--tools", "", "--allowedTools", "mcp__task09__*",
                       "--strict-mcp-config", "--mcp-config", str(home / "mcp.json"),
                       "--settings", str(home / "settings.json"), "--output-format", "stream-json",
                       "--verbose", "--", session.prompt]
        write_json(session.root / "launch.json", {"command": command, "fireworks_key_exposed_to_agent": False})
        session.start()
        started = time.monotonic()
        submission_seen = None
        try:
            with (session.root / "agent.jsonl").open("w") as out, (session.root / "agent.stderr").open("w") as err:
                process = subprocess.Popen(command, cwd=session.workspace, env=env, stdout=out, stderr=err,
                                           start_new_session=True)
                while process.poll() is None:
                    if time.monotonic() >= session.deadline:
                        os.killpg(process.pid, signal.SIGTERM)
                        break
                    if session.submitted is not None:
                        if submission_seen is None:
                            submission_seen = time.monotonic()
                        if session.condition != "astra" or time.monotonic() - submission_seen >= 10:
                            os.killpg(process.pid, signal.SIGTERM)
                            break
                    if session.condition != "astra" and meter.exhausted.is_set():
                        os.killpg(process.pid, signal.SIGTERM)
                        break
                    time.sleep(0.25)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            server.shutdown()
            server.server_close()
            if relay:
                relay.shutdown()
                relay.server_close()
        with session.lock:
            if session.submitted is None:
                session.submitted = session.freeze(session.root / "frozen")
                write_json(session.root / "frozen-manifest.json", session.submitted)
        summary = {"condition": session.condition, "agent_seconds": time.monotonic() - started,
                   "returncode": process.returncode, "development_probes": session.tests,
                   "frozen_sha256": session.submitted["sha256"], "tool_calls": len(session.history),
                   "candidate_changed": any((session.candidate / name).read_bytes() !=
                       (session.initial / name).read_bytes() for name in ENTRY_FILES)}
        if session.condition != "astra":
            summary["api"] = meter.summary()
            summary["stop_reason"] = "api_budget" if meter.exhausted.is_set() else "agent_exit"
            write_json(session.root / "api-summary.json", summary["api"])
        if session.condition == "astra":
            from codex_usage import write as write_usage
            summary["subscription_usage"] = write_usage(session.root)
        write_json(session.root / "agent-summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task09-pilot-01")
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--tests", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--final-timeout", type=int, default=3600)
    parser.add_argument("--baseline-result", type=Path)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900 or not 0 <= args.tests <= 3 or not 1 <= args.max_calls <= 12:
        parser.error("Pilot limits: seconds 1..900, tests 0..3, calls 1..12")
    checks = preflight(images=args.action != "prepare")
    print(json.dumps(checks), flush=True)
    if args.action == "preflight":
        return 0 if all(checks.values()) else 1
    if args.action == "prepare":
        if not checks["docker_running"]:
            raise SystemExit("Docker must be running; no model calls made")
        prepare_assets()
        subprocess.run(["docker", "build", "-t", AGENT_IMAGE, str(TASK / "environment")], check=True)
        subprocess.run(["docker", "build", "-f", str(TASK / "tests/Dockerfile"), "-t", IMAGE, str(REPO)], check=True)
        return 0
    if not all(checks.values()):
        raise SystemExit("Preflight failed; no model calls made")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    make_starter(output / "starter")
    baseline = None
    if args.baseline_result:
        from services.controller.storage import verify
        baseline_root = args.baseline_result.resolve().parent.parent
        manifest = json.loads((baseline_root / "manifest.json").read_text())
        verify(baseline_root / "frozen", manifest)
        import hashlib
        for name, digest in manifest["files"].items():
            if hashlib.sha256((output / "starter" / name).read_bytes()).hexdigest() != digest:
                raise SystemExit("Provided baseline does not match this pilot's public starter")
        baseline = json.loads(args.baseline_result.read_text())
    P.write_json(output / "protocol.json", {
        "kind": "shortened_public_starter_pilot_not_leaderboard_replication",
        "task": "RLE-Bench Task 09", "variants": list(VARIANTS),
        "starter": "Public bare GELLO lead models and zero-torque trim stubs; no oracle supplied",
        "seconds_per_condition": args.seconds, "development_probes_per_condition": args.tests,
        "deepseek_max_calls": args.max_calls, "deepseek_max_reported_tokens": 250000,
        "deepseek_estimated_spend_guard_usd": 2.0,
        "rax_scope": "Adapted RAX method instructions, MCP bridge and artifact snapshots; not complete upstream RAX backend",
        "development_feedback": "Public nominal helper only; no hidden verifier access",
        "baseline_reused_from": str(args.baseline_result.resolve()) if args.baseline_result else None,
        "upstream_commit": P.checked(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip(),
    })
    summaries = []
    for condition in ("astra", "deepseek-rax"):
        session = Session(output / condition, condition, args.seconds, args.tests, output / "starter")
        print(json.dumps({"event": "agent_started", "condition": condition}), flush=True)
        summaries.append(launch(session, args.max_calls))
        P.write_json(output / "summary.json", summaries)
        if not summaries[-1]["tool_calls"]:
            raise SystemExit("No tool activity; stopping before further model requests")
    baselines = [("unchanged_starter", output / "starter")] if baseline is None else []
    if baseline is not None:
        summaries.insert(0, {"condition": "unchanged_starter", "official_reward": baseline,
            "reused_from": str(args.baseline_result.resolve())})
    for condition, bundle in baselines + [
            (s["condition"], output / s["condition"] / "frozen") for s in summaries]:
        if condition == "unchanged_starter" and baseline is not None:
            continue
        print(json.dumps({"event": "final_evaluation_started", "condition": condition}), flush=True)
        if condition == "unchanged_starter":
            row = {"condition": condition}
            summaries.insert(0, row)
        else:
            row = next(s for s in summaries if s["condition"] == condition)
            from services.controller.storage import verify
            verify(bundle, json.loads((output / condition / "frozen-manifest.json").read_text()))
        try:
            row["official_reward"] = evaluate(bundle, output / condition / "final", final=True, timeout=args.final_timeout)
        except Exception as error:
            row["evaluation_error"] = str(error)
        P.write_json(output / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
