"""Prepare and run a bounded Task 08 pilot using Codex and a RAX adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor/rle-bench"
TASK = REPO / "tasks/task08"
PUBLIC = TASK / "environment/assets"
IMAGE = "rlebench-task08-verifier:dev"
MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
WORKER = ROOT / "scripts/task08_pilot_worker.py"
sys.path.insert(0, str(ROOT / "rax"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked(command, *, timeout=30, **kwargs):
    return subprocess.run(command, check=True, timeout=timeout, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def credential_status():
    import mac_evaluation as M
    keys = M.load_keys()
    return bool(keys.get("FIREWORKS_API_KEY"))


def preflight(images=True):
    checks = {"fireworks_key_configured": credential_status(),
              "codex_installed": bool(shutil.which("codex")),
              "claude_installed": bool(shutil.which("claude")),
              "assets_staged": (PUBLIC / "scene.xml").is_file()}
    try:
        checked(["docker", "info", "--format", "{{.OSType}}"], timeout=10)
        checks["docker_running"] = True
    except (OSError, subprocess.SubprocessError):
        checks["docker_running"] = False
    if images:
        try:
            checked(["docker", "image", "inspect", IMAGE], timeout=10)
            checks["verifier_image_built"] = True
        except (OSError, subprocess.SubprocessError):
            checks["verifier_image_built"] = False
    try:
        result = checked(["codex", "login", "status"], timeout=10)
        checks["codex_signed_in"] = "Logged in" in result.stdout + result.stderr
    except (OSError, subprocess.SubprocessError):
        checks["codex_signed_in"] = False
    return checks


def safe_relative(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or "\\" in value or "\x00" in value or str(path) != value):
        raise ValueError("Expected a relative workspace path")
    return path


def docker_evaluate(bundle, output, *, final=False, timeout=900):
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    name = "task08-pilot-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none",
               "--cpus", "4", "--memory", "4g", "--pids-limit", "128",
               "--security-opt", "no-new-privileges", "-e", "PYTHONPATH=/tests",
               "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
               "-e", "MUJOCO_GL=osmesa", "-v", f"{bundle.resolve()}:/submission:ro",
               "-v", f"{output.resolve()}:/logs/verifier"]
    if final:
        command += ["-e", "RLEBENCH_ARTIFACTS=/submission", IMAGE,
                    "python3", "/tests/score_task.py"]
    else:
        command += ["-v", f"{WORKER}:/pilot/probe.py:ro", IMAGE,
                    "python3", "/pilot/probe.py"]
    try:
        result = checked(command, timeout=timeout)
        (output / "container.log").write_text(result.stdout + result.stderr)
    except subprocess.CalledProcessError as error:
        (output / "container.log").write_text((error.stdout or "") + (error.stderr or ""))
        raise RuntimeError("Evaluation container failed; the operator log records the error") from None
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
        raise TimeoutError("Evaluation wall-time budget exhausted") from None
    result_file = output / ("reward.json" if final else "probe.json")
    return json.loads(result_file.read_text())


class Session:
    def __init__(self, root, condition, seconds, tests):
        self.root, self.condition = root.resolve(), condition
        root = self.root
        self.workspace = root / "workspace"
        self.workspace.mkdir(parents=True)
        self.candidate = root / "candidate"
        self.candidate.mkdir()
        for name in ("robot.xml", "controller.py"):
            shutil.copyfile(TASK / "solution/payload" / name, self.candidate / name)
        self.history = []
        self.max_tests = tests
        self.tests = 0
        self.max_seconds = seconds
        self.deadline = None
        self.submitted = None
        self.lock = threading.RLock()
        self.prompt = self.make_prompt()
        (root / "prompt.txt").write_text(self.prompt)

    def start(self):
        self.deadline = time.monotonic() + self.max_seconds

    def make_prompt(self):
        text = (TASK / "instruction.md").read_text()
        common = f"""Task 08 improvement pilot. Start from the supplied reference robot.xml and controller.py.
This differs from the official from-scratch benchmark. Improve the common base and controller while preserving validity.
You have {self.max_seconds} wall seconds and {self.max_tests} development probe invocations total.
Only task08 MCP tools may be used. Do not use shell, filesystem, network, web, external apps or other agents.
Use list_files/read_file to inspect public assets and the candidate. edit_file replaces an exact string; write_file replaces a whole artifact.
test_candidate freezes both files and runs an independent development probe: all three canonical arms,
static sweep and acceleration/braking at development seed 11, plus target indices 0,4,8 at 1 kg and 2 kg.
The probe is not the official score. Final evaluation runs the unchanged full official verifier once, after submission.
Call submit before stopping. If time expires the supervisor freezes the current candidate. Report observed facts only.
No emojis. Keep code comments minimal. Do not alter the trusted components or claim unmeasured success.
The candidate artifacts live behind MCP, not in your CLI filesystem. Both conditions receive identical public information.
The original official specification follows; its two-hour allowance is replaced by this pilot's smaller budget.
{text}
"""
        if self.condition == "deepseek-rax":
            from jinja2 import Environment, StrictUndefined
            renderer = Environment(undefined=StrictUndefined)
            common += "\nRAX adaptation: use explicit hypotheses, tested frozen versions, persistent evidence notes and a final frozen submission.\n"
            for name in ("evidence-and-regression", "change-discipline"):
                skill = (ROOT / "rax/agent_skills/method" / name / "SKILL.md").read_text()
                common += "\n" + renderer.from_string(skill.split("---", 2)[-1]).render(agent_formal=False)
            common += "\nFor this adapter, tool-returned probe reports and version manifests replace the archive paths mentioned above. Use memory_write and memory_read for cumulative evidence and version decisions. Distinguish evidence from guesses.\n"
        return common

    def definitions(self):
        def tool(name, description, properties, required=()):
            return {"name": name, "description": description,
                    "inputSchema": {"type": "object", "properties": properties,
                                    "required": list(required), "additionalProperties": False}}
        definitions = [
            tool("list_files", "List the public input files and candidate artifacts.", {}),
            tool("read_file", "Read a public input or candidate text file; prefix public/ or candidate/.",
                 {"path": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
                  "length": {"type": "integer", "minimum": 1, "maximum": 60000}}, ("path",)),
            tool("write_file", "Replace candidate robot.xml or controller.py in full.",
                 {"name": {"type": "string", "enum": ["robot.xml", "controller.py"]},
                  "content": {"type": "string"}}, ("name", "content")),
            tool("edit_file", "Replace one unique exact string in a candidate artifact.",
                 {"name": {"type": "string", "enum": ["robot.xml", "controller.py"]},
                  "old": {"type": "string"}, "new": {"type": "string"}}, ("name", "old", "new")),
            tool("test_candidate", "Freeze and test the current files, spending one development probe.",
                 {"hypothesis": {"type": "string"}}, ("hypothesis",)),
            tool("status", "Get remaining time, development tests and submission status.", {}),
            tool("submit", "Freeze the current candidate for final evaluation and end development.", {}),
        ]
        if self.condition == "deepseek-rax":
            definitions.append(tool("memory_write", "Save cumulative evidence and version decisions.",
                                    {"notes": {"type": "string"}}, ("notes",)))
            definitions.append(tool("memory_read", "Read cumulative evidence and version decisions.", {}))
        return definitions

    def freeze(self, destination):
        from services.controller.storage import snapshot
        return snapshot(self.candidate, destination, max_bytes=5 * 1024**2,
                        entrypoint="controller.py", workspace_path="code/task08")

    def tool(self, name, arguments):
        with self.lock:
            record = {"name": name, "arguments": arguments}
            try:
                result = self.dispatch(name, arguments)
                reply = {"content": [{"type": "text", "text": json.dumps(result, allow_nan=False)}], "isError": False}
            except Exception as error:
                reply = {"content": [{"type": "text", "text": str(error)}], "isError": True}
            record["reply"] = reply
            self.history.append(record)
            write_json(self.root / "tools.json", self.history)
            return reply

    def dispatch(self, name, arguments):
        if name == "status":
            return {"seconds_remaining": max(0, round(self.deadline - time.monotonic())),
                    "tests_remaining": self.max_tests - self.tests,
                    "submitted": self.submitted is not None}
        if self.submitted is not None or time.monotonic() >= self.deadline:
            raise ValueError("Development has ended")
        if name == "list_files":
            return {"candidate": ["robot.xml", "controller.py"],
                    "public": [str(p.relative_to(PUBLIC)) for p in PUBLIC.rglob("*") if p.is_file()]}
        if name == "read_file":
            path = safe_relative(arguments["path"])
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
            if arguments["name"] not in ("robot.xml", "controller.py"):
                raise ValueError("Unknown artifact")
            file = self.candidate / arguments["name"]
            if name == "edit_file":
                old = arguments["old"]
                original = file.read_text()
                if not old or original.count(old) != 1:
                    raise ValueError("old must match exactly once")
                content = original.replace(old, arguments["new"], 1)
            else:
                content = arguments["content"]
            limit = (4 if arguments["name"] == "robot.xml" else 1) * 1024**2
            if len(content.encode()) > limit:
                raise ValueError("Artifact exceeds task size limit")
            file.write_text(content)
            return {"saved": True}
        if name == "memory_write" and self.condition == "deepseek-rax":
            notes = arguments["notes"]
            if len(notes) > 20000:
                raise ValueError("Notes exceed 20000 characters")
            (self.root / "memory.txt").write_text(notes)
            return {"saved": True}
        if name == "memory_read" and self.condition == "deepseek-rax":
            path = self.root / "memory.txt"
            return {"notes": path.read_text() if path.exists() else "No saved notes."}
        if name == "test_candidate":
            if self.tests >= self.max_tests:
                raise ValueError("Development probe budget exhausted")
            index = self.tests
            self.tests += 1
            bundle = self.root / "versions" / str(index)
            bundle.parent.mkdir(exist_ok=True)
            manifest = self.freeze(bundle)
            write_json(self.root / "versions" / f"{index}.json",
                       {"manifest": manifest, "hypothesis": arguments["hypothesis"]})
            report = docker_evaluate(bundle, self.root / "probes" / str(index),
                                     timeout=max(1, min(900, self.deadline - time.monotonic())))
            return {"version": index, "manifest": manifest["sha256"], "report": report}
        if name == "submit":
            self.submitted = self.freeze(self.root / "frozen")
            write_json(self.root / "frozen-manifest.json", self.submitted)
            return {"submitted": True, "sha256": self.submitted["sha256"]}
        raise ValueError("Unknown tool")

    def mcp(self, request):
        ident, method = request.get("id"), request.get("method", "")
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "task08-pilot", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": self.definitions()}
        elif method == "tools/call":
            result = self.tool(request["params"]["name"], request["params"].get("arguments", {}))
        elif method == "ping":
            result = {}
        else:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": "Unknown method"}}
        return {"jsonrpc": "2.0", "id": ident, "result": result}


def launch(session, max_calls):
    import claude_mac_evaluation as C
    import mac_evaluation as M
    from harness.codex_cli.configuration import write_config
    from replicate_deepseek import Relay

    keys = M.load_keys()
    process = None
    relay = server = None
    with tempfile.TemporaryDirectory(prefix="t08-", dir="/tmp") as temporary:
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
                "mcp_servers": {"task08": {"command": str(ROOT / ".venv/bin/python"), "args": bridge,
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
            write_json(home / "mcp.json", {"mcpServers": {"task08": {"type": "stdio",
                       "command": str(ROOT / ".venv/bin/python"), "args": bridge}}})
            write_json(home / "settings.json", {"permissions": {"allow": ["mcp__task08__*"],
                       "deny": ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "Agent", "Task", "WebFetch", "WebSearch"]}})
            write_json(home / ".claude.json", {"hasCompletedOnboarding": True})
            env.update(CLAUDE_CONFIG_DIR=str(home), ANTHROPIC_BASE_URL=f"http://127.0.0.1:{relay.server_port}",
                       ANTHROPIC_API_KEY=meter.token, ANTHROPIC_DEFAULT_OPUS_MODEL=MODEL,
                       ANTHROPIC_DEFAULT_SONNET_MODEL=MODEL, ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL,
                       CLAUDE_CODE_MAX_OUTPUT_TOKENS="8192", CLAUDE_CODE_MAX_RETRIES="0",
                       DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1", DISABLE_AUTOUPDATER="1",
                       CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", ENABLE_TOOL_SEARCH="false")
            command = [shutil.which("claude"), "-p", "--bare", "--model", MODEL, "--effort", "max",
                       "--max-turns", str(max_calls), "--tools", "", "--allowedTools", "mcp__task08__*",
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
                       (TASK / "solution/payload" / name).read_bytes() for name in ("robot.xml", "controller.py"))}
        if session.condition == "astra":
            from codex_usage import write as write_usage
            summary["subscription_usage"] = write_usage(session.root)
        write_json(session.root / "agent-summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task08-pilot")
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--tests", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--final-timeout", type=int, default=5400)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900 or not 0 <= args.tests <= 3 or not 1 <= args.max_calls <= 12:
        parser.error("Pilot limits: seconds 1..900, tests 0..3, model calls 1..12")
    checks = preflight(images=args.action != "prepare")
    print(json.dumps(checks), flush=True)
    if args.action == "preflight":
        return 0 if all(checks.values()) else 1
    if args.action == "prepare":
        if not checks["docker_running"]:
            raise SystemExit("Docker Desktop must be running before image preparation; no model calls were made.")
        subprocess.run([sys.executable, str(TASK / "build_assets.py")], check=True, cwd=REPO)
        subprocess.run(["docker", "build", "-f", str(TASK / "tests/Dockerfile"), "-t", IMAGE, str(REPO)], check=True)
        return 0
    if not all(checks.values()):
        raise SystemExit("Preflight failed; no model calls were made. Fix the false checks and run prepare.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {
        "task": "RLE-Bench Task 08", "upstream_commit": checked(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip(),
        "kind": "reference_starter_improvement_pilot_not_leaderboard_replication",
        "conditions": ["astra", "deepseek-rax"], "development_seconds_per_condition": args.seconds,
        "development_tests_per_condition": args.tests, "deepseek_max_calls": args.max_calls,
        "deepseek_estimated_spend_guard_usd": 2.0,
        "cost_guard_note": "Existing relay estimates use historical rates; call/token limits are enforced independently.",
        "rax_scope": "adapted method skills, MCP bridge and content-addressed bundles; not a full upstream RAX backend",
        "final_evaluation": "unchanged official scorer; each frozen submission once; no feedback to agents",
    })
    summaries = []
    for condition in ("astra", "deepseek-rax"):
        session = Session(output / condition, condition, args.seconds, args.tests)
        print(json.dumps({"event": "agent_started", "condition": condition}), flush=True)
        summaries.append(launch(session, args.max_calls))
        write_json(output / "summary.json", summaries)
        if not summaries[-1]["tool_calls"]:
            raise SystemExit(f"{condition} exited without tool activity; stopping before further model requests")
    for summary in summaries:
        condition = summary["condition"]
        print(json.dumps({"event": "final_evaluation_started", "condition": condition}), flush=True)
        try:
            from services.controller.storage import verify
            manifest = json.loads((output / condition / "frozen-manifest.json").read_text())
            verify(output / condition / "frozen", manifest)
            summary["official_reward"] = docker_evaluate(output / condition / "frozen",
                output / condition / "final", final=True, timeout=args.final_timeout)
        except Exception as error:
            summary["evaluation_error"] = str(error)
        write_json(output / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
