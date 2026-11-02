"""Minimal macOS launcher: real Claude Code, Fireworks, and RAX robot MCP tools.

The full RAX exploration/compile/formal workflow is not used by this short test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socketserver
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2

import mac_evaluation as M
from services.controller.gateway import sanitized
from services.dexjoco.backend import DexjocoBackend
from harness.claude_cli.login import Credential

MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
MODELS = {"deepseek": MODEL, "kimi": "accounts/fireworks/models/kimi-k3"}
UPSTREAM = "https://api.fireworks.ai/inference"


def usage_from_response(data, streaming):
    if not streaming:
        return json.loads(data).get("usage", {})
    usage = {}
    for line in data.decode().splitlines():
        if not line.startswith("data: "):
            continue
        if line[6:].strip() == "[DONE]":
            continue
        event = json.loads(line[6:])
        usage.update(event.get("message", {}).get("usage") or {})
        usage.update(event.get("usage") or {})
    return usage


def token_total(usage):
    return sum(int(usage.get(k) or 0) for k in (
        "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))


def cli_result(transcript):
    result = None
    for line in transcript.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "result":
            result = {k: row.get(k) for k in ("subtype", "is_error", "num_turns", "result", "errors")}
    return result


class Episode:
    def __init__(self, directory, args, keys):
        self.directory, self.args, self.keys = directory, args, keys
        self.model = MODELS[args.model]
        self.lock = threading.RLock()
        self.calls, self.api_records, self.tool_records = 0, [], []
        self.reported_tokens = 0
        self.budget_reached = False
        self.relay_token = secrets.token_hex(32)
        self.sim = M.new_sim(directory, f"claude-{args.model}", args.layout, collection=getattr(args, "collection", 1))
        cfg = SimpleNamespace(root=directory, task="fold_glasses", benchmark="dexjoco", observation_profile="rgb-only", sim_gpu="mac")
        self.backend = DexjocoBackend(cfg)
        self.backend.sim = self.sim
        self.initial_observation = self.observe("initial")

    def observe(self, label):
        obs = self.sim.request("observe")
        safe, images = M.save_observation(obs, self.directory / "images", label)
        return {"safe": safe, "hashes": {c: images[c]["sha256"] for c in M.CAMERAS}}

    def definitions(self):
        return [{"name": item["function"]["name"], "description": item["function"]["description"],
                 "inputSchema": item["function"]["parameters"]} for item in M.api_tools()]

    def tool(self, name, arguments):
        with self.lock:
            sequence = len(self.tool_records)
            record = {"tool": name, "arguments": arguments, "step_before": self.sim.step_id}
            try:
                if name not in {d["name"] for d in self.definitions()}:
                    raise ValueError("Tool outside allowlist")
                if sequence >= getattr(self.args, "max_tool_calls", 60):
                    raise ValueError("Tool call budget exhausted")
                M.T.validate_request(name, arguments, action_dim=22)
                remaining = self.args.max_steps - self.sim.step_id
                if name == M.T.STEP:
                    state = self.sim.request("observe")["state"]
                    M.validate_bounded_actions(arguments["actions"], state, remaining)
                if name == M.T.HOLD and arguments["steps"] > min(20, remaining):
                    raise ValueError("Hold exceeds remaining budget or 20-step limit")
                packet, images = self.backend.call(name, arguments)
                packet = sanitized(packet)
                packet.update(remaining_steps=self.args.max_steps - self.sim.step_id)
                result = {"content": [{"type": "text", "text": json.dumps(packet)}, *images], "isError": False}
                record.update(accepted=True, result=packet)
                self.observe(f"tool-{sequence:02d}")
            except Exception as exc:
                record.update(accepted=False, error=f"{type(exc).__name__}: {exc}")
                result = {"isError": True, "content": [{"type": "text", "text": json.dumps({"error": record["error"]})}]}
            record["step_after"] = self.sim.step_id
            self.tool_records.append(record)
            M.write_json(self.directory / "tools" / f"{sequence:02d}.json", M.redact(record, self.keys))
            return result

    def mcp(self, request):
        ident, method = request.get("id"), request.get("method")
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            value = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                     "serverInfo": {"name": "rax-mac-short-evaluation", "version": "1.0"}}
        elif method == "tools/list":
            value = {"tools": self.definitions()}
        elif method == "tools/call":
            value = self.tool(request["params"]["name"], request["params"].get("arguments", {}))
        elif method == "ping":
            value = {}
        else:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": "Unsupported method"}}
        return {"jsonrpc": "2.0", "id": ident, "result": value}


def mcp_server(episode, path):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while line := self.rfile.readline(4 * 1024 * 1024):
                reply = episode.mcp(json.loads(line))
                if reply is not None:
                    self.wfile.write(json.dumps(reply).encode() + b"\n")
                    self.wfile.flush()

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

    server = Server(str(path), Handler)
    path.chmod(0o600)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def relay_server(episode):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def fail(self, status, message):
            self.reply(status, json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}}).encode())

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + episode.relay_token and self.headers.get("x-api-key") != episode.relay_token:
                self.fail(401, "Invalid local relay credential")
                return
            if self.path.split("?")[0] != "/v1/messages":
                self.fail(404, "Only the Messages API is available")
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with episode.lock:
                if episode.calls >= episode.args.max_calls or episode.reported_tokens >= episode.args.max_total_tokens:
                    episode.budget_reached = True
                    self.fail(400, "Evaluation API budget exhausted")
                    return
                index = episode.calls
                episode.calls += 1
            selected_model = getattr(episode, "model", MODEL)
            body["model"] = selected_model
            body["max_tokens"] = min(int(body.get("max_tokens", 4096)), episode.args.max_output_tokens)
            body["output_config"] = {"effort": "low"}
            if body.get("thinking", {}).get("type") == "enabled":
                body["thinking"]["budget_tokens"] = min(1024, body["max_tokens"] - 1)
            M.write_json(episode.directory / "api" / f"request-{index:02d}.json", M.redact(body, episode.keys))
            started = time.monotonic()
            record = {"call": index, "model": selected_model}
            try:
                with httpx2.Client(timeout=episode.args.timeout) as client:
                    response = client.post(UPSTREAM + "/v1/messages", json=body,
                                           headers={"Authorization": "Bearer " + episode.keys["FIREWORKS_API_KEY"], "Content-Type": "application/json"})
                data = response.content
                record.update(status_code=response.status_code, latency_seconds=time.monotonic() - started)
                content_type = response.headers.get("content-type", "application/json")
                streaming = "text/event-stream" in content_type
                if response.is_success:
                    record["usage"] = usage_from_response(data, streaming)
                    episode.reported_tokens += token_total(record["usage"])
                else:
                    record["error"] = data.decode(errors="replace")
                safe_text = M.redact({"body": data.decode(errors="replace")}, episode.keys)["body"]
                (episode.directory / "api" / f"response-{index:02d}.txt").write_text(safe_text)
                self.reply(response.status_code, safe_text.encode(), content_type)
            except Exception as exc:
                record.update(latency_seconds=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}")
                self.fail(502, M.redact(record, episode.keys)["error"])
            finally:
                episode.api_records.append(M.redact(record, episode.keys))
                M.write_json(episode.directory / "api" / f"metrics-{index:02d}.json", M.redact(record, episode.keys))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run(args, directory, keys, episode_factory=Episode):
    episode = episode_factory(directory, args, keys)
    process = None
    result = {"harness": "Claude Code", "adapter": "minimal_custom_macos_launcher", "provider": "Fireworks",
              "model": episode.model,
              "seed": 1000 + args.layout if getattr(args, "collection", 1) == 0 else args.layout // 50,
              "episode_index": 0 if getattr(args, "collection", 1) == 0 else args.layout % 50,
              "budget": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "initial": episode.initial_observation}
    result["budget"]["output"] = str(args.output) if args.output else None
    relay = relay_server(episode)
    with tempfile.TemporaryDirectory(prefix="rax-mac-", dir="/tmp") as temp:
        socket = Path(temp) / "mcp.sock"
        mcp = mcp_server(episode, socket)
        try:
            workspace = directory / "agent-workspace"
            home = directory / "claude-home"
            workspace.mkdir()
            home.mkdir()
            config = home / "mcp.json"
            M.write_json(config, {"mcpServers": {"auto_research": {"type": "stdio", "command": str(M.ROOT / ".venv/bin/python"),
                "args": ["-I", str(M.ROOT / "rax/harness/mcp_bridge.py"), str(socket)]}}})
            settings = home / "settings.json"
            M.write_json(settings, {"permissions": {"allow": ["mcp__auto_research__*"],
                "deny": ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "Agent", "Task", "WebFetch", "WebSearch"]}})
            M.write_json(home / ".claude.json", {"hasCompletedOnboarding": True})
            os.environ["ANTHROPIC_AUTH_TOKEN"] = episode.relay_token
            try:
                auth = Credential("custom", base_url=f"http://127.0.0.1:{relay.server_port}").environment(model=episode.model)
            finally:
                os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            env = {k: os.environ[k] for k in ("PATH", "LANG", "USER", "LOGNAME", "TMPDIR") if k in os.environ}
            env.update(auth)
            env["ANTHROPIC_API_KEY"] = episode.relay_token
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
            env.update(HOME=str(home), CLAUDE_CONFIG_DIR=str(home), CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(args.max_output_tokens),
                       CLAUDE_CODE_MAX_RETRIES="0", DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1",
                       DISABLE_AUTOUPDATER="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", ENABLE_TOOL_SEARCH="false")
            prompt = getattr(args, "agent_prompt", None) or M.system_prompt(args) + " Call dexjoco_observe first to obtain both images. Use the RAX robot tools to make progress now."
            command = [shutil.which("claude"), "-p", "--bare", "--model", episode.model, "--effort", "low", "--max-turns", str(args.max_calls),
                       "--tools", "", "--allowedTools", "mcp__auto_research__*", "--strict-mcp-config", "--mcp-config", str(config),
                       "--settings", str(settings), "--output-format", "stream-json", "--verbose", "--", prompt]
            M.write_json(directory / "launch.json", {"command": command, "builtin_tools": [], "mcp_tools": [d["name"] for d in episode.definitions()]})
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=workspace, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=15)
                result["timeout"] = True
            for name, text in (("claude-transcript.jsonl", stdout), ("claude-stderr.txt", stderr)):
                (directory / name).write_text(M.redact({"text": text}, {**keys, "relay": episode.relay_token})["text"])
            result.update(returncode=process.returncode, wall_seconds=time.monotonic() - started)
            result["cli_result"] = cli_result(stdout)
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            episode.sim.finish("short_episode_budget" if episode.budget_reached else "claude_code_returned")
            result.update(api=episode.api_records, model_calls=episode.calls, reported_total_tokens=episode.reported_tokens,
                          tool_calls=episode.tool_records, control_steps=episode.sim.step_id, budget_reached=episode.budget_reached,
                          official_outcome=M.outcome(episode.sim))
            turn_cap = (result.get("cli_result") or {}).get("subtype") == "error_max_turns"
            result["status"] = "budget_exhausted" if episode.budget_reached or turn_cap else ("cli_error" if result.get("returncode") else "finished")
            result["stop_reason"] = "api_call_budget" if episode.budget_reached else ("claude_turn_budget" if turn_cap else "claude_returned")
            result["success"] = result["official_outcome"]["native_score"] == 1 if any(r.get("usage") for r in episode.api_records) else None
            M.write_json(directory / "result.json", M.redact(result, keys))
            episode.sim.stop()
            mcp.shutdown()
            mcp.server_close()
            relay.shutdown()
            relay.server_close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="deepseek")
    parser.add_argument("--layout", type=int, default=0, choices=range(150))
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-total-tokens", type=int, default=80000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (1 <= args.max_calls <= 20 and 1 <= args.max_steps <= 1200 and 1025 <= args.max_output_tokens <= 8192
            and 1 <= args.max_total_tokens <= 200000 and 1 <= args.timeout <= 180):
        parser.error("Budget outside supported bounds")
    keys = M.load_keys()
    if not keys.get("FIREWORKS_API_KEY"):
        parser.error("Missing FIREWORKS_API_KEY; run scripts/enter_api_keys.py --provider fireworks")
    directory = (args.output or M.ROOT / "outputs/mac-evaluation" / f"claude-fireworks-{time.strftime('%Y%m%dT%H%M%S')}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    checked = M.smoke(directory / "smoke", args.layout)
    print(json.dumps({"smoke_passed": checked["passed"], "output": str(directory)}), flush=True)
    if not checked["passed"]:
        raise SystemExit(1)
    result = run(args, directory, keys)
    print(json.dumps({k: result.get(k) for k in ("harness", "model", "success", "model_calls", "control_steps", "reported_total_tokens", "returncode")}), flush=True)
    if not any(r.get("usage") for r in result["api"]) or result["status"] == "cli_error":
        raise SystemExit(2)


if __name__ == "__main__":
    os.umask(0o077)
    main()
