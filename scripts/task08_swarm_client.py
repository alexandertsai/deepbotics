"""Shared FIFO inference meter and isolated CLI launchers for Task 08 workers."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import task08_pilot as P

MODEL = P.MODEL
UPSTREAM = "https://api.fireworks.ai/inference/v1/messages"
USAGE_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def response_usage(data, content_type):
    if "text/event-stream" not in content_type:
        return json.loads(data).get("usage") or {}
    usage = {}
    for line in data.decode().splitlines():
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        event = json.loads(line[6:])
        usage.update(event.get("message", {}).get("usage") or {})
        usage.update(event.get("usage") or {})
    return usage


class BudgetRefused(RuntimeError):
    pass


class SharedClient:
    def __init__(self, root, keys, *, deadline, max_calls=12, max_tokens=250000,
                 max_output_tokens=8192, max_cost=2.0, per_worker_calls=3,
                 max_concurrent=1, max_workers=4, transport=None, start_server=True):
        if not keys.get("FIREWORKS_API_KEY"):
            raise ValueError("Fireworks key is not configured")
        for value, upper, label in ((max_calls, 10000, "Global request limit"),
                (per_worker_calls, 10000, "Worker request limit"),
                (max_concurrent, 8, "Concurrent request limit"),
                (max_workers, 32, "Worker count limit"),
                (max_tokens, 1_000_000_000, "Token threshold"),
                (max_output_tokens, 32768, "Output token limit")):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{label} must be a positive integer at most {upper}")
        if isinstance(max_cost, bool) or not isinstance(max_cost, (int, float)) or not math.isfinite(max_cost) or not 0 < max_cost <= 1_000_000:
            raise ValueError("Estimated-cost limit must be positive and finite")
        if not math.isfinite(float(deadline)):
            raise ValueError("Deadline must be finite")
        self.root, self.keys = Path(root).resolve(), keys
        self.root.mkdir(parents=True, exist_ok=True)
        self.deadline = float(deadline)
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.max_output_tokens, self.max_cost = max_output_tokens, max_cost
        self.per_worker_calls = per_worker_calls
        self.max_concurrent, self.max_workers = max_concurrent, max_workers
        self.condition = threading.Condition(threading.RLock())
        self.queue, self.active = deque(), 0
        self.peak_active, self.peak_queued = 0, 0
        self.reserved_tokens, self.reserved_cost = 0, 0.0
        self.active_slot_seconds, self.busy_wall_seconds = 0.0, 0.0
        self.activity_updated = time.monotonic()
        self.calls, self.tokens, self.cost, self.uncertain_tokens = 0, 0, 0.0, 0
        self.usage = dict.fromkeys(USAGE_FIELDS, 0)
        self.records, self.refusals, self.workers = [], [], {}
        self.exhausted, self.closed = threading.Event(), threading.Event()
        self.transport = transport or self._post_upstream
        self.active_clients = set()
        self.shutdown_errors = []
        self.processes = {}
        self.server = None
        self.server_thread = None
        if start_server:
            self._start_server()

    def _start_server(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, body, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def fail(self, status, message):
                body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
                self.reply(status, json.dumps(body).encode())

            def do_POST(self):
                match = re.fullmatch(r"/worker/([A-Za-z0-9_-]+)/v1/messages", self.path.split("?", 1)[0])
                if not match:
                    return self.fail(404, "Only worker Messages requests are supported")
                worker_id = match[1]
                with owner.condition:
                    worker = owner.workers.get(worker_id)
                    token = worker["token"] if worker else None
                supplied = self.headers.get("x-api-key") or self.headers.get("Authorization", "").removeprefix("Bearer ")
                if not token or not secrets.compare_digest(supplied, token):
                    return self.fail(401, "Invalid local worker credential")
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4 * 1024**2:
                        raise ValueError("Request size exceeds the pilot limit")
                    body = json.loads(self.rfile.read(size))
                    status, data, content_type = owner.request(worker_id, body)
                    self.reply(status, data, content_type)
                except BudgetRefused as error:
                    self.fail(400, str(error))
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    self.fail(400, str(error))
                except Exception as error:
                    self.fail(502, owner.redact(str(error)))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def register_worker(self, worker_id, max_calls=None, *, session=None):
        if not isinstance(worker_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", worker_id):
            raise ValueError("Invalid worker identifier")
        with self.condition:
            if worker_id in self.workers:
                raise ValueError("Worker is already registered")
            if len(self.workers) >= self.max_workers:
                raise ValueError(f"The pilot permits at most {self.max_workers} workers")
            allowance = self.per_worker_calls if max_calls is None else min(max_calls, self.per_worker_calls)
            if type(allowance) is not int or allowance < 1:
                raise ValueError("Worker allowance must be positive")
            self.workers[worker_id] = {"token": secrets.token_hex(32), "requests": 0, "allowance": allowance,
                                       "refused": threading.Event(), "session": session}
            return self.workers[worker_id]["token"]

    def redact(self, value):
        encoded = json.dumps(value, allow_nan=False)
        with self.condition:
            secrets_to_hide = [*self.keys.values(), *(w["token"] for w in self.workers.values())]
        for secret in secrets_to_hide:
            if isinstance(secret, str) and secret:
                encoded = encoded.replace(secret, "[REDACTED]")
        return json.loads(encoded)

    def _update_activity_unlocked(self):
        now = time.monotonic()
        elapsed = max(0.0, now - self.activity_updated)
        self.active_slot_seconds += elapsed * self.active
        self.busy_wall_seconds += elapsed * bool(self.active)
        self.activity_updated = now

    def _summary_unlocked(self):
        self._update_activity_unlocked()
        return {"requests": self.calls, "reported_tokens": self.tokens,
                "unknown_usage_token_allowance": self.uncertain_tokens,
                "estimated_cost_usd": self.cost, "usage": dict(self.usage),
                "budget_exhausted": (self.exhausted.is_set() or self.calls >= self.max_calls
                    or self.tokens + self.uncertain_tokens >= self.max_tokens),
                "errors": sum(bool(r.get("error")) for r in self.records),
                "active_requests": self.active, "queued_requests": len(self.queue),
                "peak_active_requests": self.peak_active, "peak_queued_requests": self.peak_queued,
                "max_concurrent": self.max_concurrent, "max_workers": self.max_workers,
                "inflight_token_reservations": self.reserved_tokens,
                "inflight_estimated_cost_usd": self.reserved_cost,
                "aggregate_active_slot_seconds": self.active_slot_seconds,
                "provider_busy_wall_seconds": self.busy_wall_seconds,
                "shutdown_errors": list(self.shutdown_errors),
                "worker_requests": {k: v["requests"] for k, v in self.workers.items()},
                "refusals": list(self.refusals),
                "limits": {"requests": self.max_calls, "reported_token_stop_threshold": self.max_tokens,
                    "output_tokens_per_request": self.max_output_tokens, "requests_per_worker": self.per_worker_calls,
                    "estimated_cost_usd": self.max_cost},
                "token_guard": ("Stop threshold checked between serialized requests; one completed call may cross it."
                    if self.max_concurrent == 1 else
                    "Concurrent admission reserves input JSON bytes, 4096 template tokens, and maximum output tokens for every in-flight request."),
                "cost_guard": "Estimate uses historical input/cache/output rates, not a provider billing cap."}

    def summary(self):
        with self.condition:
            return self._summary_unlocked()

    def _save_summary_unlocked(self):
        P.write_json(self.root / "api-summary.json", self._summary_unlocked())

    def _refuse(self, worker_id, reason, global_stop):
        if global_stop:
            self.exhausted.set()
        self.workers[worker_id]["refused"].set()
        self.refusals.append({"worker_id": worker_id, "reason": reason, "time": time.time()})
        self._save_summary_unlocked()
        self.condition.notify_all()
        raise BudgetRefused(reason)

    def _post_upstream(self, body, timeout):
        import httpx2
        with httpx2.Client(timeout=timeout) as client:
            with self.condition:
                if self.closed.is_set():
                    raise BudgetRefused("Swarm client is closed")
                self.active_clients.add(client)
            try:
                response = client.post(UPSTREAM, content=body,
                    headers={"Authorization": "Bearer " + self.keys["FIREWORKS_API_KEY"],
                             "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                return response.status_code, response.content, response.headers.get("content-type", "application/json")
            finally:
                with self.condition:
                    self.active_clients.discard(client)

    def _budget_notice(self, worker_id):
        worker = self.workers[worker_id]
        used, total = worker["requests"] + 1, self.calls + 1
        remaining = max(0, worker["allowance"] - used)
        global_remaining = max(0, self.max_calls - total)
        notice = (f"Budget notice: This is worker request {used}/{worker['allowance']}; "
            f"worker requests remaining after this response: {remaining}. "
            f"Shared requests admitted including this one: {total}/{self.max_calls}; "
            f"shared requests remaining: {global_remaining}. "
            f"Shared reported tokens committed before this response: {self.tokens}/{self.max_tokens}; "
            f"output limit for this response: {self.max_output_tokens} tokens. "
            f"Global development time remaining: {max(0, int(self.deadline - time.monotonic()))} seconds.")
        if remaining <= 2 or global_remaining <= 4 or self.tokens >= self.max_tokens * .8:
            notice += (" Budget is nearly consumed. Prioritize implementing or editing a concrete candidate "
                "and requesting a development test now, in the task's required tool order. "
                "Minimize further inspection; record measured evidence and preserve a usable submission.")
        return notice

    def _before_accounting(self, record, *, reservation, token_reservation, request_bytes, max_output_tokens):
        return None

    @staticmethod
    def _append_notice(body, notice):
        body = dict(body)
        messages = body.get("messages")
        if isinstance(messages, list):
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    content = content + "\n\n" + notice
                elif isinstance(content, list) and all(isinstance(block, dict) for block in content):
                    content = [*content, {"type": "text", "text": notice}]
                else:
                    break
                messages = list(messages)
                messages[index] = {**message, "content": content}
                body["messages"] = messages
                return body
        system = body.get("system", "")
        if isinstance(system, str):
            body["system"] = system + ("\n\n" if system else "") + notice
        elif isinstance(system, list) and all(isinstance(block, dict) for block in system):
            body["system"] = [*system, {"type": "text", "text": notice}]
        else:
            raise ValueError("Cannot safely append a budget notice to this request")
        return body

    def request(self, worker_id, body):
        if not isinstance(body, dict):
            raise ValueError("Expected a Messages request object")
        body = dict(body)
        body["model"] = MODEL
        body["max_tokens"] = min(int(body.get("max_tokens", self.max_output_tokens)), self.max_output_tokens)
        if body["max_tokens"] < 1:
            raise ValueError("max_tokens must be positive")
        body["thinking"], body["output_config"] = {"type": "adaptive"}, {"effort": "max"}
        ticket = object()
        queued_at = time.monotonic()
        with self.condition:
            if worker_id not in self.workers:
                raise ValueError("Unknown worker")
            self.queue.append(ticket)
            self.peak_queued = max(self.peak_queued, len(self.queue))
            try:
                while True:
                    session = self.workers[worker_id].get("session")
                    if session is not None and session.submitted is not None:
                        self._refuse(worker_id, "Worker has already submitted", False)
                    remaining = self.deadline - time.monotonic()
                    if self.closed.is_set() or remaining <= 0:
                        self._refuse(worker_id, "Shared experiment deadline or shutdown reached", True)
                    if self.workers[worker_id]["requests"] >= self.workers[worker_id]["allowance"]:
                        self._refuse(worker_id, "Worker request allowance exhausted", False)
                    if self.exhausted.is_set() or self.calls >= self.max_calls or self.tokens + self.uncertain_tokens >= self.max_tokens:
                        self._refuse(worker_id, "Shared experiment request or token budget exhausted", True)
                    if self.active < self.max_concurrent and self.queue[0] is ticket:
                        request_body = self._append_notice(body, self._budget_notice(worker_id))
                        encoded = json.dumps(request_body).encode()
                        reservation = (len(encoded) * .30 + body["max_tokens"] * 1.20) / 1e6
                        token_reservation = (len(encoded) + body["max_tokens"] + 4096
                                             if self.max_concurrent > 1 else 0)
                        token_room = (self.tokens + self.uncertain_tokens + self.reserved_tokens
                                      + token_reservation <= self.max_tokens)
                        cost_room = self.cost + self.reserved_cost + reservation <= self.max_cost
                        if token_room and cost_room:
                            body = request_body
                            index = self.calls
                            self.calls += 1
                            self.workers[worker_id]["requests"] += 1
                            self.queue.popleft()
                            self._update_activity_unlocked()
                            self.active += 1
                            self.peak_active = max(self.peak_active, self.active)
                            self.reserved_tokens += token_reservation
                            self.reserved_cost += reservation
                            self.condition.notify_all()
                            break
                        if not self.active:
                            reason = ("Shared conservative token reservation cannot fit remaining allowance"
                                      if not token_room else "Shared estimated-cost allowance exhausted")
                            self._refuse(worker_id, reason, self.max_concurrent == 1)
                    self.condition.wait(timeout=min(.2, remaining))
            except BaseException:
                self.queue.remove(ticket)
                self.condition.notify_all()
                raise
        started = time.monotonic()
        record = {"call": index, "worker_id": worker_id, "model": MODEL,
                  "effort": "max", "started_at": time.time(), "queue_seconds": started - queued_at,
                  "reserved_tokens": token_reservation, "reserved_estimated_cost_usd": reservation}
        status, data, content_type = 502, b"", "application/json"
        try:
            P.write_json(self.root / "api" / f"request-{index:04d}.json",
                         self.redact({"worker_id": worker_id, "body": body}))
            timeout = max(.1, min(120.0, self.deadline - time.monotonic()))
            record["provider_started_at"] = time.time()
            provider_started = time.monotonic()
            status, data, content_type = self.transport(encoded, timeout)
            record["provider_latency_seconds"] = time.monotonic() - provider_started
            record["status_code"] = status
            if 200 <= status < 300:
                usage = response_usage(data, content_type)
                if not usage:
                    raise ValueError("Provider response omitted usage accounting")
                counts = {k: int(usage.get(k) or 0) for k in USAGE_FIELDS}
                record["usage"] = counts
                record["reported_tokens"] = sum(counts.values())
                record["estimated_cost_usd"] = ((counts["input_tokens"] + counts["cache_creation_input_tokens"]) * .30
                    + counts["cache_read_input_tokens"] * .006 + counts["output_tokens"] * 1.20) / 1e6
                if token_reservation and record["reported_tokens"] > token_reservation:
                    record["reservation_overrun"] = True
            else:
                record["error"] = data.decode(errors="replace")
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            record["estimated_cost_usd"] = reservation
            record["unknown_usage_token_allowance"] = token_reservation or len(encoded) + body["max_tokens"]
            status = 502
            data = json.dumps({"type": "error", "error": {"type": "api_error", "message": self.redact(str(error))}}).encode()
            content_type = "application/json"
        finally:
            record["latency_seconds"] = time.monotonic() - started
            self._before_accounting(record, reservation=reservation, token_reservation=token_reservation,
                                    request_bytes=len(encoded), max_output_tokens=body["max_tokens"])
            record = self.redact(record)
            safe = self.redact(data.decode(errors="replace")).encode()
            try:
                (self.root / "api" / f"response-{index:04d}.txt").write_bytes(safe)
                P.write_json(self.root / "api" / f"metrics-{index:04d}.json", record)
            finally:
                with self.condition:
                    self.cost += record.get("estimated_cost_usd", 0)
                    self.tokens += record.get("reported_tokens", 0)
                    self.uncertain_tokens += record.get("unknown_usage_token_allowance", 0)
                    for key in USAGE_FIELDS:
                        self.usage[key] += record.get("usage", {}).get(key, 0)
                    self.records.append(record)
                    self._update_activity_unlocked()
                    self.active -= 1
                    self.reserved_tokens -= token_reservation
                    self.reserved_cost = max(0.0, self.reserved_cost - reservation)
                    if record.get("reservation_overrun"):
                        self.exhausted.set()
                    try:
                        self._save_summary_unlocked()
                    finally:
                        self.condition.notify_all()
        return status, safe, content_type

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            except PermissionError:
                if process.poll() is None:
                    process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if process.poll() is None:
                        process.kill()
                process.wait(timeout=5)

    def run_worker(self, session, worker_id, max_calls=None):
        import claude_mac_evaluation as C
        if self.server is None:
            raise RuntimeError("The HTTP relay must be started before launching workers")
        if session.deadline is None:
            session.deadline = self.deadline
        if abs(session.deadline - self.deadline) > .001:
            raise ValueError("Worker must share the experiment deadline")
        allowance = self.per_worker_calls if max_calls is None else min(max_calls, self.per_worker_calls)
        namespace = getattr(session, "mcp_namespace", "task08")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", namespace):
            raise ValueError("Invalid MCP namespace")
        token = self.register_worker(worker_id, allowance, session=session)
        process, mcp = None, None
        started = time.monotonic()
        stop_reason = "agent_exit"
        try:
            with tempfile.TemporaryDirectory(prefix="t08w-", dir="/tmp") as temporary:
                socket = Path(temporary) / "mcp.sock"
                mcp = C.mcp_server(session, socket)
                home = session.root / "agent-home"
                home.mkdir()
                P.write_json(home / "mcp.json", {"mcpServers": {namespace: {"type": "stdio",
                    "command": str(P.ROOT / ".venv/bin/python"),
                    "args": ["-I", str(P.ROOT / "rax/harness/mcp_bridge.py"), str(socket)]}}})
                P.write_json(home / "settings.json", {"permissions": {"allow": [f"mcp__{namespace}__*"],
                    "deny": ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "Agent", "Task", "WebFetch", "WebSearch"]}})
                P.write_json(home / ".claude.json", {"hasCompletedOnboarding": True})
                env = {k: os.environ[k] for k in ("PATH", "LANG", "USER", "LOGNAME", "TMPDIR") if k in os.environ}
                env.update(HOME=str(home), CLAUDE_CONFIG_DIR=str(home),
                    ANTHROPIC_BASE_URL=f"http://127.0.0.1:{self.server.server_port}/worker/{worker_id}",
                    ANTHROPIC_API_KEY=token, ANTHROPIC_DEFAULT_OPUS_MODEL=MODEL,
                    ANTHROPIC_DEFAULT_SONNET_MODEL=MODEL, ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL,
                    CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(self.max_output_tokens), CLAUDE_CODE_MAX_RETRIES="0",
                    DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1", DISABLE_AUTOUPDATER="1",
                    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", ENABLE_TOOL_SEARCH="false")
                command = [shutil.which("claude"), "-p", "--bare", "--model", MODEL, "--effort", "max",
                    "--max-turns", str(allowance), "--tools", "", "--allowedTools", f"mcp__{namespace}__*",
                    "--strict-mcp-config", "--mcp-config", str(home / "mcp.json"),
                    "--settings", str(home / "settings.json"), "--output-format", "stream-json",
                    "--verbose", "--", session.prompt]
                P.write_json(session.root / "launch.json", {"command": command, "worker_id": worker_id,
                    "shared_budget": True, "fireworks_key_exposed_to_agent": False})
                with (session.root / "agent.jsonl").open("w") as out, (session.root / "agent.stderr").open("w") as err:
                    process = subprocess.Popen(command, cwd=session.workspace, env=env,
                        stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                    with self.condition:
                        self.processes[worker_id] = process
                    while process.poll() is None:
                        if self.closed.is_set() or time.monotonic() >= self.deadline:
                            stop_reason = "global_deadline_or_shutdown"
                            break
                        if session.submitted is not None:
                            stop_reason = "submitted"
                            break
                        if self.workers[worker_id]["refused"].is_set():
                            stop_reason = "worker_api_budget"
                            break
                        time.sleep(.1)
                    if process.poll() is None:
                        self._stop(process)
        finally:
            if process is not None:
                self._stop(process)
            if mcp is not None:
                mcp.shutdown()
                mcp.server_close()
            with self.condition:
                self.processes.pop(worker_id, None)
            with session.lock:
                if session.submitted is None:
                    session.submitted = session.freeze(session.root / "frozen")
                    P.write_json(session.root / "frozen-manifest.json", session.submitted)
            result = {"worker_id": worker_id, "agent_seconds": time.monotonic() - started,
                "returncode": process.returncode if process is not None else None,
                "stop_reason": stop_reason, "tool_calls": len(session.history),
                "development_probes": session.tests, "frozen_sha256": session.submitted["sha256"],
                "api_requests": self.workers[worker_id]["requests"]}
            P.write_json(session.root / "agent-summary.json", result)
        return result

    def close(self, wait_timeout=10.0):
        self.closed.set()
        with self.condition:
            self.condition.notify_all()
            processes = list(self.processes.values())
            clients = list(self.active_clients)
        if processes:
            with ThreadPoolExecutor(max_workers=len(processes)) as pool:
                list(pool.map(self._stop, processes))
        for client in clients:
            try:
                client.close()
            except Exception as error:
                self.shutdown_errors.append(type(error).__name__)
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        with self.condition:
            settle_deadline = time.monotonic() + max(0.0, wait_timeout)
            while self.active or self.queue:
                remaining = settle_deadline - time.monotonic()
                if remaining <= 0:
                    self.shutdown_errors.append("InFlightRequestsDidNotSettle")
                    self._save_summary_unlocked()
                    raise TimeoutError("In-flight provider requests did not finish recording during shutdown")
                self.condition.wait(timeout=min(.2, remaining))
            self._save_summary_unlocked()
