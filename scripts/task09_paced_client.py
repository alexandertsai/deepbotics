"""Three-slot DeepSeek broker with shared TPM pacing and bounded 429 retries."""

from __future__ import annotations

from collections import deque
from email.utils import parsedate_to_datetime
import json
import math
import random
import threading
import time

from task08_swarm_client import BudgetRefused, SharedClient
import task08_pilot as P

RATE_HEADERS = {
    "prompt": "x-ratelimit-limit-tokens-prompt",
    "uncached": "x-ratelimit-limit-tokens-uncached-prompt",
    "generated": "x-ratelimit-limit-tokens-generated",
}


def retry_after_seconds(value, *, now=None):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                return None
            return max(0.0, date.timestamp() - (time.time() if now is None else now))
        except (TypeError, ValueError, OverflowError):
            return None


def safe_rate_headers(headers):
    allowed = {"retry-after", *RATE_HEADERS.values()}
    output = {}
    for key, value in headers.items():
        name = str(key).lower()
        if (name in allowed or name.startswith("x-ratelimit-remaining-")
                or name.startswith("x-ratelimit-reset-")):
            output[name] = str(value).replace("\r", "").replace("\n", "")[:256]
    return output


class PacedClient(SharedClient):
    def __init__(self, root, keys, *, deadline, max_calls=270, per_worker_calls=18,
                 logical_requests_per_worker=6, max_workers=32, max_concurrent=3,
                 max_tokens=6_000_000, max_output_tokens=8192, max_cost=5.0,
                 prompt_tpm=120000, uncached_prompt_tpm=120000, generated_tpm=30000,
                 start_stagger_seconds=2.0, max_retries=2, base_backoff_seconds=5.0,
                 max_backoff_seconds=60.0, jitter_seconds=1.0, prompt_bytes_per_token=2.0,
                 transport=None, start_server=True, _window_seconds=60.0):
        for value, label in ((logical_requests_per_worker, "Logical request allowance"),
                             (prompt_tpm, "Prompt TPM"), (uncached_prompt_tpm, "Uncached prompt TPM"),
                             (generated_tpm, "Generated TPM")):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if type(max_retries) is not int or not 0 <= max_retries <= 2:
            raise ValueError("429 retries must be between zero and two")
        if not 1 <= max_concurrent <= 3:
            raise ValueError("This paced trial permits at most three active API slots")
        for value in (start_stagger_seconds, base_backoff_seconds, max_backoff_seconds,
                      jitter_seconds, prompt_bytes_per_token, _window_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Pacing and backoff settings must be finite and nonnegative")
        if prompt_bytes_per_token == 0 or _window_seconds == 0:
            raise ValueError("Pacing token ratio and window must be positive")
        self.logical_requests_per_worker = logical_requests_per_worker
        self.logical_requests = 0
        self.logical_active = 0
        self.rejected_cost_allowance = 0.0
        self.rejected_token_allowance = 0
        self.logical_results = []
        self.max_retries = max_retries
        self.pacing_config = {"prompt": prompt_tpm, "uncached": uncached_prompt_tpm, "generated": generated_tpm}
        self.header_limits = {}
        self.rate_scale = 1.0
        self.start_stagger_seconds = start_stagger_seconds
        self.base_backoff_seconds, self.max_backoff_seconds = base_backoff_seconds, max_backoff_seconds
        self.jitter_seconds, self.prompt_bytes_per_token = jitter_seconds, prompt_bytes_per_token
        self.window_seconds = _window_seconds
        self.pacing_history, self.pacing_queue = deque(), deque()
        self.pacing_leases = {}
        self.next_lease = 0
        self.last_pacing_start = -float("inf")
        self.cooldown_until, self.last_429_at = 0.0, -float("inf")
        self.rate_limited_attempts = 0
        self.random = random.Random(0)
        self.attempt_context = threading.local()
        self.raw_transport = transport
        super().__init__(root, keys, deadline=deadline, max_calls=max_calls,
            per_worker_calls=per_worker_calls, max_workers=max_workers, max_concurrent=max_concurrent,
            max_tokens=max_tokens, max_output_tokens=max_output_tokens, max_cost=max_cost,
            transport=self._attempt_transport, start_server=start_server)

    def register_worker(self, worker_id, max_calls=None, *, session=None):
        if max_calls is not None and (type(max_calls) is not int or max_calls < 1):
            raise ValueError("Logical request allowance must be positive")
        logical_limit = self.logical_requests_per_worker if max_calls is None else min(max_calls, self.logical_requests_per_worker)
        token = super().register_worker(worker_id, self.per_worker_calls, session=session)
        with self.condition:
            self.workers[worker_id].update(logical_requests=0, logical_allowance=logical_limit)
        return token

    def run_worker(self, session, worker_id, max_calls=None):
        allowance = self.logical_requests_per_worker if max_calls is None else min(max_calls, self.logical_requests_per_worker)
        result = super().run_worker(session, worker_id, max_calls=allowance)
        with self.condition:
            result["logical_requests"] = self.workers[worker_id]["logical_requests"]
            result["provider_attempts"] = self.workers[worker_id]["requests"]
        P.write_json(session.root / "agent-summary.json", result)
        return result

    def _rates_unlocked(self):
        return {key: max(1.0, min(value * self.rate_scale, self.header_limits.get(key, value)))
                for key, value in self.pacing_config.items()}

    def _summary_unlocked(self):
        summary = super()._summary_unlocked()
        summary.update(logical_requests=self.logical_requests, provider_attempts=self.calls,
            active_logical_requests=self.logical_active,
            rejected_request_billing="Unknown; conservative full-request cost and token allowances retained for 429 attempts.",
            rejected_request_estimated_cost_allowance_usd=self.rejected_cost_allowance,
            rejected_request_unknown_token_allowance=self.rejected_token_allowance,
            worker_logical_requests={key: value.get("logical_requests", 0) for key, value in self.workers.items()},
            logical_requests_per_worker=self.logical_requests_per_worker,
            max_429_retries_per_logical_request=self.max_retries,
            rate_limited_attempts=self.rate_limited_attempts,
            pacing={"configured_tpm": dict(self.pacing_config), "effective_tpm": self._rates_unlocked(),
                    "header_limits": dict(self.header_limits), "throughput_scale": self.rate_scale,
                    "cooldown_seconds_remaining": max(0.0, self.cooldown_until - time.monotonic()),
                    "waiting_requests": len(self.pacing_queue), "reserved_requests": len(self.pacing_leases),
                    "start_stagger_seconds": self.start_stagger_seconds,
                    "prompt_bytes_per_token_estimate": self.prompt_bytes_per_token,
                    "window_seconds": self.window_seconds})
        return summary

    def _budget_notice(self, worker_id):
        worker = self.workers[worker_id]
        self.attempt_context.call_index = self.calls
        logical = worker["logical_requests"]
        remaining = max(0, worker["logical_allowance"] - logical)
        notice = (f"Budget notice: Logical model request {logical}/{worker['logical_allowance']}; "
            f"logical requests remaining after this response: {remaining}. "
            f"This is provider attempt {worker['requests'] + 1}/{worker['allowance']} for this worker; "
            f"global provider attempts admitted including this one: {self.calls + 1}/{self.max_calls}. "
            f"429 retries consume provider attempts but do not consume another logical request. "
            f"Shared reported tokens committed: {self.tokens}/{self.max_tokens}; "
            f"global time remaining: {max(0, int(self.deadline - time.monotonic()))} seconds.")
        if remaining <= 2:
            notice += " Implement or edit a concrete candidate and request its development test now. Minimize inspection and preserve a usable submission."
        return notice

    def _check_cancel_unlocked(self, worker_id):
        worker = self.workers[worker_id]
        if self.closed.is_set() or time.monotonic() >= self.deadline:
            self._refuse(worker_id, "Paced request deadline or shutdown reached", True)
        session = worker.get("session")
        if session is not None and session.submitted is not None:
            self._refuse(worker_id, "Worker has already submitted", False)

    def _estimate(self, worker_id, body):
        with self.condition:
            decorated = self._append_notice(body, self._budget_notice(worker_id))
        prompt = math.ceil(len(json.dumps(decorated).encode()) / self.prompt_bytes_per_token) + 1024
        generated = min(int(body.get("max_tokens", self.max_output_tokens)), self.max_output_tokens)
        return {"prompt": prompt, "uncached": prompt, "generated": generated}

    def _acquire_pacing(self, worker_id, estimate):
        ticket, started = object(), time.monotonic()
        with self.condition:
            self.pacing_queue.append(ticket)
            try:
                while True:
                    self._check_cancel_unlocked(worker_id)
                    now = time.monotonic()
                    while self.pacing_history and self.pacing_history[0][0] <= now - self.window_seconds:
                        self.pacing_history.popleft()
                    if self.rate_scale < 1 and now >= max(self.cooldown_until, self.last_429_at + self.window_seconds):
                        self.rate_scale = 1.0
                    rates = self._rates_unlocked()
                    hard_rates = {key: min(value, self.header_limits.get(key, value))
                                  for key, value in self.pacing_config.items()}
                    if any(estimate[key] > hard_rates[key] for key in ("prompt", "uncached")):
                        self._refuse(worker_id,
                            f"Request token estimate exceeds a configured pacing ceiling: estimated prompt={estimate['prompt']}, "
                            f"uncached={estimate['uncached']}; limits prompt={hard_rates['prompt']}, uncached={hard_rates['uncached']}", False)
                    estimate = {**estimate, "generated": min(estimate["generated"], max(1, int(rates["generated"])))}
                    used = {key: sum(entry[1][key] for entry in self.pacing_history)
                            + sum(entry[key] for entry in self.pacing_leases.values()) for key in rates}
                    ready = (self.pacing_queue[0] is ticket and now >= self.cooldown_until
                        and now >= self.last_pacing_start + self.start_stagger_seconds
                        and all(used[key] + estimate[key] <= rates[key] for key in rates))
                    if ready:
                        lease = self.next_lease
                        self.next_lease += 1
                        self.pacing_leases[lease] = dict(estimate)
                        self.pacing_queue.popleft()
                        self.last_pacing_start = now
                        self.condition.notify_all()
                        return lease, now - started
                    self.condition.wait(timeout=min(.2, max(.001, self.deadline - now)))
            except BaseException:
                self.pacing_queue.remove(ticket)
                self.condition.notify_all()
                raise

    def _settle_pacing(self, lease, record, *, forwarded):
        with self.condition:
            reserved = self.pacing_leases.pop(lease)
            if forwarded:
                usage = record.get("usage") or {}
                if usage:
                    counts = {"prompt": sum(usage.get(k, 0) for k in
                        ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
                        "uncached": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
                        "generated": usage.get("output_tokens", 0)}
                elif record.get("status_code") == 429:
                    counts = {**reserved, "generated": 0}
                else:
                    counts = reserved
                self.pacing_history.append((time.monotonic(), counts))
            self.condition.notify_all()
            self._save_summary_unlocked()

    def _attempt_transport(self, encoded, timeout):
        if self.raw_transport is not None:
            self.attempt_context.forwarded = True
            result = self.raw_transport(encoded, timeout)
            status, data, content_type = result[:3]
            headers = result[3] if len(result) > 3 else {}
        else:
            import httpx2
            with httpx2.Client(timeout=timeout) as client:
                with self.condition:
                    if self.closed.is_set():
                        raise BudgetRefused("Paced client is closed")
                    self.active_clients.add(client)
                try:
                    from task08_swarm_client import UPSTREAM
                    self.attempt_context.forwarded = True
                    response = client.post(UPSTREAM, content=encoded, headers={
                        "Authorization": "Bearer " + self.keys["FIREWORKS_API_KEY"],
                        "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                    status, data = response.status_code, response.content
                    content_type, headers = response.headers.get("content-type", "application/json"), response.headers
                finally:
                    with self.condition:
                        self.active_clients.discard(client)
        self.attempt_context.headers = safe_rate_headers(headers)
        self.attempt_context.retry_delay = self._apply_headers(status, self.attempt_context.headers, self.attempt_context.retry_number)
        return status, data, content_type

    def _before_accounting(self, record, *, reservation, token_reservation, request_bytes, max_output_tokens):
        if record.get("status_code") == 429:
            tokens = token_reservation or request_bytes + max_output_tokens + 4096
            record.update(estimated_cost_usd=reservation, unknown_usage_token_allowance=tokens,
                          rejected_billing_unknown=True)
            with self.condition:
                self.rejected_cost_allowance += reservation
                self.rejected_token_allowance += tokens

    def _apply_headers(self, status, headers, retry_number):
        delay = None
        with self.condition:
            for key, header in RATE_HEADERS.items():
                try:
                    value = float(headers.get(header, ""))
                    if math.isfinite(value) and value > 0:
                        self.header_limits[key] = min(self.pacing_config[key], value)
                except ValueError:
                    pass
            if status == 429:
                self.rate_limited_attempts += 1
                self.last_429_at = time.monotonic()
                self.rate_scale = max(.6, self.rate_scale * .8)
                delay = retry_after_seconds(headers.get("retry-after"))
                if delay is None:
                    delay = min(self.max_backoff_seconds, self.base_backoff_seconds * 2 ** retry_number)
                    delay += self.random.uniform(0, self.jitter_seconds)
                self.cooldown_until = max(self.cooldown_until, time.monotonic() + delay)
            elif 200 <= status < 300 and time.monotonic() - self.last_429_at >= self.window_seconds:
                self.rate_scale = min(1.0, self.rate_scale + .05)
            self.condition.notify_all()
            return delay

    def request(self, worker_id, body):
        if not isinstance(body, dict):
            raise ValueError("Expected a Messages request object")
        if int(body.get("max_tokens", self.max_output_tokens)) < 1:
            raise ValueError("max_tokens must be positive")
        with self.condition:
            self.logical_active += 1
        try:
            return self._request_logical(worker_id, body)
        finally:
            with self.condition:
                self.logical_active -= 1
                self._save_summary_unlocked()
                self.condition.notify_all()

    def _request_logical(self, worker_id, body):
        with self.condition:
            if worker_id not in self.workers:
                raise ValueError("Unknown worker")
            self._check_cancel_unlocked(worker_id)
            worker = self.workers[worker_id]
            if worker["logical_requests"] >= worker["logical_allowance"]:
                self._refuse(worker_id, "Logical worker request allowance exhausted", False)
            self.logical_requests += 1
            logical_id = self.logical_requests
            worker["logical_requests"] += 1
        for attempt in range(self.max_retries + 1):
            lease, waited = self._acquire_pacing(worker_id, self._estimate(worker_id, body))
            attempt_body = {**body, "max_tokens": self.pacing_leases[lease]["generated"]}
            self.attempt_context.forwarded = False
            self.attempt_context.headers = {}
            self.attempt_context.retry_delay = None
            self.attempt_context.call_index = None
            self.attempt_context.retry_number = attempt
            record = {}
            try:
                result = super().request(worker_id, attempt_body)
                status = result[0]
                headers = self.attempt_context.headers
                with self.condition:
                    index = self.attempt_context.call_index
                    record = next((entry for entry in self.records if entry["call"] == index), {})
                    if record:
                        record.update(logical_request=logical_id, logical_attempt=attempt + 1,
                            safe_response_headers=headers, pacing_wait_seconds=waited,
                            retry_after_seconds=retry_after_seconds(headers.get("retry-after")),
                            applied_retry_delay_seconds=self.attempt_context.retry_delay,
                            pacing_estimated_tokens=dict(self.pacing_leases[lease]),
                            effective_pacing_tpm=self._rates_unlocked(),
                            paced_output_token_cap=attempt_body["max_tokens"],
                            output_cap_reduced=attempt_body["max_tokens"] < min(int(body.get("max_tokens", self.max_output_tokens)), self.max_output_tokens))
                        P.write_json(self.root / "api" / f"metrics-{index:04d}.json", record)
                    self._save_summary_unlocked()
            finally:
                self._settle_pacing(lease, record, forwarded=self.attempt_context.forwarded)
            if status != 429 or attempt >= self.max_retries:
                with self.condition:
                    self.logical_results.append({"logical_request": logical_id, "worker_id": worker_id,
                                                 "provider_attempts": attempt + 1, "status_code": status})
                    P.write_json(self.root / "logical-results.json", self.logical_results)
                return result

    def close(self, *, wait_timeout=10.0):
        started = time.monotonic()
        super().close(wait_timeout=wait_timeout)
        with self.condition:
            while self.logical_active:
                remaining = wait_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Paced logical requests did not settle during shutdown")
                self.condition.wait(min(.1, remaining))
            self._save_summary_unlocked()
