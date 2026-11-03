"""Public-input-only paired planning before a separately authorized Task09 run."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import task09_pilot as Q

FIELDS = {"id", "title", "hypothesis", "first_test", "success_criteria"}
SYSTEM = ("You are a planning agent. Use only the supplied public task instruction and starter files. "
    "Propose concrete, distinct, testable hypotheses that satisfy the task. No tools are available. "
    "Return only a JSON object with the key hypotheses, whose value is an array. "
    "Each item must contain exactly id, title, hypothesis, first_test, success_criteria; all values must be nonempty strings. "
    "Do not include Markdown fences or surrounding prose.")


class PlanError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(value).hexdigest()


def strict_json(value):
    def pairs(items):
        result = {}
        for key, entry in items:
            if key in result:
                raise PlanError("Duplicate JSON key")
            result[key] = entry
        return result
    try:
        return json.loads(value, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(PlanError("Nonfinite JSON constant")))
    except (ValueError, TypeError, UnicodeError) as error:
        raise PlanError("Invalid planner JSON") from error


def validate(value, count, *, final=False):
    if not isinstance(value, dict) or set(value) != {"hypotheses"}:
        raise PlanError("Planner must return only a hypotheses array")
    rows = value["hypotheses"]
    if not isinstance(rows, list) or len(rows) != count:
        raise PlanError(f"Planner must return exactly {count} hypotheses")
    seen = {key: set() for key in ("id", "title", "hypothesis")}
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise PlanError("Hypothesis fields do not match the required schema")
        if any(not isinstance(text, str) or not text.strip() or len(text) > 16000 for text in row.values()):
            raise PlanError("Hypothesis fields must be bounded nonempty strings")
        for key in seen:
            normalized = " ".join(row[key].casefold().split())
            if normalized in seen[key]:
                raise PlanError("Hypothesis IDs, titles and hypotheses must be distinct")
            seen[key].add(normalized)
    if final and {row["id"] for row in rows} != {f"hypothesis_{i}" for i in range(1,7)}:
        raise PlanError("Final IDs must be hypothesis_1 through hypothesis_6")
    return rows


def public_inputs(starter):
    starter = Path(starter).resolve()
    instruction = (Q.TASK / "instruction.md").read_bytes()
    files = {}
    hashes = {"instruction.md": digest(instruction)}
    for name in Q.ENTRY_FILES:
        path = starter / name
        if path.is_symlink() or not path.resolve().is_relative_to(starter):
            raise PlanError("Starter entry files must remain within the supplied starter")
        raw = path.read_bytes()
        if len(raw) > 1024**2:
            raise PlanError("Starter entry file exceeds planning input limit")
        files[name] = raw.decode("utf-8")
        hashes[name] = digest(raw)
    source = {"public_task_instruction": instruction.decode("utf-8"), "starter_files": files}
    encoded = json.dumps(source, ensure_ascii=False, sort_keys=True).encode()
    return encoded.decode(), hashes, digest(encoded)


def build_plan(pool, starter, output, deadline):
    started = time.monotonic()
    if not math.isfinite(float(deadline)):
        raise PlanError("Planning deadline must be finite")
    phase_deadline = min(float(deadline), started+180)
    if phase_deadline <= started:
        raise PlanError("Planning deadline has already expired")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source, input_hashes, source_hash = public_inputs(starter)
    if (output / "plan.json").exists():
        raise PlanError("Planning output already contains a plan")
    lanes = {name: pool.lane_for(name) for name in ("planner_a", "planner_b")}
    for name, lane in lanes.items():
        with lane.condition:
            lane.deadline = min(lane.deadline, phase_deadline)
            lane.condition.notify_all()
        lane.register_worker(name, max_calls=3)
    pool.deadline = phase_deadline
    prompts, completed = {}, []
    log_lock = threading.Lock()

    def call(worker, stage, instruction, count, *, final=False):
        if time.monotonic() >= phase_deadline:
            raise PlanError("Planning phase deadline reached")
        body = {"max_tokens": 8192, "stream": False, "system": SYSTEM,
            "messages": [{"role": "user", "content": source+"\n\n"+instruction}]}
        encoded = json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        with log_lock:
            prompts[stage] = digest(encoded)
            Q.write_json(output / f"{stage}-request.json", body)
        status, response, content_type = lanes[worker].request(worker, body)
        (output / f"{stage}-response.json").write_bytes(response)
        if not 200 <= status < 300 or "json" not in content_type:
            raise PlanError(f"Planner {stage} failed with HTTP status {status}")
        envelope = strict_json(response)
        blocks = envelope.get("content") if isinstance(envelope, dict) else None
        if not isinstance(blocks, list):
            raise PlanError("Planner response omitted content blocks")
        text = "".join(block["text"] for block in blocks
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str))
        parsed = strict_json(text)
        validate(parsed, count, final=final)
        with log_lock:
            completed.append({"stage": stage, "worker": worker, "response_sha256": digest(response)})
        return parsed

    executor = ThreadPoolExecutor(max_workers=2)
    futures = []
    try:
        for worker in ("planner_a", "planner_b"):
            futures.append(executor.submit(call, worker, worker+"_proposal",
                "Independently propose exactly three distinct hypotheses. Assign unique proposal IDs.", 3))
        proposals = [future.result(timeout=max(.001,phase_deadline-time.monotonic())) for future in futures]
        both = json.dumps({"proposal_a": proposals[0], "proposal_b": proposals[1]}, ensure_ascii=False)
        merged = call("planner_a", "planner_a_merge", "Review both independent proposals below. "
            "Merge them into exactly six distinct hypotheses, preserving useful differences and resolving duplication. "
            "Assign IDs hypothesis_1 through hypothesis_6.\n"+both, 6, final=True)
        approved = call("planner_b", "planner_b_review", "Review both proposals and the merged plan below. "
            "Approve by returning the six merged hypotheses, or revise them to ensure six substantively distinct, "
            "legal and testable hypotheses. Return the complete six-item plan with IDs hypothesis_1 through hypothesis_6.\n"
            +both+"\nMerged plan:\n"+json.dumps(merged,ensure_ascii=False), 6, final=True)
        if time.monotonic() >= phase_deadline:
            raise PlanError("Planning phase exceeded its deadline")
        plan = {"hypotheses": approved["hypotheses"], "provenance": {
            "input_sha256": input_hashes, "combined_public_input_sha256": source_hash,
            "source_prompt_sha256": prompts, "completed_stages": completed,
            "maximum_phase_seconds": 180, "elapsed_seconds": time.monotonic()-started,
            "planner_logical_request_limit_each": 3, "planner_logical_requests_each": 2}}
        Q.write_json(output / "plan.json", plan)
        return plan
    except BaseException as error:
        for lane in set(lanes.values()):
            with lane.condition:
                lane.deadline = min(lane.deadline,time.monotonic())
                clients = list(lane.active_clients)
                lane.condition.notify_all()
            for client in clients:
                try:
                    client.close()
                except Exception:
                    pass
        for future in futures:
            future.cancel()
        Q.write_json(output / "planning-error.json", {"error": type(error).__name__,
            "completed_stages": completed, "source_prompt_sha256": prompts, "input_sha256": input_hashes})
        if isinstance(error, PlanError):
            raise
        raise PlanError(f"Planning failed: {type(error).__name__}") from error
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
