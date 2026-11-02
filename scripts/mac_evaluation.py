"""Minimal custom hosted API adapter using RAX's DexJoCo process and tool contract.

This is a short RGB/proprioception smoke evaluation, not RAX formal benchmarking.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rax"))
os.environ.setdefault("DEXJOCO_SOURCE", str(ROOT / "dexjoco"))
os.environ.setdefault("DEXJOCO_PYTHON", str(ROOT / ".venv/bin/python"))

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from services.dexjoco import tools as T
from services.dexjoco.process import DexjocoProcess
from services.dexjoco.session import hold_row

PROVIDERS = {
    "kimi": {"model": "kimi-k3.1-pro", "base_url": "https://api.moonshot.ai/v1",
             "key_env": "MOONSHOT_API_KEY",
             "docs": "https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart"},
    "deepseek": {"model": "deepseek-flash", "base_url": "https://api.deepseek.com",
                 "key_env": "DEEPSEEK_API_KEY", "docs": "https://api-docs.deepseek.com/guides/vision/"},
}
CAMERAS = ("front", "wrist")
KEY_PATH = Path.home() / ".config/kimirax/api_keys.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_keys():
    keys = {}
    if KEY_PATH.exists():
        if KEY_PATH.is_symlink() or stat.S_IMODE(KEY_PATH.stat().st_mode) & 0o077:
            raise ValueError("Credential file must be a regular owner-only file (chmod 600).")
        keys.update(json.loads(KEY_PATH.read_text()))
    for name in ("MOONSHOT_API_KEY", "DEEPSEEK_API_KEY", "FIREWORKS_API_KEY"):
        if os.environ.get(name):
            keys[name] = os.environ[name]
    if not keys.get("MOONSHOT_API_KEY") and os.environ.get("KIMI_API_KEY"):
        keys["MOONSHOT_API_KEY"] = os.environ["KIMI_API_KEY"]
    return keys


def redact(value, keys):
    text = json.dumps(value, allow_nan=False)
    for secret in keys.values():
        if isinstance(secret, str) and secret:
            text = text.replace(secret, "[REDACTED]")
    return json.loads(text)


def save_observation(obs, directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    state = np.asarray(obs["state"], dtype=float)
    if state.shape != (23,) or not np.isfinite(state).all():
        raise ValueError("Expected exactly 23 finite robot proprioception values")
    safe = {"state": state.tolist(), "instruction": str(obs["instruction"]), "step_id": int(obs["step_id"])}
    images = {}
    for camera in CAMERAS:
        pixels = np.asarray(obs[camera])
        if pixels.shape != (640, 640, 3) or pixels.dtype != np.uint8 or np.std(pixels) < 1:
            raise ValueError(f"Invalid or blank {camera} camera frame")
        path = directory / f"{label}_{camera}.png"
        Image.fromarray(pixels).save(path)
        images[camera] = {"path": str(path.relative_to(directory.parent)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(directory / f"{label}.json", {**safe, "images": images})
    return safe, images


def hand_limits():
    path = ROOT / "dexjoco/dexjoco/dexjoco/sim/envs/xmls/wonik_allegro/right_hand.xml"
    xml = ET.parse(path).getroot()
    by_name = {item.attrib["name"]: list(map(float, item.attrib["range"].split()))
               for item in xml.iter("joint") if "name" in item.attrib and "range" in item.attrib}
    return np.asarray([by_name[f"{name[:-1]}j{name[-1]}"] for name in T.HAND_JOINTS])


def validate_bounded_actions(actions, state, remaining, max_rows=20):
    T.validate_request(T.STEP, {"actions": actions}, action_dim=22)
    if len(actions) > min(max_rows, remaining):
        raise ValueError("Action chunk exceeds remaining budget or 20-row limit")
    limits = hand_limits()
    previous = np.asarray(hold_row(state, False))
    for row in np.asarray(actions, dtype=float):
        if np.any(row[:3] < [-0.85, -0.85, 0.55]) or np.any(row[:3] > [0.85, 0.85, 1.8]):
            raise ValueError("Flange target is outside configured workspace")
        if np.linalg.norm(row[3:6]) > np.pi + 1e-6:
            raise ValueError("Rotation vector must have norm <= pi")
        if np.any(row[6:] < limits[:, 0]) or np.any(row[6:] > limits[:, 1]):
            raise ValueError("Hand target exceeds official joint limits")
        if np.linalg.norm(row[:3] - previous[:3]) > 0.025 + 1e-9:
            raise ValueError("Translation exceeds 0.025 metres per step")
        angular = (Rotation.from_rotvec(previous[3:6]).inv() * Rotation.from_rotvec(row[3:6])).magnitude()
        if angular > 0.25 + 1e-9 or np.max(np.abs(row[6:] - previous[6:])) > 0.25 + 1e-9:
            raise ValueError("Rotation or hand target exceeds 0.25 radians per step")
        previous = row


def new_sim(directory, run_id, layout, collection=1):
    sim = DexjocoProcess(output_root=directory / "rax")
    try:
        sim.start(task="fold_glasses", collection=collection, layout_id=layout, run_id=run_id,
                  sim_gpu="mac", include_depth=False, episode_timeout_seconds=1200)
    except BaseException:
        sim.stop()
        raise
    return sim


def outcome(sim):
    return json.loads((sim.output_dir / "sim/evaluation_outcome.json").read_text())


def smoke(directory, layout):
    report = {"mode": "no_model_calls", "passed": False, "layout": layout, "runs": []}
    initial = []
    try:
        for repeat in range(2):
            sim = new_sim(directory, f"reset-{repeat}", layout)
            try:
                obs = sim.request("observe")
                safe, images = save_observation(obs, directory / "images", f"reset-{repeat}")
                initial.append((safe["state"], images))
                held = sim.request("hold", steps=30)
                save_observation(held["frames"][-1], directory / "images", f"hold-{repeat}")
                write_json(directory / f"hold-{repeat}-actions.json", held["steps"])
                assert held["step_id"] == 30 and not held["terminated"]
                assert np.isfinite(held["frames"][-1]["state"]).all()
                row = np.asarray(hold_row(held["frames"][-1]["state"], False))
                actions = []
                for index in range(3):
                    target = row.copy()
                    target[2] -= 0.001 * (index + 1)
                    actions.append(target.tolist())
                validate_bounded_actions(actions, held["frames"][-1]["state"], remaining=3)
                moved = sim.request("chunk_step", actions=actions)
                assert moved["step_id"] == 33 and not moved["terminated"]
                save_observation(moved["frames"][-1], directory / "images", f"bounded-action-{repeat}")
                write_json(directory / f"bounded-action-{repeat}.json", moved["steps"])
                sim.finish("no_model_smoke")
                report["runs"].append({"reset": repeat, "state_dim": 23,
                                       "camera_shapes": {c: [640, 640, 3] for c in CAMERAS},
                                       "hold_steps": 30, "bounded_action_steps": 3, "official_outcome": outcome(sim)})
            finally:
                sim.stop()
        report["reset_state_max_difference"] = float(np.max(np.abs(np.asarray(initial[0][0]) - initial[1][0])))
        report["reset_camera_hashes_match"] = all(initial[0][1][c]["sha256"] == initial[1][1][c]["sha256"] for c in CAMERAS)
        assert report["reset_state_max_difference"] < 1e-10
        assert report["reset_camera_hashes_match"]
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_json(directory / "smoke-result.json", report)
    return report


def api_tools():
    result = []
    for definition in T.tool_definitions():
        if definition["name"] not in (T.STEP, T.HOLD, T.STATUS, T.OBSERVE):
            continue
        schema = definition["inputSchema"]
        description = definition["description"]
        if definition["name"] == T.STEP:
            schema["properties"]["actions"]["maxItems"] = 20
            schema["properties"]["actions"]["items"].update(minItems=22, maxItems=22)
            description = "Execute 1..20 absolute 22-D action rows, using the RAX single-arm policy layout."
        elif definition["name"] == T.HOLD:
            schema["properties"]["steps"]["maximum"] = 20
            description = "Hold using the official stay() for 1..20 control steps."
        elif definition["name"] == T.OBSERVE:
            description = "Read current RGB camera images, 23-value robot proprioception and instruction. No physics."
        result.append({"type": "function", "function": {"name": definition["name"],
                       "description": description, "parameters": schema}})
    return result


def system_prompt(args):
    return (
        "Control the simulated Franka Panda and Allegro hand to fold both glasses temples and place the glasses in the case. "
        "Use only the supplied front/wrist RGB images and robot proprioception. Make one tool call per response. "
        "State layout: world flange xyz metres, quaternion wxyz, 16 hand joint angles in radians. "
        "Action layout: absolute world flange xyz, world rotation vector xyz radians, 16 hand joint targets. "
        f"Hand joint order: {list(T.HAND_JOINTS)}; inclusive limits: {hand_limits().tolist()}. "
        "Position bounds: x/y [-0.85,0.85], z [0.55,1.8] metres; rotation-vector norm <= pi. "
        "Per step, target translation may change by at most 0.025m; geodesic orientation and each hand joint by at most 0.25rad. "
        "Chunks contain 1..20 rows. The first row is compared with current measured robot state. "
        "Invalid actions are rejected in full without advancing physics. Use small changes and inspect images. "
        f"Budget: {args.max_calls} model calls, {args.max_steps} control steps, {args.max_output_tokens} output tokens per call, "
        f"{args.max_total_tokens} total reported input/output tokens. Control rate 30Hz, physics advances 0.02s per step. "
        "Physics pauses during API inference, following RAX synchronous evaluation. "
        "These are short test budgets; do not claim success without the official termination signal."
    )


def user_message(safe, images, directory, remaining, calls_left):
    content = [{"type": "text", "text": json.dumps({**safe, "remaining_steps": remaining, "remaining_model_calls": calls_left})}]
    for camera in CAMERAS:
        path = directory / images[camera]["path"]
        content.extend([{"type": "text", "text": f"{camera} camera"},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}}])
    return {"role": "user", "content": content}


def episode(provider, directory, args, keys):
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    result = {"provider": provider, "requested_model": cfg["model"], "base_url": cfg["base_url"],
              "layout": args.layout, "seed": args.layout // 50, "official_episode_index": args.layout % 50,
              "budget": {"max_calls": args.max_calls, "max_steps": args.max_steps,
                         "max_output_tokens": args.max_output_tokens, "max_total_tokens": args.max_total_tokens,
                         "request_timeout_seconds": args.timeout},
              "model_calls": 0, "api_latency_seconds": [], "reported_total_tokens": 0,
              "usage": [], "errors": [], "success": None, "status": "starting"}
    directory.mkdir(parents=True, exist_ok=False)
    sim = None
    started = time.monotonic()
    try:
        sim = new_sim(directory, provider, args.layout)
        obs = sim.request("observe")
        safe, images = save_observation(obs, directory / "images", "initial")
        result["initial_camera_hashes"] = {c: images[c]["sha256"] for c in CAMERAS}
        result["initial_robot_state_sha256"] = hashlib.sha256(np.asarray(safe["state"], dtype="<f8").tobytes()).hexdigest()
        key = keys.get(cfg["key_env"])
        if not key:
            result.update(status="blocked_missing_key", errors=[{"type": "MissingCredential", "key_env": cfg["key_env"]}])
            return result
        client = OpenAI(api_key=key, base_url=cfg["base_url"], max_retries=0, timeout=args.timeout)
        messages = [{"role": "system", "content": system_prompt(args)}]
        reason = "model_call_budget"
        for turn in range(args.max_calls):
            remaining = args.max_steps - sim.step_id
            if remaining <= 0:
                reason = "control_step_budget"
                break
            if result["reported_total_tokens"] >= args.max_total_tokens:
                reason = "reported_token_budget"
                break
            messages.append(user_message(safe, images, directory, remaining, args.max_calls - turn))
            params = {"model": cfg["model"], "messages": messages, "tools": api_tools(),
                      "reasoning_effort": "low", "stream": False}
            if provider == "kimi":
                params["max_completion_tokens"] = args.max_output_tokens
            else:
                params.update(max_tokens=args.max_output_tokens, extra_body={"thinking": {"type": "enabled"}})
            write_json(directory / f"request-{turn:02d}.json", redact(params, keys))
            call_start = time.monotonic()
            result["model_calls"] += 1
            try:
                response = client.chat.completions.create(**params)
            except Exception as exc:
                elapsed = time.monotonic() - call_start
                error = {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", None),
                         "message": str(exc), "latency_seconds": elapsed}
                result["errors"].append(redact(error, keys))
                result["api_latency_seconds"].append(elapsed)
                reason = "api_error"
                break
            elapsed = time.monotonic() - call_start
            result["api_latency_seconds"].append(elapsed)
            raw = response.model_dump(mode="json")
            write_json(directory / f"response-{turn:02d}.json", redact({"latency_seconds": elapsed, "response": raw}, keys))
            usage = raw.get("usage") or {}
            result["usage"].append(usage)
            result["reported_total_tokens"] += int(usage.get("total_tokens") or 0)
            result["served_model"] = raw.get("model")
            choice = response.choices[0]
            assistant = choice.message.model_dump(mode="json", exclude_unset=True)
            messages.append(assistant)
            calls = choice.message.tool_calls or []
            if not calls:
                reason = "output_token_limit" if choice.finish_reason == "length" else "model_stopped"
                break
            feedback = {"accepted": False}
            if len(calls) != 1:
                feedback["error"] = "Exactly one tool call per response is required; no action executed"
            else:
                call = calls[0]
                try:
                    arguments = json.loads(call.function.arguments)
                    name = call.function.name
                    T.validate_request(name, arguments, action_dim=22)
                    if name == T.STEP:
                        validate_bounded_actions(arguments["actions"], safe["state"], remaining)
                        packet = sim.request("chunk_step", actions=arguments["actions"])
                    elif name == T.HOLD:
                        if arguments["steps"] > min(20, remaining):
                            raise ValueError("Hold exceeds remaining budget or 20-step limit")
                        packet = sim.request("hold", steps=arguments["steps"])
                    elif name in (T.STATUS, T.OBSERVE):
                        packet = None
                    else:
                        raise ValueError("Tool is outside hosted adapter allowlist")
                    feedback = {"accepted": True, "step_id": sim.step_id}
                    if name == T.STATUS:
                        feedback.update(action_layout=T.action_layout(False), state_layout=T.state_layout(False),
                                        remaining_steps=remaining, instruction=safe["instruction"], cameras=list(CAMERAS))
                    if packet:
                        write_json(directory / f"actions-{turn:02d}.json", packet["steps"])
                        for frame_id, frame in enumerate(packet["frames"]):
                            save_observation(frame, directory / "images", f"turn-{turn:02d}-frame-{frame_id:02d}")
                        feedback.update(terminated=packet["terminated"], success=packet.get("success"))
                except Exception as exc:
                    feedback["error"] = f"{type(exc).__name__}: {exc}"
                    result["errors"].append({"turn": turn, **feedback})
            write_json(directory / f"tool-result-{turn:02d}.json", redact(feedback, keys))
            for call in calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(feedback)})
            obs = sim.request("observe")
            safe, images = save_observation(obs, directory / "images", f"turn-{turn:02d}-final")
            if feedback.get("terminated"):
                reason = "official_termination"
                break
        result.update(status="api_error" if reason == "api_error" else "finished_short_episode", stop_reason=reason)
    except Exception as exc:
        result.update(status="error", errors=[*result["errors"], redact({"type": type(exc).__name__, "message": str(exc)}, keys)])
    finally:
        if sim:
            try:
                sim.finish(result.get("stop_reason", result["status"]))
                result["official_outcome"] = outcome(sim)
                result["control_steps"] = sim.step_id
                if result["usage"]:
                    result["success"] = bool(result["official_outcome"]["native_score"] == 1)
            except Exception as exc:
                result["errors"].append(redact({"type": type(exc).__name__, "message": str(exc)}, keys))
            finally:
                sim.stop()
        result["wall_seconds"] = time.monotonic() - started
        write_json(directory / "result.json", redact(result, keys))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("smoke", "evaluate", "credentials"))
    p.add_argument("--provider", choices=("kimi", "deepseek", "both"), default="both")
    p.add_argument("--layout", type=int, default=0, choices=range(150))
    p.add_argument("--max-calls", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--max-output-tokens", type=int, default=4096)
    p.add_argument("--max-total-tokens", type=int, default=80000)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not (1 <= args.max_calls <= 20 and 1 <= args.max_steps <= 1200 and 1 <= args.max_output_tokens <= 8192
            and 1 <= args.max_total_tokens <= 200000 and 1 <= args.timeout <= 180):
        p.error("Budget outside supported bounds")
    keys = load_keys()
    if args.mode == "credentials":
        print(json.dumps({name: "present" if keys.get(cfg["key_env"]) else "missing" for name, cfg in PROVIDERS.items()}))
        return
    directory = (args.output or ROOT / "outputs/mac-evaluation" / f"{args.mode}-{time.strftime('%Y%m%dT%H%M%S')}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "manifest.json", {"adapter": "minimal_custom_hosted_api_adapter", "platform": platform.platform(),
                "python": sys.version, "rax_commit": subprocess.check_output(["git", "-C", str(ROOT / "rax"), "rev-parse", "HEAD"], text=True).strip(),
                "dexjoco_commit": subprocess.check_output(["git", "-C", str(ROOT / "dexjoco"), "rev-parse", "HEAD"], text=True).strip(),
                "providers": PROVIDERS, "physics_modified": False, "model_inputs": ["front_rgb", "wrist_rgb", "robot_state_23", "instruction", "tool_feedback"],
                "short_episode_not_official_benchmark": True, "token_budget_note": "Reported token budget is checked between calls and may be exceeded by the last request."})
    checked = smoke(directory / "smoke", args.layout)
    print(json.dumps({"smoke_passed": checked["passed"], "output": str(directory)}), flush=True)
    if not checked["passed"]:
        raise SystemExit(1)
    if args.mode == "smoke":
        return
    providers = list(PROVIDERS) if args.provider == "both" else [args.provider]
    results = [episode(provider, directory / provider, args, keys) for provider in providers]
    compare = {"results": results, "same_initial_images": None, "same_initial_robot_state": None}
    if len(results) == 2 and all("initial_camera_hashes" in r for r in results):
        compare["same_initial_images"] = results[0]["initial_camera_hashes"] == results[1]["initial_camera_hashes"]
        compare["same_initial_robot_state"] = results[0]["initial_robot_state_sha256"] == results[1]["initial_robot_state_sha256"]
    write_json(directory / "comparison.json", redact(compare, keys))
    for result in results:
        print(json.dumps({k: result.get(k) for k in ("provider", "requested_model", "status", "success", "model_calls", "control_steps", "reported_total_tokens")}), flush=True)
    if any(r["status"] in ("blocked_missing_key", "api_error", "error") for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    os.umask(0o077)
    main()
