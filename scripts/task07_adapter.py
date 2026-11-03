"""Bounded Task07 public probes and immutable policy artifacts."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import task08_pilot as P

ROOT, REPO = P.ROOT, P.REPO
TASK = REPO / "tasks/task07"
PUBLIC = TASK / "environment/assets"
AGENT_IMAGE = "rlebench-task07-agent:dev"
IMAGE = "rlebench-task07-verifier:dev"
WORKER = ROOT / "scripts/task07_probe.py"
PUBLIC_SEEDS = (35,)
PUBLIC_SECONDS = 10.0
SIMULATOR_CPUS, SIMULATOR_MEMORY = 4, "4g"
ZERO_POLICY = '''import numpy as np

class Policy:
    def reset(self, cell_spec, seed):
        self.home = np.asarray(cell_spec["arm_home"], dtype=float)

    def act(self, obs):
        return np.r_[self.home, 0.0]

def make_policy():
    return Policy()
'''


def local_deadlines(now=None):
    current = now or datetime.now(ZoneInfo("America/Los_Angeles"))
    return current.replace(hour=14, minute=0, second=0, microsecond=0).timestamp(), current.replace(
        hour=15, minute=0, second=0, microsecond=0).timestamp()


def make_starter(destination):
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "policy.py").write_text(ZERO_POLICY)


def evaluate(bundle, output, *, final=False, timeout=900, seeds=PUBLIC_SEEDS,
             episode_seconds=PUBLIC_SECONDS, stop_at=None):
    if stop_at is not None:
        timeout = min(timeout, stop_at - time.time())
    if timeout <= 0:
        raise TimeoutError("Task07 hard stop reached before evaluation")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    name = "task07-pilot-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none",
        "--cpus", str(SIMULATOR_CPUS), "--memory", SIMULATOR_MEMORY, "--pids-limit", "128",
        "--security-opt", "no-new-privileges", "-e", "OPENBLAS_NUM_THREADS=1",
        "-e", "OMP_NUM_THREADS=1", "-e", "MUJOCO_GL=osmesa", "-e", "PYTHONDONTWRITEBYTECODE=1"]
    if final:
        command += ["-v", f"{bundle.resolve()}:/logs/artifacts/policy:ro",
            "-v", f"{output.resolve()}:/logs/verifier", IMAGE, "/tests/test.sh"]
    else:
        command += ["--read-only", "--tmpfs", "/tmp:rw,nosuid,size=1g",
            "-e", "TASK07_PUBLIC_SEEDS=" + ",".join(str(seed) for seed in seeds),
            "-e", f"TASK07_PUBLIC_SECONDS={float(episode_seconds)}",
            "-v", f"{bundle.resolve()}:/submission:ro", "-v", f"{output.resolve()}:/logs",
            "-v", f"{WORKER}:/pilot/probe.py:ro", AGENT_IMAGE, "python3", "/pilot/probe.py"]
    try:
        completed = P.checked(command, timeout=timeout)
        (output / "container.log").write_text(completed.stdout + completed.stderr)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        raise TimeoutError("Task07 evaluation deadline reached; container stopped") from None
    except subprocess.CalledProcessError as error:
        (output / "container.log").write_text((error.stdout or "") + (error.stderr or ""))
        raise RuntimeError("Task07 container failed; operator log retains diagnostics") from None
    return json.loads((output / ("reward.json" if final else "probe.json")).read_text())


def partial_score(report):
    episodes = report.get("episodes", [])
    if len(episodes) != len(PUBLIC_SEEDS) or {row.get("public_seed") for row in episodes} != set(PUBLIC_SEEDS):
        raise ValueError("Public probe must cover the fixed published design seed")
    scores = []
    for row in episodes:
        if row.get("error") or row.get("aborted") or row.get("policy_faults", 0):
            raise ValueError("Public policy load, physics or control failure")
        if row.get("episode_seconds") != PUBLIC_SECONDS or row.get("ticks") != int(PUBLIC_SECONDS * 20):
            raise ValueError("Public probe did not complete its fixed episode horizon")
        fraction = float(row["clear_frac"])
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Invalid measured clearance fraction")
        counts = [row[name] for name in ("floor_drops", "damage_events", "bin_hits")]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Invalid observed event counts")
        scores.append(fraction - .03 * counts[0] - .05 * counts[1] - .05 * counts[2])
    return sum(scores) / len(scores)


def preflight(images=True):
    checks = {"codex_installed": bool(shutil.which("codex")),
        "claude_installed": bool(shutil.which("claude")),
        "public_assets_staged": (PUBLIC / "dev_runner.py").is_file(),
        "codex_auth_present": (Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json").is_file()}
    try:
        P.checked(["docker", "info", "--format", "{{.OSType}}"], timeout=10)
        checks["docker_running"] = True
    except (OSError, subprocess.SubprocessError):
        checks["docker_running"] = False
    if images:
        for image in (AGENT_IMAGE, IMAGE):
            try:
                P.checked(["docker", "image", "inspect", image], timeout=10)
                checks[image] = True
            except (OSError, subprocess.SubprocessError):
                checks[image] = False
    return checks
