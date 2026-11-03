"""Public Task09 probe with one CPU and one GiB per parallel container."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

import task09_pilot as Q

P = Q.P
WORKER = Q.ROOT / "scripts/task09_followup_probe.py"
AGENT_IMAGE, IMAGE = Q.AGENT_IMAGE, Q.IMAGE
CPUS, MEMORY = 1, "1g"
_FINAL_EVALUATE = Q.evaluate


def hard_stop_timestamp():
    return datetime.now(ZoneInfo("America/Los_Angeles")).replace(
        hour=15, minute=0, second=0, microsecond=0).timestamp() - 30


def task09_container_ids(*, timeout=5):
    completed = subprocess.run(["docker", "ps", "--format", "{{json .}}"],
        capture_output=True, text=True, check=True, timeout=timeout)
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    return {str(row["ID"]) for row in rows if re.search(r"task0?9(?:[^0-9]|$)",
        str(row.get("Names", "")) + " " + str(row.get("Image", "")))}


Task09_container_ids = task09_container_ids


def evaluate(bundle, output, *, final=False, timeout=900):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Probe timeout must be positive and finite")
    timeout = min(float(timeout), hard_stop_timestamp() - time.time())
    if timeout <= 0:
        raise TimeoutError("Task09 hard stop reached before evaluation")
    bundle, output = Path(bundle), Path(output)
    if final:
        return _FINAL_EVALUATE(bundle, output, final=True, timeout=timeout)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    name = "task09-parallel-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
        "--cpus", str(CPUS), "--memory", MEMORY, "--memory-swap", MEMORY,
        "--pids-limit", "128", "--tmpfs", "/tmp:rw,nosuid,size=256m",
        "--security-opt", "no-new-privileges", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
        "-e", "MUJOCO_GL=disable", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{bundle.resolve()}:/submission:ro", "-v", f"{output.resolve()}:/logs",
        "-v", f"{WORKER}:/pilot/probe.py:ro", AGENT_IMAGE, "python3", "/pilot/probe.py"]
    try:
        completed = P.checked(command, timeout=timeout)
        (output / "container.log").write_text(completed.stdout + completed.stderr)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10, check=False)
        raise TimeoutError("Task09 public probe timeout; owned container stopped") from None
    except subprocess.CalledProcessError as error:
        (output / "container.log").write_text((error.stdout or "") + (error.stderr or ""))
        raise RuntimeError("Task09 public probe failed; operator container.log retains diagnostics") from None
    return json.loads((output / "probe.json").read_text())
