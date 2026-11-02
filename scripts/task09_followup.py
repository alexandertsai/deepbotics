"""Public Task 09 adapter with headroom feedback for a separate follow-up."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import task09_pilot as Q

P = Q.P
ROOT, REPO, TASK, PUBLIC = Q.ROOT, Q.REPO, Q.TASK, Q.PUBLIC
IMAGE, AGENT_IMAGE, MODEL = Q.IMAGE, Q.AGENT_IMAGE, Q.MODEL
VARIANTS, ENTRY_FILES, ZERO_TRIM = Q.VARIANTS, Q.ENTRY_FILES, Q.ZERO_TRIM
WORKER = ROOT / "scripts/task09_followup_probe.py"
make_starter, preflight, prepare_assets = Q.make_starter, Q.preflight, Q.prepare_assets
launch, write_json = Q.launch, Q.write_json


def evaluate(bundle, output, *, final=False, timeout=900):
    if final:
        return Q.evaluate(bundle, output, final=True, timeout=timeout)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    name = "task09-followup-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
        "--cpus", "4", "--memory", "4g", "--pids-limit", "128", "--tmpfs", "/tmp:rw,nosuid,size=1g",
        "--security-opt", "no-new-privileges", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
        "-e", "MUJOCO_GL=disable", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{bundle.resolve()}:/submission:ro", "-v", f"{output.resolve()}:/logs",
        "-v", f"{WORKER}:/pilot/probe.py:ro", AGENT_IMAGE, "python3", "/pilot/probe.py"]
    try:
        completed = P.checked(command, timeout=timeout)
        (output / "container.log").write_text(completed.stdout + completed.stderr)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
        raise TimeoutError("Evaluation wall-time budget exhausted") from None
    except subprocess.CalledProcessError as error:
        (output / "container.log").write_text((error.stdout or "") + (error.stderr or ""))
        raise RuntimeError("Evaluation container failed; operator log records the error") from None
    return json.loads((output / "probe.json").read_text())


class Session(Q.Session):
    def make_prompt(self):
        return super().make_prompt() + (
            "\nThe follow-up public probe preserves hold errors, servo saturation, and passive residuals. "
            "It additionally reports nominal.servo_headroom_mean and adapted.servo_headroom_mean. "
            "Each is the mean across public hold steps and joints of 1-abs(clipped applied servo torque)/0.35, "
            "then averaged over the public holds. Values are in [0,1]; higher means more remaining servo capacity. "
            "Passive residual and hold error remain important; a low hold error alone does not demonstrate balanced hardware.\n")

    def definitions(self):
        definitions = super().definitions()
        for tool in definitions:
            if tool["name"] == "test_candidate":
                tool["description"] = "Freeze all variants and spend one public nominal probe including passive residual and servo headroom."
        return definitions

    def dispatch(self, name, arguments):
        if name != "test_candidate":
            return super().dispatch(name, arguments)
        if self.submitted is not None or time.monotonic() >= self.deadline:
            raise ValueError("Development has ended")
        if self.tests >= self.max_tests:
            raise ValueError("Development probe budget exhausted")
        index = self.tests
        self.tests += 1
        bundle = self.root / "versions" / str(index)
        bundle.parent.mkdir(exist_ok=True)
        manifest = self.freeze(bundle)
        P.write_json(self.root / "versions" / f"{index}.json",
            {"manifest": manifest, "hypothesis": arguments["hypothesis"]})
        report = evaluate(bundle, self.root / "probes" / str(index),
            timeout=max(1, min(900, self.deadline - time.monotonic())))
        return {"version": index, "manifest": manifest["sha256"], "report": report}
