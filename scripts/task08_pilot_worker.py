"""Private Docker-side development probe for the Task 08 pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, "/tests")


def main():
    from harness.base_design.checkpoints import analyze_model, evaluate
    from harness.base_design.controlled_pick import run_target
    from harness.base_design.pickscene import scored_target_schedule
    from harness.sim.arm_variants import trusted_arm_specs
    from harness.sim.mecanum import make_base_controller
    from harness.sim.runner import run_all
    from harness.sim.scenarios import Envelope

    import mujoco

    if mujoco.__version__ != "3.5.0":
        raise RuntimeError("Task 08 requires MuJoCo 3.5.0")
    os.makedirs("/logs/verifier", mode=0o700, exist_ok=True)
    os.chmod("/logs/verifier", 0o700)
    staged = Path("/tmp/pilot-submission")
    staged.mkdir()
    for name in ("robot.xml", "controller.py"):
        shutil.copyfile(Path("/submission") / name, staged / name)
    shutil.copytree("/tests/models/assets", staged / "assets")
    reference = "/tests/models/assets/franka_emika_panda/panda_nohand.xml"
    reports = {}
    for name, arm in trusted_arm_specs(reference).items():
        hidden = run_all(str(staged / "robot.xml"), make_base_controller,
                         seeds=(11,), scenarios=("A", "B"), env=Envelope(), arm=arm)
        info = analyze_model(str(staged / "robot.xml"), arm)
        shelf = []
        if hidden["validity"]["loads"]:
            schedule = scored_target_schedule(0.40, arm)
            for index in (0, 4, 8):
                _, target_name, target = schedule[index]
                for payload in (1.0, 2.0):
                    result = run_target(str(staged / "robot.xml"), 0.40, arm,
                                        target, target_name, payload, str(staged / "controller.py"))
                    shelf.append({k: result.get(k) for k in (
                        "ok", "reason", "target_name", "payload_kg", "ee_err",
                        "min_fasm_hold", "shelf_force_max", "controller_fault")})
        picks = {"score": sum(r["ok"] for r in shelf if r["payload_kg"] == 1) / 3,
                 "margin_score": sum(r["ok"] for r in shelf if r["payload_kg"] == 2) / 3}
        checkpoints = evaluate(hidden, info, picks)
        reports[name] = {
            "checkpoints": {k: v for k, v in checkpoints.items()
                            if not k.startswith("S5.") or k == "S5.testB"},
            "shelf": shelf,
            "design": info.get("design_efficiency"),
            "load_error": hidden["validity"].get("load_error"),
        }
    output = {"kind": "development_probe_not_official_score", "seed": 11,
              "scenarios": ["A", "B"], "target_indices": [0, 4, 8], "arms": reports}
    Path("/logs/verifier/probe.json").write_text(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        Path("/logs/verifier/probe.json").write_text(json.dumps({
            "kind": "development_probe_not_official_score", "error": f"{type(error).__name__}: {error}"}))
