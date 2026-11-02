"""Public nominal development probe for the Task 09 pilot."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

VARIANTS = ("franka", "ur5e", "xarm7")


def runtime(variant):
    sys.path.insert(0, f"/workspace/{variant}")
    from harness import scenarios, spec
    from harness.torques import residual_active_torque
    return scenarios, spec, residual_active_torque


def child_sample(variant, directory, request):
    def drop():
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    completed = subprocess.run(
        [sys.executable, "-I", __file__, "--sample", variant, str(directory)],
        input=json.dumps(request), capture_output=True, text=True, timeout=45,
        preexec_fn=drop, env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MUJOCO_GL": "disable"})
    if completed.returncode or len(completed.stdout) > 1024**2:
        raise ValueError("trim sampling failed or exceeded output limit")
    return json.loads(completed.stdout)


def sample(variant, directory):
    scenarios, _, _ = runtime(variant)
    request = json.load(sys.stdin)
    model, _ = scenarios.compose_lead(str(directory / "lead.xml"))
    module_spec = importlib.util.spec_from_file_location("submitted_trim", directory / "trim.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if request["kind"] == "plan":
        poses = np.asarray(module.plan_probe(model, tuple(np.array(q) for q in request["sample"]),
            np.array(request["lower"]), np.array(request["upper"])), dtype=float)
        if poses.shape != (15, model.nv) or not np.isfinite(poses).all():
            raise ValueError("plan_probe must return 15 finite poses")
        output = {"poses": poses.tolist()}
    else:
        function = module.make_trim(model)
        if request["kind"] == "adapted":
            probe = [(np.array(a), np.array(b)) for a, b in request["probe"]]
            params = module.adapt(model, probe)
            function = module.make_trim_adapted(model, params)
        rows = np.asarray([function(np.asarray(q)) for q in request["qmat"]], dtype=float)
        if rows.shape != (len(request["qmat"]), model.nv) or not np.isfinite(rows).all():
            raise ValueError("trim must return one finite torque per joint")
        output = {"torques": rows.tolist()}
    print(json.dumps(output, allow_nan=False))


def hold(model, target, torque, seconds=1.2):
    data = mujoco.MjData(model)
    data.qpos[:] = target
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    saturated = 0
    steps = round(seconds / model.opt.timestep)
    for _ in range(steps):
        command = 8.0 * (target - data.qpos) - 0.4 * data.qvel + torque
        saturated += int(np.any(np.abs(command) > 0.35))
        data.qfrc_applied[:] = np.clip(command, -0.35, 0.35)
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        raise ValueError("non-finite simulation state")
    return data.qpos.copy(), saturated / steps


def variant_probe(variant):
    scenarios, spec, residual_torque = runtime(variant)
    directory = Path("/submission") / variant
    model, data = scenarios.compose_lead(str(directory / "lead.xml"))
    path = scenarios.canonical_path()
    qmat = np.vstack([np.array(spec.LEAD_HOME), path[len(path) // 3], path[2 * len(path) // 3]])
    grid = scenarios.workspace_grid(levels=2)
    residuals = np.asarray([residual_torque(model, data, q) for q in grid])
    result = {"kind": "public_nominal_probe_not_official_score", "variant": variant,
        "mass_kg": float(np.sum(model.body_mass)), "mass_budget_kg": 2.30,
        "passive_residual_rms_nm": float(np.sqrt(np.mean(residuals**2))),
        "passive_residual_max_nm": float(np.max(np.abs(residuals))),
        "tested_hold_poses": qmat.tolist(), "samples": len(grid)}
    nominal = child_sample(variant, directory, {"kind": "nominal", "qmat": qmat.tolist()})
    home = np.asarray(spec.LEAD_HOME)
    lower = np.maximum(home - np.asarray(spec.WORKSPACE_HALFWIDTH), model.jnt_range[:, 0])
    upper = np.minimum(home + np.asarray(spec.WORKSPACE_HALFWIDTH), model.jnt_range[:, 1])
    measured, _ = hold(model, home, np.zeros(model.nv))
    probe = [(home.tolist(), measured.tolist())]
    try:
        poses = np.asarray(child_sample(variant, directory, {"kind": "plan", "sample": probe[0],
            "lower": lower.tolist(), "upper": upper.tolist()})["poses"])
        if np.any(poses < lower - 1e-9) or np.any(poses > upper + 1e-9):
            raise ValueError("calibration poses are outside the allowed workspace")
        for pose in poses:
            measured, _ = hold(model, pose, np.zeros(model.nv))
            probe.append((pose.tolist(), measured.tolist()))
        adapted = child_sample(variant, directory, {"kind": "adapted", "qmat": qmat.tolist(), "probe": probe})
        result["probe_measurements"] = len(probe)
    except Exception as error:
        result["adaptation_error"] = f"{type(error).__name__}: {error}"
        adapted = None
    for label, samples in (("nominal", nominal), ("adapted", adapted)):
        if samples is None:
            continue
        errors, saturation = [], []
        for target, torque in zip(qmat, np.asarray(samples["torques"])):
            measured, share = hold(model, target, np.clip(torque, -0.35, 0.35))
            errors.append(float(np.sqrt(np.mean((measured - target)**2))))
            saturation.append(share)
        result[label] = {"hold_rms_error_rad": float(np.mean(errors)),
            "hold_max_error_rad": max(errors), "servo_saturation_fraction": float(np.mean(saturation))}
    print(json.dumps(result, allow_nan=False))


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--sample":
        sample(sys.argv[2], Path(sys.argv[3]))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--variant":
        variant_probe(sys.argv[2])
        return
    reports = {}
    for variant in VARIANTS:
        try:
            completed = subprocess.run([sys.executable, __file__, "--variant", variant],
                capture_output=True, text=True, timeout=180)
            if completed.returncode:
                raise ValueError(completed.stderr[-1000:])
            reports[variant] = json.loads(completed.stdout)
        except Exception as error:
            reports[variant] = {"error": f"{type(error).__name__}: {error}"}
    Path("/logs/probe.json").write_text(json.dumps({"kind": "public_nominal_probe_not_official_score",
        "variants": reports}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
