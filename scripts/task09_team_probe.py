"""Declared public development scenarios for the fourteen-agent Task09 run."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

sys.path.insert(0, '/pilot')
import base_probe as B


def sample(variant, directory):
    scenarios, _, _ = B.runtime(variant)
    request = json.load(sys.stdin)
    source = directory / 'lead.xml' if request['model_source'] == 'candidate' else Path('/workspace') / variant / 'lead.xml'
    model, _ = scenarios.compose_lead(str(source))
    spec = importlib.util.spec_from_file_location('submitted_trim', directory / 'trim.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kind = request['kind']
    if kind == 'plan':
        value = np.asarray(module.plan_probe(model, tuple(np.array(q) for q in request['sample']),
            np.array(request['lower']), np.array(request['upper'])), dtype=float)
        if value.shape != (15, model.nv) or not np.isfinite(value).all():
            raise ValueError('plan_probe must return 15 finite poses')
        result = {'poses': value.tolist()}
    else:
        function = module.make_trim(model)
        if kind == 'adapted':
            params = module.adapt(model, [(np.array(a), np.array(b)) for a, b in request['probe']])
            function = module.make_trim_adapted(model, params)
        value = np.asarray([function(np.array(q)) for q in request['qmat']], dtype=float)
        if value.shape != (len(request['qmat']), model.nv) or not np.isfinite(value).all():
            raise ValueError('Trim must return finite joint torques')
        result = {'torques': value.tolist()}
    print(json.dumps(result, allow_nan=False))


def query(variant, directory, request):
    def drop():
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    result = subprocess.run([sys.executable, '-I', __file__, '--sample', variant, str(directory)],
        input=json.dumps(request), capture_output=True, text=True, timeout=25, preexec_fn=drop,
        env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp', 'OPENBLAS_NUM_THREADS': '1',
             'OMP_NUM_THREADS': '1', 'MUJOCO_GL': 'disable'})
    if result.returncode or len(result.stdout) > 1024**2:
        raise ValueError('Trim sampling failed: ' + result.stderr[-1200:])
    return json.loads(result.stdout)


def metrics(model, poses, torques):
    errors, headroom = [], []
    for target, torque in zip(poses, torques):
        measured, _, reserve = B.hold(model, target, np.clip(torque, -.35, .35))
        errors.append(float(np.sqrt(np.mean((measured-target)**2))))
        headroom.append(reserve)
    return {'hold_rms_error_rad': float(np.mean(errors)), 'hold_max_error_rad': max(errors),
            'servo_headroom_mean': float(np.mean(headroom))}


def variant_probe(variant):
    scenarios, spec, residual = B.runtime(variant)
    directory = Path('/submission') / variant
    model, data = scenarios.compose_lead(str(directory / 'lead.xml'))
    rng = np.random.default_rng(271828)
    home, hw = np.array(spec.LEAD_HOME), np.array(spec.WORKSPACE_HALFWIDTH)
    lower = np.maximum(home-hw, model.jnt_range[:, 0])
    upper = np.minimum(home+hw, model.jnt_range[:, 1])
    poses = np.vstack([home, np.clip(home + rng.uniform(-.85, .85, (5, model.nv))*hw, lower, upper)])
    residuals = np.array([residual(model, data, q) for q in scenarios.workspace_grid(levels=2)])
    base = {'model_source': 'candidate', 'qmat': poses.tolist()}
    nominal = query(variant, directory, {**base, 'kind': 'nominal'})['torques']
    result = {'mass_kg': float(np.sum(model.body_mass)), 'mass_budget_kg': 2.30,
        'passive_residual_rms_nm': float(np.sqrt(np.mean(residuals**2))),
        'passive_residual_max_nm': float(np.max(np.abs(residuals))),
        'nominal': metrics(model, poses, nominal)}
    drift = []
    for target in poses[:3]:
        state = mujoco.MjData(model)
        state.qpos[:] = target
        for _ in range(round(.6/model.opt.timestep)):
            mujoco.mj_step(model, state)
        drift.append(float(np.sqrt(np.mean((state.qpos-target)**2))))
    result['passive_drift_rad'] = float(np.mean(drift))
    spring = np.array([model.jnt_stiffness * (q-model.qpos_spring) for q in poses])
    result['spring_motion_rms_nm'] = float(np.sqrt(np.mean(spring**2)))
    for model_source in ('candidate', 'public_reference'):
        xml = directory / 'lead.xml' if model_source == 'candidate' else Path('/workspace') / variant / 'lead.xml'
        physical, _ = scenarios.compose_lead(str(xml))
        factors = rng.uniform(.88, 1.12, physical.nbody)
        physical.body_mass[1:] *= factors[1:]
        physical.body_inertia[1:] *= factors[1:, None]
        handle = mujoco.mj_name2id(physical, mujoco.mjtObj.mjOBJ_BODY, 'lead_handle')
        physical.body_mass[handle] += .06
        mujoco.mj_setConst(physical, mujoco.MjData(physical))
        measured, _, _ = B.hold(physical, home, np.zeros(physical.nv))
        initial = (home.tolist(), (measured+rng.normal(0, .005, physical.nv)).tolist())
        plan = query(variant, directory, {'kind': 'plan', 'model_source': model_source,
            'sample': initial, 'lower': lower.tolist(), 'upper': upper.tolist()})['poses']
        plan = np.array(plan)
        if np.any(plan < lower-1e-9) or np.any(plan > upper+1e-9):
            raise ValueError('Calibration poses outside public workspace')
        probe = [initial]
        for pose in plan:
            measured, _, _ = B.hold(physical, pose, np.zeros(physical.nv))
            probe.append((pose.tolist(), (measured+rng.normal(0, .005, physical.nv)).tolist()))
        request = {'model_source': model_source, 'qmat': poses.tolist()}
        adapted = query(variant, directory, {**request, 'kind': 'adapted', 'probe': probe})['torques']
        result['adapted' if model_source == 'candidate' else 'transfer_adapted'] = metrics(physical, poses, adapted)
        if model_source == 'public_reference':
            torque = query(variant, directory, {**request, 'kind': 'nominal'})['torques']
            result['transfer_nominal'] = metrics(physical, poses, torque)
    result['probe_measurements'] = 16
    print(json.dumps(result, allow_nan=False))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        sample(sys.argv[2], Path(sys.argv[3]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == '--variant':
        variant_probe(sys.argv[2])
        return
    reports = {}
    for variant in B.VARIANTS:
        try:
            completed = subprocess.run([sys.executable, __file__, '--variant', variant],
                capture_output=True, text=True, timeout=100)
            if completed.returncode:
                raise ValueError(completed.stderr[-1800:])
            reports[variant] = json.loads(completed.stdout)
        except Exception as error:
            reports[variant] = {'error': f'{type(error).__name__}: {error}'}
    Path('/logs/probe.json').write_text(json.dumps({'kind': 'declared_public_robustness_v1',
        'development_seed': 271828, 'mass_factor_range': [.88, 1.12], 'added_handle_mass_kg': .06,
        'encoder_noise_std_rad': .005, 'variants': reports}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
