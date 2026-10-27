"""Prospective finite same-state future-joint probe. Preparation never runs ONNX.

The execution primitive is dependency injected for finite fixtures; CLI deliberately
offers preparation only until a separate reviewed execution supplement exists.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/runs/agentic/app_walk_admission_v1/trials/identity_global'
EDIT = ROOT / 'artifacts/runs/agentic/app_walk_gain_probe_v1/trials/left_gain_a'
MODEL = ROOT / 'research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx'
PUBLIC_PYTHON = Path('C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe')
DECISIONS = (50, 60, 70)
INPUTS = ('current_anchor_rot', 'current_dof_pos', 'current_dof_vel',
          'current_root_local_ang_vel', 'historical_processed_actions',
          'mimic_future_anchor_rot', 'mimic_future_dof_pos', 'mimic_future_dof_vel')
OUTPUTS = ('actions', 'joint_pos_targets', 'stiffness_targets', 'damping_targets')
SWAPPED = frozenset(('mimic_future_dof_pos', 'mimic_future_dof_vel'))
SHAPES = ((1, 4), (1, 29), (1, 29), (1, 3), (1, 1, 29), (1, 4, 4), (1, 4, 29), (1, 4, 29))
ATOL = 1e-6


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_pair(baseline, edited):
    if set(baseline) != set(INPUTS) or set(edited) != set(INPUTS):
        raise ValueError('exact eight input names required')
    for name, shape in zip(INPUTS, SHAPES):
        for feed in (baseline, edited):
            a = feed[name]
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('invalid float32 finite feed: ' + name)
        if name not in SWAPPED and baseline[name].tobytes() != edited[name].tobytes():
            raise ValueError('non-joint counterfactual change: ' + name)
    if all(np.array_equal(baseline[n], edited[n]) for n in SWAPPED):
        raise ValueError('future joint edit is absent')


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not sys.flags.isolated or Path(sys.executable).resolve() != PUBLIC_PYTHON.resolve():
        raise ValueError('preparation requires pinned public interpreter with -I')
    sidecar = yaml.safe_load(MODEL.with_suffix('.yaml').read_text())
    if tuple(sidecar['_runtime']['onnx_in_names']) != INPUTS or tuple(sidecar['_runtime']['onnx_out_names']) != OUTPUTS:
        raise ValueError('runtime interface changed')
    if digest(MODEL) != 'a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c':
        raise ValueError('model changed')
    if sidecar['control']['pd_target_max_accel'] is not None or sidecar['control']['action_ema_alpha'] != 1:
        raise ValueError('PD postprocessor differs from documented direct target')
    pins = {}
    for directory in (BASE, EDIT):
        for name in ('manifest.json', 'proposal.json', 'policy_calls.json', 'policy_inputs.npz',
                     'policy_outputs.npz', 'reference_cache.npz', 'applied_joint_targets.npy',
                     'preexecution_inputs.json', 'source.motion', 'state_times.npy'):
            path = directory / name
            pins[str(path.relative_to(ROOT))] = digest(path)
        if json.loads((directory / 'manifest.json').read_text())['status'] != 'success':
            raise ValueError('archived execution was not successful')
    for path in (Path(__file__), MODEL, MODEL.with_suffix('.yaml'),
                 ROOT / 'experiments/agentic/app/execute_walk_probe_v1.py',
                 ROOT / 'research/vendor/ProtoMotions/deployment/test_tracker_mujoco.py',
                 ROOT / 'research/vendor/ProtoMotions/deployment/export_bm_tracker_onnx.py',
                 ROOT / 'artifacts/runs/agentic/app_reference_carrier_review_v1/REVIEW.md',
                 ROOT / 'artifacts/runs/agentic/app_reference_carrier_review_v1/evidence.json'):
        pins[str(path.relative_to(ROOT))] = digest(path)
    calls = [json.loads((p / 'policy_calls.json').read_text()) for p in (BASE, EDIT)]
    records = []
    with np.load(BASE / 'policy_inputs.npz') as bi, np.load(EDIT / 'policy_inputs.npz') as ei, np.load(BASE / 'policy_outputs.npz') as bo, np.load(BASE / 'reference_cache.npz') as bc, np.load(EDIT / 'reference_cache.npz') as ec:
        applied = np.load(BASE / 'applied_joint_targets.npy')
        for decision in DECISIONS:
            indices = []
            for rows in calls:
                found = [r for r in rows if r['role'] == 'control' and r['control_index'] == decision]
                if len(found) != 1 or tuple(found[0]['output_names']) != OUTPUTS:
                    raise ValueError('ambiguous actual online call mapping')
                indices.append(found[0]['call_index'])
            if indices[0] != indices[1]:
                raise ValueError('different archived call indexing')
            prefix = f'call_{indices[0]:04d}__'
            baseline = {n: bi[prefix + n].copy() for n in INPUTS}
            edited = {n: (ei if n in SWAPPED else bi)[prefix + n].copy() for n in INPUTS}
            validate_pair(baseline, edited)
            for name, cache in (('mimic_future_dof_pos', 'forecast__dof_pos'), ('mimic_future_dof_vel', 'forecast__dof_vel')):
                if not np.array_equal(baseline[name][0], bc[cache][decision]) or not np.array_equal(edited[name][0], ec[cache][decision]):
                    raise ValueError('feed/reference-cache mismatch')
            expected = {n: bo[prefix + n].copy() for n in OUTPUTS}
            if not np.array_equal(expected['joint_pos_targets'][0], applied[decision]):
                raise ValueError('graph target not actual archived PD command')
            paths = {}
            for kind, arrays in (('baseline', baseline), ('edited', edited), ('expected', expected)):
                path = output / f'decision_{decision}_{kind}.npz'
                np.savez(path, **arrays)
                paths[kind] = {'file': path.name, 'sha256': digest(path)}
            records.append({'decision': decision, 'time_s': decision * .02, 'archived_call_index': indices[0], 'feeds': paths})
    protocol = dict(schema='pulsemorph.future_joint_counterfactual_plan.v1', status='prepared_not_executed',
        maximum_session_run_calls=6, warmups=0, physics_steps=0, llm_calls=0,
        order=[{'decision': d, 'condition': c} for d in DECISIONS for c in ('baseline', 'edited')],
        replay_tolerance={'rtol': 0, 'atol': ATOL, 'all_four_outputs': True, 'no_relaxation': True},
        stop_rule='Any pin/interface/feed/nonfinite/replay/runtime failure stops immediately; no retries/replacements/resume.',
        runtime={'python': str(PUBLIC_PYTHON), 'isolated': True, 'providers': ['CPUExecutionProvider'], 'intra_op_num_threads': 2, 'inter_op_num_threads': 1},
        inputs=[{'name': n, 'shape': list(s), 'dtype': 'float32'} for n,s in zip(INPUTS,SHAPES)],
        outputs=[{'name': n, 'shape': [1,29], 'dtype': 'float32'} for n in OUTPUTS],
        joint_names=sidecar['joint_names'], future_offsets_controls=[1,2,4,8], pairs=records, source_pins=pins,
        pd_semantics='actions raw and graph joint_pos_targets measured directly. Exporter documents offset + scale * actions, but pinned sidecar has no coefficients; none inferred. Actual archived graph targets equal applied PD commands at all three selected decisions; clamp=None, EMA=1.',
        interpretation='Same baseline state/history/torso, edited future q/dq only. Finite open-loop mapping sensitivity, not closed-loop motion, physical features, training causality or global conclusion.',
        inference_authority='NONE: preparation only; reviewed execution supplement and root approval required.')
    write_new(output / 'protocol.json', protocol)
    return protocol


def finite_calls(session, pairs, attempt_dir):
    """Bounded injectable primitive; no real session is created by this module.

    A reviewed runner must verify all frozen pins/environment before supplying a
    real session. Exclusive attempt directory prevents any resume or replacement.
    """
    pairs = list(pairs)
    if len(pairs) != 3 or [p[0] for p in pairs] != list(DECISIONS):
        raise ValueError('exact three preregistered pairs required')
    for _, baseline, edited, expected in pairs:
        validate_pair(baseline, edited)
        if set(expected) != set(OUTPUTS) or any(a.shape != (1,29) or a.dtype != np.float32 or not np.isfinite(a).all() for a in expected.values()):
            raise ValueError('invalid archived output')
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for decision, baseline, edited, expected in pairs:
        for condition, feed in (('baseline', baseline), ('edited', edited)):
            number = len(results)
            if number >= 6:
                raise RuntimeError('call budget exceeded')
            write_new(attempt_dir / f'attempt_{number:02d}.json', {'decision': decision, 'condition': condition, 'attempt': number, 'status': 'attempted_before_session_run'})
            values = session.run(list(OUTPUTS), feed)
            if len(values) != 4 or any(a.shape != (1,29) or a.dtype != np.float32 or not np.isfinite(a).all() for a in values):
                raise ValueError('invalid model output; stop')
            if condition == 'baseline' and any(not np.allclose(a, expected[n], rtol=0, atol=ATOL) for n,a in zip(OUTPUTS,values)):
                raise ValueError('archived baseline replay mismatch; stop')
            np.savez(attempt_dir / f'output_{number:02d}.npz', **dict(zip(OUTPUTS, values)))
            results.append((decision, condition, values))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.prepare)
