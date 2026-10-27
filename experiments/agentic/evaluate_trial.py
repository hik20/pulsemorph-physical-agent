"""Read-only independent screening of the local four-second G1 trial protocol."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_accents = _module('_agentic_accents', ROOT/'experiments/music_control/accents.py')
_old = sys.modules.get('accents')
sys.modules['accents'] = _accents
try:
    _music = _module('_agentic_music_evaluation', ROOT/'experiments/music_control/evaluate.py')
finally:
    if _old is None:
        del sys.modules['accents']
    else:
        sys.modules['accents'] = _old
_state = _module('_agentic_state_utils', ROOT/'research/vendor/ProtoMotions/deployment/state_utils.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(directory, name):
    return json.loads((directory/name).read_text(encoding='utf8'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _audit(directory):
    directory = Path(directory).resolve()
    manifest = _json(directory, 'manifest.json')
    hashes = manifest['artifact_hashes']
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    _require(actual == set(hashes), 'local artifact inventory mismatch')
    for name, digest in hashes.items():
        path = (directory/name).resolve()
        _require(directory in path.parents and path.is_file() and _sha(path) == digest, 'artifact hash mismatch: '+name)
    pinned = _json(directory, 'preexecution_inputs.json')
    _require(bool(pinned) and pinned == manifest['preexecution_input_hashes'], 'pinned input manifest mismatch')
    live = {}
    for name, digest in pinned.items():
        snapshot = directory/'source_snapshot'/name
        if snapshot.exists():
            _require(_sha(snapshot) == digest, 'source snapshot mismatch: '+name)
        else:
            _require((ROOT/name).is_file() and _sha(ROOT/name) == digest, 'unsnapshotted model input mismatch: '+name)
        live[name] = (ROOT/name).is_file() and _sha(ROOT/name) == digest
    _require(manifest['status'] == 'success' and manifest['actual_control_steps'] == 200
             and manifest['expected_control_steps'] == 200 and manifest['actual_physics_steps'] == 4000
             and manifest['expected_physics_steps'] == 4000
             and manifest['online_onnx_call_count'] == 200 and manifest['warmup_onnx_call_count'] == 1
             and manifest['preexecution_hashes_unchanged'] is True
             and manifest['instrumented_set_initial_pose_call_count'] == 1
             and manifest['mujoco_warning_counts_before'] == manifest['mujoco_warning_counts_after']
             and manifest['all_substeps_finite'] and manifest['all_substeps_time_contiguous'], 'execution counts/status/warnings mismatch')
    _require(manifest['control_dt'] == .02 and manifest['physics_dt'] == .001 and manifest['decimation'] == 20, 'checkpoint timing mismatch')
    arrays = {name: np.load(directory/(name+'.npy'), allow_pickle=False)
              for name in ['qpos', 'qvel', 'state_times', 'applied_joint_targets']}
    for name, shape in [('qpos', (201, 36)), ('qvel', (201, 35)), ('state_times', (201,)), ('applied_joint_targets', (200, 29))]:
        _require(arrays[name].shape == shape and np.isfinite(arrays[name]).all(), 'invalid state/target array: '+name)
    _require(np.allclose(arrays['state_times'], np.arange(201)*.02, rtol=0, atol=1e-10), 'state clock mismatch')
    with np.load(directory/'substep_telemetry.npz', allow_pickle=False) as archive:
        sub = archive['values']
        _require(sub.shape == (4000, 10) and np.isfinite(sub).all(), 'invalid substep telemetry')
        _require(np.array_equal(sub[:, 0], np.arange(1, 4001))
                 and np.allclose(sub[:, 1], np.arange(1, 4001)*.001, rtol=0, atol=1e-10)
                 and np.all(sub[:, 8:] == 1), 'substep clock/finite flags mismatch')
    draft = _json(directory, 'motion_reference_draft.json')
    names, bodies = draft['joint_names'], draft['body_names']
    _require(len(names) == 29 and len(set(names)) == 29 and bodies.count('torso_link') == 1, 'G1 names mismatch')
    anchor = bodies.index('torso_link')
    cache = dict(np.load(directory/'reference_cache.npz', allow_pickle=False))
    _require(all(np.isfinite(v).all() for v in cache.values()), 'nonfinite reference carrier')
    for key, shape in [('committed__dof_pos', (201,29)), ('forecast__dof_pos', (200,4,29)),
                       ('forecast__dof_vel', (200,4,29)), ('committed__body_rot', (201,len(bodies),4)),
                       ('forecast__body_rot', (200,4,len(bodies),4))]:
        _require(cache[key].shape == shape and np.isfinite(cache[key]).all(), 'reference shape/finite mismatch: '+key)
    _require(np.allclose(np.linalg.norm(cache['forecast__body_rot'],axis=-1),1,rtol=0,atol=1e-5), 'reference quaternion norm mismatch')
    calls = _json(directory, 'policy_calls.json')
    _require(len(calls) == 201, 'policy call count mismatch')
    feeds = dict(np.load(directory/'policy_inputs.npz', allow_pickle=False))
    outputs = dict(np.load(directory/'policy_outputs.npz', allow_pickle=False))
    _require(all(np.isfinite(v).all() for v in [*feeds.values(), *outputs.values()]), 'nonfinite policy arrays')
    first = feeds['call_0001__current_anchor_rot'][0]
    offset = _state.compute_yaw_offset_np(first, cache['committed__body_rot'][0,anchor])
    aligned = _state.apply_heading_offset_np(offset, cache['forecast__body_rot'][:,:,anchor])
    for i, call in enumerate(calls):
        _require(call['call_index'] == i and call['role'] == ('warmup' if i == 0 else 'control')
                 and call['control_index'] == (None if i == 0 else i-1)
                 and call.get('all_finite') is True, 'policy call ordering/status mismatch')
        if not i:
            continue
        prefix = f'call_{i:04d}__'
        for semantic, expected in [('mimic_future_dof_pos', cache['forecast__dof_pos'][i-1][None]),
                                   ('mimic_future_dof_vel', cache['forecast__dof_vel'][i-1][None]),
                                   ('current_dof_pos', arrays['qpos'][i-1,7:][None].astype('float32')),
                                   ('current_dof_vel', arrays['qvel'][i-1,6:][None].astype('float32'))]:
            _require(np.array_equal(feeds[prefix+semantic], expected), 'policy input mismatch: '+semantic)
        _require(np.allclose(feeds[prefix+'mimic_future_anchor_rot'], aligned[i-1][None], rtol=0, atol=2e-7), 'torso heading alignment mismatch')
        _require(np.array_equal(outputs[prefix+'joint_pos_targets'][0], arrays['applied_joint_targets'][i-1]), 'PD output/applied target mismatch')
    rmse = np.sqrt(np.mean((arrays['qpos'][1:,7:]-cache['committed__dof_pos'][1:])**2, axis=1))
    return dict(manifest=manifest, arrays=arrays, cache=cache, names=names,
                sub=sub, calls=calls, live_input_hashes_match=live,
                mean_joint_rmse_rad=float(np.mean(rmse)))


def evaluate_trial(directory, baseline_directory, protocol):
    """Return independent metrics; malformed/partial evidence fails closed."""
    result = dict(schema='pulsemorph.agentic_trial_evaluation.v1', integrity_passed=False,
                  integrity_errors=[], physical_gate_passed=False, intent_ratio=None,
                  intent_abs_error=None, intent_passed=False, mean_joint_rmse_rad=None,
                  min_root_height_m=None, music={'available': False}, latency={},
                  limitation='local trial screening; not hardware safety, aesthetics or general success')
    try:
        trial, baseline = _audit(directory), _audit(baseline_directory)
        q, b = trial['arrays']['qpos'], baseline['arrays']['qpos']
        _require(trial['names'] == baseline['names'] and np.array_equal(q[0], b[0])
                 and np.array_equal(trial['arrays']['qvel'][0], baseline['arrays']['qvel'][0]), 'baseline initial state/profile mismatch')
        for key in ('onnx_sha256', 'motion_sha256', 'physics_contract'):
            _require(trial['manifest'][key] == baseline['manifest'][key], 'baseline controller/source mismatch: '+key)
        _require(trial['manifest']['versions'] == baseline['manifest']['versions'], 'baseline runtime versions mismatch')
        trial_pins=trial['manifest']['preexecution_input_hashes']
        baseline_pins=baseline['manifest']['preexecution_input_hashes']
        _require(trial_pins == baseline_pins, 'baseline pinned controller/source profile mismatch')
        arms = [i for i,n in enumerate(trial['names']) if any(part in n for part in ('shoulder_', 'elbow_', 'wrist_'))]
        _require(len(arms) == 14 and arms == list(range(15,29)), 'unsupported G1 arm order')
        start, end = protocol['task']['measurement_window_s']
        times = trial['arrays']['state_times']
        window = (times >= start-1e-10) & (times <= end+1e-10)
        _require(window.any() and 0 <= start < end <= 4, 'invalid intent measurement window')
        center = b[0,7:][arms]
        denominator = float(np.sqrt(np.mean((b[window,7:][:,arms]-center)**2)))
        _require(denominator > 1e-10, 'baseline arm amplitude is zero')
        numerator = float(np.sqrt(np.mean((q[window,7:][:,arms]-center)**2)))
        ratio = numerator/denominator
        error = abs(ratio-float(protocol['task']['target_ratio']))
        tolerance = float(protocol['task'].get('tolerance', .05))
        _require(np.isfinite(tolerance) and tolerance >= 0, 'invalid intent tolerance')
        height, rmse = float(np.min(trial['sub'][:,2])), trial['mean_joint_rmse_rad']
        result.update(integrity_passed=True, intent_ratio=ratio, intent_abs_error=error,
                      intent_passed=error <= tolerance, intent_tolerance=tolerance,
                      mean_joint_rmse_rad=rmse, min_root_height_m=height,
                      physical_gate_passed=height >= protocol['gates']['minimum_root_height_m'] and rmse <= protocol['gates']['maximum_joint_rmse_rad'],
                      arm_rms_rad=numerator, baseline_arm_rms_rad=denominator,
                      lower_body_deviation_rmse_rad=float(np.sqrt(np.mean((q[:,7:22]-b[:,7:22])**2))),
                      outside_edit_joint_deviation_rmse_rad=float(np.sqrt(np.mean((q[(times<1)|(times>3),7:]-b[(times<1)|(times>3),7:])**2))),
                      max_contacts=int(np.max(trial['sub'][:,3])), max_abs_actuator_force=float(np.max(trial['sub'][:,4])),
                      live_input_hashes_match=trial['live_input_hashes_match'])
        changed = np.flatnonzero(np.max(np.abs(q-b), axis=1)>1e-10)
        result['first_actual_state_deviation_s'] = float(times[changed[0]]) if len(changed) else None
        result['outside_edit_limitation'] = 'future reference conditioning may affect physics from 0.84s; no unchanged actual prefix until 1s claim'
        result['latency'] = dict(wall_seconds=trial['manifest']['wall_seconds'],
            online_onnx_ms_p50_p95_p99=np.percentile([row['elapsed_ms'] for row in trial['calls'][1:]], [50,95,99]).tolist())
        beats = protocol.get('evaluation_beats_s')
        events = _accents.extract_accents(trial['arrays']['qvel'][:,6:], .02)['times_seconds']
        result['music'] = dict(available=bool(beats), accent_times_s=events,
                              limitation='kinematic velocity accent proxy; evaluation beats are offline labels')
        if beats:
            result['music'].update(nearest_beat=_music.nearest_beat_errors(events,beats),
                                   one_to_one=_music.match_events(beats,events))
    except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as error:
        result.update(integrity_passed=False, physical_gate_passed=False, intent_passed=False)
        result['integrity_errors'].append(str(error))
    return result
