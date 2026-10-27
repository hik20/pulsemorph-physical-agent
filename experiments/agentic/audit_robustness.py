"""Independent, read-only audit of the frozen initial true-state sensitivity panel.

No mj_step, policy execution, model calls, retries, or outcome-dependent selection.
Filesystem chronology is local provenance evidence, not signed attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from audit_multifeature import common_parity
from evaluate_generalization import _audit, _fresh_fk_orientations
from robustness_scenarios import build_scenarios
from run_tool_compatibility_smoke import trace_parity, array_archive_parity

RUNTIME = 'experiments/agentic/execute_robustness.py'
OLD_RUNTIME = 'experiments/agentic/execute_tool_selection.py'
COMPATIBLE_RUNTIME = 'experiments/agentic/execute_tool_selection_compatible.py'
ROLES = ('calibration_wave', 'calibration_high5', 'selected_wave_a', 'selected_wave_b',
         'selected_high5_a', 'selected_high5_b')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def inventory(directory):
    """Independent complete inventory, hashing shared immutable inode bytes once."""
    directory = Path(directory)
    result, cached = {}, {}
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in cached:
                cached[key] = sha(path)
            result[path.relative_to(directory).as_posix()] = cached[key]
    return result


def array_sha(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def byte_equal(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.dtype == expected.dtype and actual.shape == expected.shape
            and actual.tobytes() == expected.tobytes(), label + ' dtype/shape/bytes mismatch')


def pinned(link):
    require(isinstance(link, dict) and set(link) == {'path', 'sha256'}, 'exact path/SHA link required')
    path = Path(link['path'])
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_file() and sha(path) == link['sha256'], 'linked evidence changed: ' + str(path))
    return path.resolve()


def reconstruct_initial_state(nominal_qpos, nominal_qvel, spec):
    """Independent bounded spec reconstruction; no import of runtime injection."""
    q = np.asarray(nominal_qpos).copy()
    v = np.asarray(nominal_qvel).copy()
    require(q.shape == (36,) and v.shape == (35,), 'nominal G1 state shape')
    fields = {'joint_position_offset_rad': (29, .02),
              'joint_velocity_offset_rad_s': (29, .1),
              'root_linear_velocity_offset_world_m_s': (3, .05),
              'root_yaw_offset_rad': (None, float(np.deg2rad(2)))}
    require(isinstance(spec, dict) and set(spec) == set(fields), 'exact perturbation fields')
    arrays = {}
    for key, (size, limit) in fields.items():
        values = [spec[key]] if size is None else spec[key]
        require(isinstance(values, list) and len(values) == (1 if size is None else size), 'perturbation vector length')
        require(all(type(x) in (int, float) and np.isfinite(x) and abs(x) <= limit for x in values),
                'perturbation numeric bounds')
        arrays[key] = np.asarray(values, dtype=float)
    nonzero = any(np.any(value) for value in arrays.values())
    if nonzero:
        q[7:] += arrays['joint_position_offset_rad']
        v[6:] += arrays['joint_velocity_offset_rad_s']
        v[:3] += arrays['root_linear_velocity_offset_world_m_s']
        yaw = arrays['root_yaw_offset_rad'][0]
        if yaw != 0.:
            # world-z rotation is the LEFT operand, MuJoCo quaternion order wxyz.
            w, x, y, z = q[3:7]
            c, s = np.cos(yaw/2), np.sin(yaw/2)
            q[3:7] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
    return q, v, bool(nonzero)


def check_initial_state(trial, original, scenario, audit=None):
    trial, original = Path(trial), Path(original)
    manifest = read(trial/'manifest.json')
    before_q = np.load(original/'qpos.npy', allow_pickle=False)[0]
    before_v = np.load(original/'qvel.npy', allow_pickle=False)[0]
    expected_q, expected_v, nonzero = reconstruct_initial_state(before_q, before_v, scenario['initial_perturbation'])
    log = manifest['initial_perturbation']
    require(log['spec'] == scenario['initial_perturbation'], 'applied perturbation spec differs from frozen scenario')
    for key, expected in [('before_qpos', before_q), ('before_qvel', before_v),
                          ('after_qpos', expected_q), ('after_qvel', expected_v)]:
        byte_equal(np.asarray(log[key], dtype=expected.dtype), expected, 'initial log ' + key)
        require(log[key + '_sha256'] == array_sha(expected), 'initial state hash mismatch: ' + key)
    require(log['applied_once_at_time_s'] == 0. and log['nonzero'] is nonzero
            and log['extra_forward_count'] == int(nonzero), 'initial application time/forward count')
    require(log['root_linear_velocity_frame'] == 'world'
            and log['root_yaw_frame'] == 'world_left_multiply'
            and log['quaternion_order'] == 'wxyz' and log['reference_rotated'] is False,
            'initial coordinate convention')
    require(manifest['instrumented_set_initial_pose_call_count'] == 1, 'exactly one reset required')
    qpos = np.load(trial/'qpos.npy', allow_pickle=False)
    qvel = np.load(trial/'qvel.npy', allow_pickle=False)
    times = np.load(trial/'state_times.npy', allow_pickle=False)
    require(times[0] == 0., 'initial state time is not zero')
    byte_equal(qpos[0], expected_q, 'initial qpos array')
    byte_equal(qvel[0], expected_v, 'initial qvel array')
    byte_equal(expected_v[3:6], before_v[3:6], 'angular velocity must remain unchanged')
    for name, expected in [('actual_initial_qpos_sha256', expected_q), ('actual_initial_qvel_sha256', expected_v)]:
        require(manifest[name] == array_sha(expected), 'actual initial state hash mismatch')
    with np.load(trial/'policy_inputs.npz', allow_pickle=False) as feeds:
        first_anchor = feeds['call_0001__current_anchor_rot'].copy() if audit is not None else None
        byte_equal(feeds['call_0001__current_dof_pos'], expected_q[7:][None].astype('float32'), 'first current q')
        byte_equal(feeds['call_0001__current_dof_vel'], expected_v[6:][None].astype('float32'), 'first current dq')
        for index in range(1, 201):
            byte_equal(feeds[f'call_{index:04d}__current_root_local_ang_vel'],
                       qvel[index-1, 3:6][None].astype('float32'), 'current root local angular velocity')
    require(log['initial_root_height_m'] == float(expected_q[2]), 'initial root height diagnostic')
    require(log['warning_counts_after_forward'] == manifest['mujoco_warning_counts_before'], 'post-injection warnings')
    require(type(log['initial_contact_count']) is int and log['initial_contact_count'] >= 0,
            'initial contact count diagnostic')
    if audit is not None:
        require(audit.get('fresh_anchor_audit', {}).get('control_rows') == 200, 'all200 same-time FK checks required')
        pins = manifest['preexecution_input_hashes']
        names = [name for name in pins if name.replace('\\', '/').endswith('/g1_holo_compat.xml')]
        require(len(names) == 1, 'exact pinned FK XML for initial anchor')
        xml_name = names[0]
        fk = _fresh_fk_orientations(trial/'source_snapshot'/xml_name, expected_q[None], (ROOT/xml_name).parent)
        byte_equal(first_anchor, fk.astype('float32'), 'first current anchor FK')
    return dict(reconstructed_initial_state_bytes=True, first_policy_joint_inputs_exact=True,
                angular_velocity_unchanged=True, perturbation_application_count=1,
                time_s=0., nonzero=nonzero, yaw_is_heading_equivariance=True)


def normalized_pins(pins):
    result = {}
    for key, value in pins.items():
        normalized = key.replace('\\', '/')
        require(normalized not in result, 'normalized source pin collision')
        result[normalized] = value
    return result


def check_input_pins(original, actual):
    old = normalized_pins(original['preexecution_input_hashes'])
    new = normalized_pins(actual['preexecution_input_hashes'])
    removed = {OLD_RUNTIME}
    added = {RUNTIME, COMPATIBLE_RUNTIME}
    require(set(old) - removed == set(new) - added, 'unexpected source pin path difference')
    require(OLD_RUNTIME in old and RUNTIME in new and COMPATIBLE_RUNTIME in new, 'declared runtime pin switch missing')
    for key in set(old) - removed:
        require(old[key] == new[key], 'shared source pin changed: ' + key)
    for key in added:
        require(new[key] == sha(ROOT/key), 'new runtime source changed: ' + key)
    for key in ('onnx_sha256', 'motion_sha256', 'body_profile_sha256', 'controller_profile_sha256',
                'checkpoint_sha256', 'physics_contract', 'versions'):
        require(original[key] == actual[key], 'controller/body/source/environment changed: ' + key)
    return dict(only_declared_runtime_source_changes=True, shared_inputs_identical=True)


def check_scenarios(scenarios, joint_names):
    expected = build_scenarios(joint_names)
    require(scenarios == expected, 'frozen scenario sign/seed/name/vector panel mismatch')
    require(len(scenarios) == 13 and len({s['scenario_id'] for s in scenarios}) == 13, 'exact13 scenarios required')
    return expected


def check_center_definitions(evaluation, candidate, nominal, condition, joint_names, protocol, goal):
    """Recompute four RMS dimensions without calling the production evaluator."""
    require(evaluation['integrity_passed'] is True, 'candidate evaluator integrity failure')
    require(evaluation['initial']['primary_center'] == 'nominal baseline actual initial joints'
            and evaluation['initial']['secondary_center'] == 'condition baseline actual initial joints',
            'primary/secondary center definitions changed')
    times = np.asarray(candidate['state_times'])
    q, b0, bp = [np.asarray(row['qpos']) for row in (candidate, nominal, condition)]
    require(np.array_equal(times, nominal['state_times']) and np.array_equal(times, condition['state_times']),
            'center-check clocks differ')
    target = np.asarray(goal['feature_targets'], dtype=float)
    tolerance = np.asarray(protocol['feature_tolerances'], dtype=float)
    require(target.shape == tolerance.shape == (5,), 'center-check target/tolerance shape')
    primary, secondary = [], []
    index = 0
    for label, (start, end) in zip(('a', 'b'), protocol['feature_windows_s']):
        mask = (times >= start-1e-10) & (times <= end+1e-10)
        for side in ('left', 'right'):
            group = [i for i, name in enumerate(joint_names) if name.startswith(side+'_')
                     and any(part in name for part in ('shoulder_', 'elbow_', 'wrist_'))]
            require(len(group) == 7, 'center-check arm group')
            c0, cp = b0[0, 7:][group], bp[0, 7:][group]
            def amplitude(values, center):
                return float(np.sqrt(np.mean((values[mask, 7:][:, group]-center)**2)))
            C0, B0, Bcondition0, Cp, Bp = (amplitude(q,c0), amplitude(b0,c0),
                amplitude(bp,c0), amplitude(q,cp), amplitude(bp,cp))
            require(B0 > 1e-10, 'nominal denominator unavailable')
            primary.append(C0/B0); secondary.append(Cp/Bp if Bp > 1e-10 else None)
            expected = dict(name=side+'_ratio_'+label, candidate_rms_about_nominal_rad=C0,
                nominal_baseline_rms_rad=B0, condition_baseline_rms_about_nominal_rad=Bcondition0,
                candidate_rms_about_condition_rad=Cp, condition_baseline_rms_rad=Bp,
                condition_center_minus_nominal_rad=(cp-c0).tolist(),
                joint_names=[joint_names[i] for i in group],
                primary_absolute_rms_target_rad=float(target[index]*B0),
                primary_absolute_rms_tolerance_rad=float(tolerance[index]*B0),
                secondary_available=Bp > 1e-10,
                secondary_unavailable_reason=None if Bp > 1e-10 else 'zero condition baseline amplitude')
            require(evaluation['dimensions'][index] == expected, 'independent RMS center/denominator dimension mismatch')
            index += 1
    require(len(evaluation['dimensions']) == 4, 'exact four RMS dimensions')
    for name, features in [('primary', primary), ('secondary', secondary)]:
        score = evaluation[name]
        require(score['features'][:4] == features, name+' center-derived ratio mismatch')
        require(score['feature_tolerances'] == tolerance.tolist(), name+' tolerance changed')
        all_features = features + [evaluation['accent']['peak_time_s']]
        errors = [None if value is None else float(value-wanted) for value,wanted in zip(all_features,target)]
        passes = [error is not None and abs(error) <= tol for error,tol in zip(errors,tolerance)]
        require(score['features'] == all_features and score['feature_errors'] == errors
                and score['per_dimension_pass'] == passes and score['intent_passed'] == all(passes),
                name+' independent score mismatch')
        require(score['joint_success'] == bool(evaluation['physical_gate_passed'] and all(passes)),
                name+' combined physical/intent success mismatch')
        norms = None if any(error is None for error in errors) else np.abs(np.asarray(errors)/tolerance)
        require(score['utility'] == (None if norms is None else float(np.max(norms)))
                and score['utility_l2'] == (None if norms is None else float(np.sum(norms**2))),
                name+' independent utility mismatch')
    return dict(primary_fixed_nominal_center_and_denominator=True,
                secondary_condition_center_and_denominator=True,
                absolute_primary_targets_and_tolerances_verified=True, dimensions=4)


def check_runtime_hook_source(path):
    """Audit source structure in addition to once-only execution evidence."""
    import ast
    tree = ast.parse(Path(path).read_text(encoding='utf8'))
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_trial')
    hook = next(n for n in ast.walk(execute) if isinstance(n, ast.FunctionDef) and n.name == 'initial')
    step = next(n for n in ast.walk(execute) if isinstance(n, ast.FunctionDef) and n.name == 'step')
    hook_calls = [n for n in ast.walk(hook) if isinstance(n, ast.Call)]
    named = lambda name: [n for n in hook_calls if ast.unparse(n.func) == name]
    require(len(named('real_initial')) == len(named('apply_initial_perturbation')) == 1,
            'reset and injection must each occur exactly once in initial hook')
    require(named('real_initial')[0].lineno < named('apply_initial_perturbation')[0].lineno
            < named('states.append')[0].lineno, 'injection must follow reset and precede capture')
    for node in ast.walk(step):
        if isinstance(node, ast.Call):
            require(ast.unparse(node.func) not in ('apply_initial_perturbation', 'real_initial', 'initial'),
                    'per-step injection/reset forbidden')
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            require(not any(ast.unparse(target).startswith(('data.qpos', 'data.qvel')) for target in targets),
                    'per-step direct state writes forbidden')
    return dict(reset_then_injection_then_capture=True, no_per_step_injection_or_state_writes=True)


def calibration_measurements(audited):
    """Goal-free calibration representation; no scored target or success labels."""
    return dict(mean_joint_rmse_rad=audited['mean_joint_rmse_rad'],
                min_root_height_m=float(np.min(audited['sub'][:,2])),
                actual_initial_qpos_sha256=audited['manifest']['actual_initial_qpos_sha256'],
                actual_initial_qvel_sha256=audited['manifest']['actual_initial_qvel_sha256'],
                goal=None, target_features=None, scored_against_target=False)


def check_candidate_evaluation(panel, identity, scenario, case, protocol, goal, audited, all_audits):
    from evaluate_robustness import evaluate_robustness
    nominal_id = 'nominal__calibration_' + case['source_id']
    condition_id = scenario['scenario_id'] + '__calibration_' + case['source_id']
    paths = [Path(panel)/'trials'/name for name in (identity, nominal_id, condition_id)]
    recomputed = evaluate_robustness(*paths, protocol, goal)
    require(recomputed['integrity_passed'] is True, 'candidate independent evaluation failed: '+identity)
    observed = read(Path(panel)/'evaluations'/(identity+'.json'))
    require(observed == recomputed, 'stored evaluator result differs from recomputation: '+identity)
    centers = check_center_definitions(recomputed, audited['arrays'], all_audits[nominal_id]['arrays'],
        all_audits[condition_id]['arrays'], audited['names'], protocol, goal)
    return dict(evaluation=recomputed, independent_center_check=centers)


def check_chronology(panel, identities, nominal_ids):
    from datetime import datetime
    events = [json.loads(line) for line in (Path(panel)/'execution_journal.jsonl').read_text().splitlines() if line]
    require(all(row['event'] in ('panel_start','trial_start','trial_complete','nominal_gate','panel_complete') for row in events), 'unexpected execution journal event')
    require(len(events)==159 and events[0]['event']=='panel_start' and events[-1]['event']=='panel_complete', 'exact complete panel journal envelope')
    require(events[0]['protocol_sha256']==sha(Path(panel)/'protocol.json')
            and events[-1]['completed'] is True and events[-1]['physical_trials']==78, 'panel journal completion/protocol mismatch')
    dates = [datetime.fromisoformat(row['time_utc']) for row in events]
    require(all(value.tzinfo is not None for value in dates) and dates == sorted(dates), 'timezone-aware monotonic execution journal required')
    starts = [row['trial_id'] for row in events if row['event']=='trial_start']
    completes = [row['trial_id'] for row in events if row['event']=='trial_complete']
    require(len(starts)==len(completes)==78 and set(starts)==set(completes)==set(identities), 'exact78 journal starts/completions')
    gates = [i for i,row in enumerate(events) if row['event']=='nominal_gate']
    require(len(gates)==1, 'exactly one nominal parity gate event required')
    gate = gates[0]
    require(events[gate]['passed'] is True
            and events[gate]['nominal_parity_sha256']==sha(Path(panel)/'nominal_parity.json'), 'nominal gate artifact hash/pass mismatch')
    require({row['trial_id'] for row in events[:gate] if row['event']=='trial_complete'} == set(nominal_ids),
            'all six zero runs must complete before nominal gate')
    require({row['trial_id'] for row in events[:gate] if row['event']=='trial_start'} == set(nominal_ids),
            'nonzero trial started before nominal gate')
    for identity in identities:
        start = next(i for i,row in enumerate(events) if row['event']=='trial_start' and row['trial_id']==identity)
        complete = next(i for i,row in enumerate(events) if row['event']=='trial_complete' and row['trial_id']==identity)
        require(start < complete, 'trial completion before start')
        if identity not in nominal_ids:
            require(gate < start, 'perturbed runtime preceded zero parity gate')
        require(events[start]['proposal_sha256']==sha(Path(panel)/'proposals'/(identity+'.json')), 'journal executed proposal digest')
        row = events[complete]
        require(row['status']=='success' and row['control_steps']==200 and row['physics_steps']==4000, 'journal completed execution count/status')
        digest = row.get('manifest_sha256', row.get('manifestsha'))
        require(digest == sha(Path(panel)/'trials'/identity/'manifest.json'), 'journal completed manifest changed')
    require((Path(panel)/'nominal_parity.json').is_file(), 'nominal parity artifact missing')
    return dict(nominal_all_six_before_any_perturbation=True, starts=78, completions=78,
                chronology_scope='local UTC journal; not signed attestation')


def _resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT/value).resolve()


def check_protected(protocol):
    links = {}
    for key in ('original_protocol','original_comparison','accepted_projection_audit'):
        links[key] = pinned(protocol[key])
    original = read(links['original_protocol'])
    comparison = read(links['original_comparison'])
    projection = read(links['accepted_projection_audit'])
    require(comparison['comparison_valid'] is True and comparison['physical_trials']==100,
            'original completed100 comparison required')
    require(projection['audit_passed'] is True, 'passed projection audit required')
    require(comparison['protocol_sha256']==sha(links['original_protocol']), 'original comparison/protocol mismatch')
    for key in ('feature_windows_s','accent_window_s','accent_smoothing_samples','accent_minimum_prominence',
                'physical_gate','feature_tolerances','parameter_names','bounds','section_windows_s',
                'duration_s','source_start_s'):
        require(protocol[key]==original[key], 'original frozen configuration changed: '+key)
    require(protocol['tasks']==original['tasks'], 'original task/goals changed')
    require(protocol['goals']=={task['task_id']:task['goal'] for task in original['tasks']}, 'exact original goals mapping')
    expected_protected = {'artifacts/runs/agentic/tool_selection_development_v2',
        'artifacts/runs/agentic/tool_selection_development_v2_compatible_refs_v1',
        'artifacts/runs/agentic/tool_selection_development_v2_projection_audit_v1'}
    require({row['path'] for row in protocol['protected_inventories']} == expected_protected, 'exact protected evidence scope')
    for protected in protocol['protected_inventories']:
        require(inventory(_resolve(protected['path'])) == protected['files'], 'protected evidence inventory changed')
    require(len(protocol['protected_inventories'])==3, 'all three original/projection inventories required')
    return original, comparison


def check_fixed_cases(cases, comparison):
    require(len(cases)==6 and {case['role'] for case in cases}==set(ROLES), 'exact six frozen roles required')
    for case in cases:
        proposal_path = pinned(case['original_proposal'])
        proposal = read(proposal_path)
        original_trial = _resolve(case['original_trial'])
        require(sha(original_trial/'proposal.json') == sha(proposal_path), 'original executed proposal differs')
        require(proposal['source_id']==case['source_id'], 'fixed case source identity')
        require(proposal['task_id']==case['task_id'] and proposal['source_motion']['sha256'] == read(original_trial/'manifest.json')['motion_sha256'],
                'fixed case task/source mismatch')
        if case['role'].startswith('calibration_'):
            require(case['role']=='calibration_'+case['source_id'], 'calibration role/source identity')
            require(proposal['method']=='calibration' and case['goal'] is None, 'goal-free calibration origin required')
        else:
            task = next(row for row in comparison['tasks'] if row['task_id']==case['task_id'])
            selected = next(row for row in task['methods'] if row['method']=='llm_feedback')['selected']
            require(proposal['method']=='llm_feedback' and selected['candidate_id']==proposal['candidate_id'],
                    'fixed selected output differs from original selection')
            require(case['role']=='selected_'+case['task_id'], 'selected role/task identity')
    return True


def audit(panel, output):
    """Verify completed78 artifacts without changing panel or original evidence."""
    panel, output = Path(panel).resolve(), Path(output).resolve()
    require(output!=panel and panel not in output.parents, 'audit output must be outside panel')
    output.mkdir(parents=True, exist_ok=False)
    before = inventory(panel)
    protocol = read(panel/'protocol.json')
    sources = {name:sha(ROOT/name) for name in protocol['execution_code_sha256']}
    sources[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    for name in sources:
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    protected_before = {item['path']:inventory(_resolve(item['path'])) for item in protocol['protected_inventories']}
    report = dict(schema='pulsemorph.agentic_initial_state_sensitivity_audit.v1', audit_passed=False,
        errors=[], trials={}, selected_rows=[], calibrations=[], new_physical_trials=0, new_llm_calls=0,
        source_hashes_before=sources, input_protocol_sha256=sha(panel/'protocol.json'),
        scope='Read-only recomputation; deterministic fixed-output sensitivity, no recovery/probability/aesthetics claim',
        limitations=['Local journal chronology is provenance evidence, not signed attestation.',
                     'MuJoCo numerical outcomes do not establish hardware safety or human preference.'])
    try:
        original_protocol, comparison = check_protected(protocol)
        check_fixed_cases(protocol['cases'], comparison)
        from pulsemorph.motion.g1 import profile_from_yaml
        names = profile_from_yaml()['joint_names']
        scenarios = check_scenarios(protocol['scenarios'], names)
        require(protocol['expected_counts']==dict(physical_trials=78,calibration_trials=26,
            selected_trials=52,perturbed_selected_trials=48,new_llm_calls=0), 'fixed panel budget changed')
        for name,digest in protocol['execution_code_sha256'].items():
            require(sources[name]==digest and sha(panel/'source_snapshot'/name)==digest, 'frozen execution source changed: '+name)
        for index, protected in enumerate(protocol['protected_inventories']):
            require(protocol['protected_evidence'][index]==dict(path=protected['path'],inventory_file=f'protected_inventories/{index}.json')
                and read(panel/'protected_inventories'/(str(index)+'.json'))==protected['files'], 'protected inventory file/protocol mismatch')
        pinned(protocol['structural_preflight'])
        report['runtime_source_contract'] = check_runtime_hook_source(panel/'source_snapshot'/RUNTIME)
        identities = [scenario['scenario_id']+'__'+case['role'] for scenario in scenarios for case in protocol['cases']]
        require(len(identities)==78, 'exact78 planned trial identities')
        require({path.name for path in (panel/'trials').iterdir() if path.is_dir()}==set(identities), 'exact78 trial directory inventory')
        require({path.stem for path in (panel/'proposals').glob('*.json')}==set(identities), 'exact78 proposal inventory')
        for folder in ('model_calls','observations','responses'):
            require(not (panel/folder).exists() or not any((panel/folder).rglob('*')), 'unexpected new model/selection artifacts')
        nominal_ids = ['nominal__'+role for role in ROLES]
        report['execution_chronology'] = check_chronology(panel, identities, nominal_ids)
        all_audits = {}
        # Preserve every row, including failures; no outcome-dependent filtering.
        for scenario in scenarios:
            for case in protocol['cases']:
                identity = scenario['scenario_id']+'__'+case['role']
                trial = panel/'trials'/identity
                entry = dict(candidate_id=identity,scenario_id=scenario['scenario_id'],role=case['role'],audit_passed=False)
                report['trials'][identity] = entry
                try:
                    proposal = read(panel/'proposals'/(identity+'.json'))
                    require(sha(trial/'proposal.json')==sha(panel/'proposals'/(identity+'.json')), 'executed proposal mismatch')
                    original_path = pinned(case['original_proposal']); original = read(original_path)
                    digest = sha(panel/'protocol.json')
                    expected_proposal = dict(original, candidate_id=identity,
                        execution_kind='initial_state_sensitivity_replay',
                        observation_id=identity+':replay-observation', observation_artifact=None, response_artifact=None,
                        parameter_source=case['original_proposal'],
                        initial_perturbation=scenario['initial_perturbation'], protocol_sha256=digest,
                        runtime_session_id=digest+':'+scenario['scenario_id']+':'+case['role'])
                    require(proposal==expected_proposal, 'entire replay proposal differs from frozen original transformation')
                    require(proposal['candidate_id']==identity and proposal['protocol_sha256']==digest, 'proposal identity/protocol mismatch')
                    require(proposal['execution_kind']=='initial_state_sensitivity_replay'
                            and proposal['parameter_source']==case['original_proposal'], 'replay origin linkage mismatch')
                    require(proposal['initial_perturbation']==scenario['initial_perturbation'], 'proposal scenario differs from frozen panel')
                    for key in ('method','control_tool','parameters','source_motion','task_id'):
                        require(proposal[key]==original[key], 'fixed original proposal changed: '+key)
                    require(sha(trial/'source_proposal.json') == sha(original_path), 'runtime copied origin proposal differs')
                    audited = _audit(trial); all_audits[identity] = audited
                    manifest = audited['manifest']
                    require(manifest.get('strict_g1_admission_passed') is True and manifest.get('strict_g1_preflight_count')==2,
                            'strict G1 preflight admission evidence missing')
                    require(manifest.get('new_llm_calls')==0 and manifest.get('execution_kind')=='initial_state_sensitivity_replay', 'new runtime replay disclosure')
                    require(manifest.get('reference_initial_posture_is_physics_state') is False, 'reference posture mislabeled as physical state')
                    entry['initial_state'] = check_initial_state(trial, _resolve(case['original_trial']), scenario, audited)
                    entry['input_pins'] = check_input_pins(read(_resolve(case['original_trial'])/'manifest.json'), manifest)
                    array_archive_parity(_resolve(case['original_trial'])/'reference_cache.npz',trial/'reference_cache.npz')
                    array_archive_parity(_resolve(case['original_trial'])/'common_reference/arrays.npz',trial/'common_reference/arrays.npz')
                    entry['old_reference_arrays_unchanged'] = True
                    entry['strict_common_all200_parity'] = common_parity(trial,proposal,output/'roundtrip_packages'/identity)
                    entry['fresh_anchor_audit'] = audited['fresh_anchor_audit']
                    if scenario['scenario_id']=='nominal' and case['role'].startswith('calibration_'):
                        require(float(np.min(audited['sub'][:,2])) >= protocol['physical_gate']['min_root_height_m']
                            and audited['mean_joint_rmse_rad'] <= protocol['physical_gate']['max_mean_joint_rmse_rad'],
                            'nominal calibration physical gate failed')
                        entry['nominal_calibration_physical_gate_passed'] = True
                    if scenario['scenario_id']=='nominal':
                        entry['full_zero_trace_parity'] = trace_parity(_resolve(case['original_trial']),trial)
                    entry['audit_passed'] = True
                except Exception as error:
                    entry['error'] = str(error); report['errors'].append(identity+': '+str(error))
        for scenario in scenarios:
            for case in protocol['cases']:
                identity = scenario['scenario_id']+'__'+case['role']
                if identity not in all_audits:
                    continue
                try:
                    if case['role'].startswith('calibration_'):
                        measured = calibration_measurements(all_audits[identity])
                        require(read(panel/'calibrations'/(identity+'.json'))==measured, 'goal-free calibration artifact mismatch')
                        report['calibrations'].append(dict(candidate_id=identity,scenario_id=scenario['scenario_id'],role=case['role'],**measured))
                    else:
                        goal = protocol['goals'][case['task_id']]
                        require(case['goal']==goal, 'selected case original goal differs')
                        scored = check_candidate_evaluation(panel,identity,scenario,case,protocol,goal,all_audits[identity],all_audits)
                        report['selected_rows'].append(dict(candidate_id=identity,scenario_id=scenario['scenario_id'],role=case['role'],**scored))
                except Exception as error:
                    report['errors'].append(identity+': '+str(error))
        require(len(report['trials'])==78 and len(report['calibrations'])==26 and len(report['selected_rows'])==52,
                'full78 evidence rows must be preserved')
        require(sum(row['scenario_id']!='nominal' for row in report['selected_rows'])==48, 'all48 perturbed selected outcomes required')
        require(all(row['audit_passed'] for row in report['trials'].values()), 'one or more trial audits failed')
        require(not report['errors'], 'audit encountered evidence errors')
        report['nominal_parity_report'] = check_nominal_report(panel,protocol,report['trials'])
        report['summary_report'] = check_summary(panel,protocol,report['selected_rows'])
        report['audit_passed'] = True
    except Exception as error:
        report['errors'].append(str(error))
    finally:
        report['panel_evidence_unchanged'] = inventory(panel)==before
        report['protected_evidence_unchanged'] = all(inventory(_resolve(name))==value for name,value in protected_before.items())
        report['source_hashes_after'] = {name:sha(ROOT/name) for name in sources}
        report['audit_sources_unchanged'] = report['source_hashes_after']==sources
        if not all(report[key] for key in ('panel_evidence_unchanged','protected_evidence_unchanged','audit_sources_unchanged')):
            report['audit_passed'] = False; report['errors'].append('evidence or auditor source changed during read-only audit')
        (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    return report





def check_nominal_report(panel, protocol, trials):
    report = read(Path(panel)/'nominal_parity.json')
    require(set(report)=={'passed','cases','protocol_sha256'} and report['passed'] is True
            and report['protocol_sha256']==sha(Path(panel)/'protocol.json'), 'nominal parity report envelope')
    expected = []
    for case in protocol['cases']:
        identity = 'nominal__'+case['role']
        entry = trials[identity]
        expected.append(dict(role=case['role'],trial_id=identity,passed=True,
            trace_parity=entry['full_zero_trace_parity'], input_pin_parity=entry['input_pins']))
    require(report['cases']==expected, 'nominal parity six-case report differs from independent reconstruction')
    return dict(exact_six_nominal_cases=True, all_full_trace_bytes_identical=True)


def check_summary(panel, protocol, selected_rows):
    """Rebuild driver rows and per-task counts without importing its summarizer."""
    summary = read(Path(panel)/'summary.json')
    require(summary['schema']=='pulsemorph.agentic_robustness_summary.v1' and summary['panel_completed'] is True
            and summary['errors']==[] and summary['protocol_sha256']==sha(Path(panel)/'protocol.json'), 'completed summary envelope')
    require(summary['physical_trials']==78 and summary['control_steps']==15600 and summary['physics_steps']==312000
            and summary['new_llm_calls']==0 and summary['training_admission'] is False
            and summary['source_pins_unchanged'] is True, 'summary counts/admission/source pins')
    case_map = {case['role']:case for case in protocol['cases']}
    expected_rows = []
    for row in selected_rows:
        case = case_map[row['role']]; evaluation = row['evaluation']
        expected_rows.append(dict(trial_id=row['candidate_id'],task_id=case['task_id'],source_id=case['source_id'],
            scenario_id=row['scenario_id'],primary=evaluation['primary'],secondary=evaluation['secondary'],
            physical_gate_passed=evaluation['physical_gate_passed'],min_root_height_m=evaluation['min_root_height_m'],
            mean_joint_rmse_rad=evaluation['mean_joint_rmse_rad']))
    require(summary['rows']==expected_rows, 'summary excludes/changes fixed selected outcome rows')
    expected_tasks = []
    for task in ('wave_a','wave_b','high5_a','high5_b'):
        rows = [row for row in expected_rows if row['task_id']==task]
        nominal = [row for row in rows if row['scenario_id']=='nominal']
        nonzero = [row for row in rows if row['scenario_id']!='nominal']
        require(len(rows)==13 and len(nominal)==1 and len(nonzero)==12, 'per-task all13 outcome preservation')
        expected_tasks.append(dict(task_id=task,nominal_primary_passed=nominal[0]['primary']['joint_success'],
            nonzero_cases=12,primary_passed=sum(row['primary']['joint_success'] for row in nonzero),
            secondary_passed=sum(row['secondary']['joint_success'] for row in nonzero),
            physical_passed=sum(row['physical_gate_passed'] for row in nonzero),
            differing_success_cases=[row['scenario_id'] for row in nonzero if row['primary']['joint_success']!=row['secondary']['joint_success']],
            primary_failures=[row['scenario_id'] for row in nonzero if not row['primary']['joint_success']]))
    require(summary['tasks']==expected_tasks, 'summary task pass/failure counts differ from all48 fixed nonzero outcomes')
    return dict(exact52_selected_rows=True, all48_perturbed_selected_rows_preserved=True,
                task_counts_independently_recomputed=True, physical_trials=78, control_steps=15600, physics_steps=312000)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.panel,arguments.output)
    print(json.dumps({key:result[key] for key in ('audit_passed','errors','new_physical_trials','new_llm_calls')}))
    raise SystemExit(0 if result['audit_passed'] else 1)
