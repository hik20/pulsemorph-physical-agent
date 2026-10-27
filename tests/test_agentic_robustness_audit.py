"""Audit tamper tests; no policy execution or physics stepping."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/agentic'))
import audit_robustness as audit
import execute_robustness as runtime
from robustness_scenarios import build_scenarios


def physical_model():
    children = ''.join(f'<body pos="0 0 .03"><joint name="j{i}" range="-1 1"/><geom type="sphere" size=".01" mass=".01"/></body>' for i in range(29))
    return mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody><body name="torso_link" pos="0 0 1"><freejoint/><geom type="sphere" size=".03" mass="1"/>'+children+'</body></worldbody></mujoco>')


def initial_fixture(tmp_path, scenario_index=11):
    original, trial = tmp_path/'original', tmp_path/'trial'
    original.mkdir(); trial.mkdir()
    model = physical_model(); data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(.3), np.sin(.3), 0., 0.]
    mujoco.mj_forward(model, data)
    q, v = data.qpos.copy(), data.qvel.copy()
    np.save(original/'qpos.npy', q[None]); np.save(original/'qvel.npy', v[None])
    scenario = build_scenarios([f'j{i}' for i in range(29)])[scenario_index]
    record = runtime.apply_initial_perturbation(model, data, scenario['initial_perturbation'])
    manifest = dict(initial_perturbation=record, instrumented_set_initial_pose_call_count=1,
        actual_initial_qpos_sha256=runtime.sha256_array(data.qpos),
        actual_initial_qvel_sha256=runtime.sha256_array(data.qvel),
        mujoco_warning_counts_before=record['warning_counts_after_forward'])
    (trial/'manifest.json').write_text(json.dumps(manifest))
    np.save(trial/'qpos.npy', np.repeat(data.qpos[None], 201, axis=0))
    np.save(trial/'qvel.npy', np.repeat(data.qvel[None], 201, axis=0))
    np.save(trial/'state_times.npy', np.arange(201)*.02)
    feeds = {'call_0001__current_dof_pos': data.qpos[7:][None].astype('float32'),
             'call_0001__current_dof_vel': data.qvel[6:][None].astype('float32')}
    feeds.update({f'call_{i:04d}__current_root_local_ang_vel': data.qvel[3:6][None].astype('float32') for i in range(1,201)})
    np.savez(trial/'policy_inputs.npz', **feeds)
    return trial, original, scenario


@pytest.mark.parametrize('scenario_index', [0,1,2,3,4,5,6,7,8,9,10,11,12])
def test_independent_initial_reconstruction_matches_actual_mujoco_injection(tmp_path, scenario_index):
    trial, original, scenario = initial_fixture(tmp_path, scenario_index)
    result = audit.check_initial_state(trial, original, scenario)
    assert result['reconstructed_initial_state_bytes'] is True
    assert result['nonzero'] is (scenario_index != 0)


def test_sign_tamper_is_rejected(tmp_path):
    trial, original, scenario = initial_fixture(tmp_path)
    wrong = copy.deepcopy(scenario)
    wrong['initial_perturbation']['joint_position_offset_rad'][0] *= -1
    with pytest.raises(ValueError, match='spec differs'):
        audit.check_initial_state(trial, original, wrong)


def test_actual_initial_array_tamper_is_rejected(tmp_path):
    trial, original, scenario = initial_fixture(tmp_path)
    q = np.load(trial/'qpos.npy'); q[0,7] *= -1; np.save(trial/'qpos.npy', q)
    with pytest.raises(ValueError, match='initial qpos array'):
        audit.check_initial_state(trial, original, scenario)


def test_all200_angular_velocity_feed_tamper_is_rejected(tmp_path):
    trial, original, scenario = initial_fixture(tmp_path)
    with np.load(trial/'policy_inputs.npz') as z: feeds = dict(z)
    feeds['call_0100__current_root_local_ang_vel'][0,2] = .1
    np.savez(trial/'policy_inputs.npz', **feeds)
    with pytest.raises(ValueError, match='angular velocity'):
        audit.check_initial_state(trial, original, scenario)


def test_joint_name_order_tamper_rejected():
    names = [f'j{i}' for i in range(29)]
    scenarios = build_scenarios(names)
    audit.check_scenarios(scenarios, names)
    with pytest.raises(ValueError, match='sign/seed/name/vector'):
        audit.check_scenarios(scenarios, names[::-1])
    scenarios[1]['initial_perturbation']['joint_position_offset_rad'][0] *= -1
    with pytest.raises(ValueError, match='sign/seed/name/vector'):
        audit.check_scenarios(scenarios, names)


def test_source_pin_only_declared_runtime_difference():
    old_pins = {audit.OLD_RUNTIME: 'a'*64, 'shared.py': 'b'*64}
    new_pins = {'shared.py': 'b'*64, audit.RUNTIME: audit.sha(ROOT/audit.RUNTIME),
               audit.COMPATIBLE_RUNTIME: audit.sha(ROOT/audit.COMPATIBLE_RUNTIME)}
    common = dict.fromkeys(('onnx_sha256','motion_sha256','body_profile_sha256','controller_profile_sha256',
                            'checkpoint_sha256','physics_contract','versions'), 'same')
    old = dict(common, preexecution_input_hashes=old_pins)
    new = dict(common, preexecution_input_hashes=new_pins)
    assert audit.check_input_pins(old,new)['shared_inputs_identical']
    new_pins['shared.py'] = 'c'*64
    with pytest.raises(ValueError, match='shared source pin changed'):
        audit.check_input_pins(old,new)


def test_center_definitions_and_denominator_tamper_rejected():
    from test_agentic_robustness_evaluation import RobustnessEvaluationTests
    fixture = RobustnessEvaluationTests()
    fixture.setUp()
    evaluation = fixture.evaluate()
    args = [fixture.candidate['arrays'], fixture.nominal['arrays'], fixture.condition['arrays'],
            fixture.candidate['names'], fixture.protocol, fixture.goal]
    assert audit.check_center_definitions(evaluation, *args)['primary_fixed_nominal_center_and_denominator']
    changed = copy.deepcopy(evaluation)
    changed['initial']['primary_center'] = changed['initial']['secondary_center']
    with pytest.raises(ValueError, match='center definitions'):
        audit.check_center_definitions(changed, *args)
    changed = copy.deepcopy(evaluation)
    changed['dimensions'][0]['nominal_baseline_rms_rad'] = changed['dimensions'][0]['condition_baseline_rms_rad']
    with pytest.raises(ValueError, match='center/denominator'):
        audit.check_center_definitions(changed, *args)


def test_runtime_hook_source_proves_order_and_no_perstep_writes(tmp_path):
    source = ROOT/audit.RUNTIME
    assert audit.check_runtime_hook_source(source)['no_per_step_injection_or_state_writes']
    changed = source.read_text().replace('        def step(model, data):', '        def step(model, data):\n            data.qvel[:] = 0')
    path = tmp_path/'tampered.py'; path.write_text(changed)
    with pytest.raises(ValueError, match='per-step direct state writes'):
        audit.check_runtime_hook_source(path)


def test_calibration_representation_has_no_scored_targets():
    result = audit.calibration_measurements(dict(mean_joint_rmse_rad=.1, sub=np.array([[0,0,.8]]),
        manifest=dict(actual_initial_qpos_sha256='a'*64,actual_initial_qvel_sha256='b'*64)))
    assert result['goal'] is None and result['target_features'] is None
    assert result['scored_against_target'] is False
    assert 'intent_passed' not in result and 'utility' not in result


def test_reconstruction_world_yaw_is_left_and_zero_preserves_nonunit_bits():
    q = np.zeros(36); q[3:7] = [np.cos(.4), np.sin(.4), 0., 0.]
    v = np.zeros(35); v[3:6] = [.1, -.2, .3]
    spec = build_scenarios([f'j{i}' for i in range(29)])[9]['initial_perturbation']
    out, velocity, nonzero = audit.reconstruct_initial_state(q,v,spec)
    expected = np.zeros(4); wrong = np.zeros(4)
    rotation = np.array([np.cos(spec['root_yaw_offset_rad']/2),0.,0.,np.sin(spec['root_yaw_offset_rad']/2)])
    mujoco.mju_mulQuat(expected, rotation, q[3:7])
    mujoco.mju_mulQuat(wrong, q[3:7], rotation)
    np.testing.assert_array_equal(out[3:7], expected)
    assert not np.array_equal(out[3:7], wrong)
    np.testing.assert_array_equal(velocity[3:6], v[3:6])
    q[3] = np.nextafter(q[3], np.inf)
    zero = build_scenarios([f'j{i}' for i in range(29)])[0]['initial_perturbation']
    out, velocity, nonzero = audit.reconstruct_initial_state(q,v,zero)
    assert not nonzero and out.tobytes() == q.tobytes() and velocity.tobytes() == v.tobytes()


def test_first_anchor_fk_exact_bytes_and_tamper(tmp_path):
    trial, original, scenario = initial_fixture(tmp_path)
    model = physical_model()
    xml_name = 'custom/g1_holo_compat.xml'
    xml = trial/'source_snapshot'/xml_name; xml.parent.mkdir(parents=True)
    mujoco.mj_saveLastXML(str(xml), model)
    manifest = json.loads((trial/'manifest.json').read_text())
    manifest['preexecution_input_hashes'] = {xml_name: audit.sha(xml)}
    (trial/'manifest.json').write_text(json.dumps(manifest))
    data = mujoco.MjData(model); data.qpos[:] = np.load(trial/'qpos.npy')[0]
    mujoco.mj_kinematics(model, data)
    anchor = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
    with np.load(trial/'policy_inputs.npz') as z: feeds = dict(z)
    feeds['call_0001__current_anchor_rot'] = data.xquat[anchor][[1,2,3,0]][None].astype('float32')
    np.savez(trial/'policy_inputs.npz', **feeds)
    base = {'fresh_anchor_audit': {'control_rows': 200}}
    assert audit.check_initial_state(trial, original, scenario, base)['first_policy_joint_inputs_exact']
    feeds['call_0001__current_anchor_rot'][0,0] += np.float32(.000001)
    np.savez(trial/'policy_inputs.npz', **feeds)
    with pytest.raises(ValueError, match='first current anchor FK'):
        audit.check_initial_state(trial, original, scenario, base)


def test_execution_chronology_requires_all_six_zero_before_perturbations(tmp_path):
    from datetime import datetime, timedelta, timezone
    scenarios = build_scenarios([f'j{i}' for i in range(29)])
    identities = [s['scenario_id']+'__'+role for s in scenarios for role in audit.ROLES]
    nominal = ['nominal__'+role for role in audit.ROLES]
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'nominal_parity.json').write_text('{}')
    (tmp_path/'proposals').mkdir()
    events = [dict(event='panel_start',protocol_sha256=audit.sha(tmp_path/'protocol.json'))]
    for index, identity in enumerate(identities):
        trial = tmp_path/'trials'/identity; trial.mkdir(parents=True)
        (trial/'manifest.json').write_text('{}')
        (tmp_path/'proposals'/(identity+'.json')).write_text('{}')
        events.append(dict(event='trial_start',trial_id=identity,proposal_sha256=audit.sha(tmp_path/'proposals'/(identity+'.json'))))
        events.append(dict(event='trial_complete',trial_id=identity,status='success',manifest_sha256=audit.sha(trial/'manifest.json'),control_steps=200,physics_steps=4000))
        if index==5:
            events.append(dict(event='nominal_gate',passed=True,nominal_parity_sha256=audit.sha(tmp_path/'nominal_parity.json')))
    events.append(dict(event='panel_complete',completed=True,physical_trials=78))
    for index,event in enumerate(events):
        event['time_utc'] = (datetime(2026,10,27,tzinfo=timezone.utc)+timedelta(seconds=index)).isoformat()
    def write(rows): (tmp_path/'execution_journal.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    write(events)
    assert audit.check_chronology(tmp_path,identities,nominal)['nominal_all_six_before_any_perturbation']
    tampered = copy.deepcopy(events)
    tampered[12]['trial_id'] = identities[6]  # Sixth nominal completion falsely attributes a perturbed run.
    write(tampered)
    with pytest.raises(ValueError): audit.check_chronology(tmp_path,identities,nominal)


def test_summary_keeps_all48_nonzero_results_and_exact_failure_counts(tmp_path):
    from robustness_driver import summarize_rows
    (tmp_path/'protocol.json').write_text('{}')
    scenarios = build_scenarios([f'j{i}' for i in range(29)])
    cases = [dict(role='selected_'+task,task_id=task,source_id=task.split('_')[0]) for task in ('wave_a','wave_b','high5_a','high5_b')]
    audit_rows, driver_rows = [], []
    for index,scenario in enumerate(scenarios):
        for case in cases:
            identity = scenario['scenario_id']+'__'+case['role']
            evaluation = dict(primary={'joint_success':index%2==0},secondary={'joint_success':index%3==0},
                physical_gate_passed=True,min_root_height_m=.8,mean_joint_rmse_rad=.1)
            audit_rows.append(dict(candidate_id=identity,scenario_id=scenario['scenario_id'],role=case['role'],evaluation=evaluation))
            driver_rows.append(dict(trial_id=identity,task_id=case['task_id'],source_id=case['source_id'],scenario_id=scenario['scenario_id'],**evaluation))
    summary = dict(schema='pulsemorph.agentic_robustness_summary.v1',panel_completed=True,errors=[],
        protocol_sha256=audit.sha(tmp_path/'protocol.json'),physical_trials=78,control_steps=15600,physics_steps=312000,
        new_llm_calls=0,training_admission=False,source_pins_unchanged=True,rows=driver_rows,tasks=summarize_rows(driver_rows))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    assert audit.check_summary(tmp_path,{'cases':cases},audit_rows)['all48_perturbed_selected_rows_preserved']
    summary['rows'].pop()
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='excludes/changes'):
        audit.check_summary(tmp_path,{'cases':cases},audit_rows)
