import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('agentic_evaluation', Path(__file__).resolve().parents[1]/'experiments/agentic/evaluate_trial.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

PROTOCOL = dict(task=dict(target_ratio=.7, measurement_window_s=[1.5,2.5]),
                gates=dict(minimum_root_height_m=.6, maximum_joint_rmse_rad=.35), evaluation_beats_s=[])


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def rehash(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['artifact_hashes'] = {str(p.relative_to(directory)): evaluation._sha(p)
                                 for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    write_json(directory/'manifest.json', manifest)


def trial(directory, amplitude=1.):
    directory.mkdir()
    times = np.arange(201)*.02
    q = np.zeros((201,36)); q[:,2] = .8; q[:,3] = 1
    q[:,22:] = amplitude*np.sin(times[:,None])* .1
    v = np.zeros((201,35)); v[:,21:] = amplitude*np.cos(times[:,None])*.1; v[0] = 0
    for name, value in dict(qpos=q,qvel=v,state_times=times,applied_joint_targets=np.zeros((200,29))).items():
        np.save(directory/(name+'.npy'),value)
    sub = np.zeros((4000,10)); sub[:,0]=np.arange(1,4001); sub[:,1]=sub[:,0]*.001; sub[:,2]=.8; sub[:,8:]=1
    np.savez(directory/'substep_telemetry.npz',values=sub)
    names = [f'leg_{i}' for i in range(15)] + [f'{side}_{part}_joint' for side in ('left','right') for part in
             ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
    write_json(directory/'motion_reference_draft.json',dict(joint_names=names,body_names=['torso_link']))
    cache = dict(committed__dof_pos=np.zeros((201,29),dtype='float32'),forecast__dof_pos=np.zeros((200,4,29),dtype='float32'),
                 forecast__dof_vel=np.zeros((200,4,29),dtype='float32'),committed__body_rot=np.tile([0,0,0,1.],(201,1,1)).astype('float32'),
                 forecast__body_rot=np.tile([0,0,0,1.],(200,4,1,1)).astype('float32'))
    np.savez(directory/'reference_cache.npz',**cache)
    feeds,outputs,calls={},{},[]
    for i in range(201):
        calls.append(dict(call_index=i,role='control' if i else 'warmup',control_index=i-1 if i else None,all_finite=True,elapsed_ms=1))
        if not i: continue
        prefix=f'call_{i:04d}__'
        feeds.update({prefix+'current_anchor_rot':np.array([[0,0,0,1]],dtype='float32'),
                      prefix+'mimic_future_anchor_rot':cache['forecast__body_rot'][i-1,:,0][None],
                      prefix+'mimic_future_dof_pos':cache['forecast__dof_pos'][i-1][None],
                      prefix+'mimic_future_dof_vel':cache['forecast__dof_vel'][i-1][None],
                      prefix+'current_dof_pos':q[i-1,7:][None].astype('float32'),
                      prefix+'current_dof_vel':v[i-1,6:][None].astype('float32')})
        outputs[prefix+'joint_pos_targets']=np.zeros((1,29))
    np.savez(directory/'policy_inputs.npz',**feeds); np.savez(directory/'policy_outputs.npz',**outputs)
    write_json(directory/'policy_calls.json',calls)
    snapshot=directory/'source_snapshot'/'fixture.txt'; snapshot.parent.mkdir(); snapshot.write_text('fixed')
    pinned={'fixture.txt':evaluation._sha(snapshot)}
    write_json(directory/'preexecution_inputs.json',pinned)
    manifest=dict(status='success',actual_control_steps=200,expected_control_steps=200,actual_physics_steps=4000,
        expected_physics_steps=4000,instrumented_set_initial_pose_call_count=1,mujoco_warning_counts_before=[0],
        mujoco_warning_counts_after=[0],all_substeps_finite=True,all_substeps_time_contiguous=True,
        control_dt=.02,physics_dt=.001,decimation=20,preexecution_input_hashes=pinned,
        onnx_sha256='model',motion_sha256='motion',physics_contract={},wall_seconds=4,
        versions={'mujoco':'fixture-1','onnxruntime':'fixture-1','numpy':np.__version__})
    manifest.update(online_onnx_call_count=200,warmup_onnx_call_count=1,preexecution_hashes_unchanged=True)
    write_json(directory/'manifest.json',manifest); rehash(directory)
    return directory


def test_intent_uses_physical_qpos_and_missing_beats_are_unavailable(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate',.7)
    result=evaluation.evaluate_trial(candidate,baseline,PROTOCOL)
    assert result['integrity_passed'],result
    assert result['physical_gate_passed'] and result['intent_passed']
    assert abs(result['intent_ratio']-.7)<1e-12
    assert result['music']['available'] is False
    assert 'nearest_beat' not in result['music']


def test_partial_state_arrays_fail_closed(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate')
    np.save(candidate/'qpos.npy',np.zeros((200,36))); rehash(candidate)
    result=evaluation.evaluate_trial(candidate,baseline,PROTOCOL)
    assert not result['integrity_passed'] and not result['physical_gate_passed']


def test_future_velocity_mismatch_even_with_fresh_artifact_hash(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate')
    with np.load(candidate/'policy_inputs.npz') as archive: feeds=dict(archive)
    feeds['call_0002__mimic_future_dof_vel'][0,0,0]=1
    np.savez(candidate/'policy_inputs.npz',**feeds); rehash(candidate)
    result=evaluation.evaluate_trial(candidate,baseline,PROTOCOL)
    assert not result['integrity_passed']
    assert 'mimic_future_dof_vel' in result['integrity_errors'][0]


def test_hash_tampering_is_rejected(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate')
    (candidate/'source_snapshot'/'fixture.txt').write_text('changed')
    assert not evaluation.evaluate_trial(candidate,baseline,PROTOCOL)['integrity_passed']


@pytest.mark.parametrize('name', ['current_dof_pos','current_dof_vel'])
def test_current_sensor_mismatch_rejected(tmp_path,name):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate')
    with np.load(candidate/'policy_inputs.npz') as archive: feeds=dict(archive)
    feeds['call_0002__'+name][0,0]+=1
    np.savez(candidate/'policy_inputs.npz',**feeds); rehash(candidate)
    result=evaluation.evaluate_trial(candidate,baseline,PROTOCOL)
    assert not result['integrity_passed']
    assert name in result['integrity_errors'][0]


def test_pd_output_applied_target_mismatch_rejected(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate')
    targets=np.load(candidate/'applied_joint_targets.npy'); targets[1,0]=1
    np.save(candidate/'applied_joint_targets.npy',targets); rehash(candidate)
    result=evaluation.evaluate_trial(candidate,baseline,PROTOCOL)
    assert not result['integrity_passed']
    assert 'PD output' in result['integrity_errors'][0]


def test_explicit_beats_have_separate_proxy_metrics(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate',.7)
    protocol={**PROTOCOL,'evaluation_beats_s':[.5,1.,1.5,2.,2.5,3.,3.5]}
    result=evaluation.evaluate_trial(candidate,baseline,protocol)
    assert result['integrity_passed'] and result['music']['available']
    assert result['music']['one_to_one']['expected_count']==7
    assert 'nearest_beat' in result['music']


def test_runtime_version_change_rejected_even_when_artifacts_are_valid(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate',.7)
    manifest=json.loads((candidate/'manifest.json').read_text())
    manifest['versions']['mujoco']='fixture-2'
    write_json(candidate/'manifest.json',manifest)
    result=evaluation.evaluate_trial(candidate,baseline,PROTOCOL)
    assert not result['integrity_passed']
    assert 'runtime versions mismatch' in result['integrity_errors'][0]


def test_pinned_profile_change_rejected_even_with_consistent_snapshot(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate',.7)
    snapshot=candidate/'source_snapshot'/'fixture.txt'; snapshot.write_text('new controller profile')
    pinned={'fixture.txt':evaluation._sha(snapshot)}
    write_json(candidate/'preexecution_inputs.json',pinned)
    manifest=json.loads((candidate/'manifest.json').read_text())
    manifest['preexecution_input_hashes']=pinned
    write_json(candidate/'manifest.json',manifest); rehash(candidate)
    result=evaluation.evaluate_trial(candidate,baseline,PROTOCOL)
    assert not result['integrity_passed']
    assert 'pinned controller/source profile mismatch' in result['integrity_errors'][0]


def test_protocol_tolerance_controls_intent_acceptance(tmp_path):
    baseline=trial(tmp_path/'baseline'); candidate=trial(tmp_path/'candidate',.73)
    strict={**PROTOCOL,'task':{**PROTOCOL['task'],'tolerance':.01}}
    result=evaluation.evaluate_trial(candidate,baseline,strict)
    assert result['integrity_passed'] and not result['intent_passed']
    assert result['intent_tolerance']==.01
    assert evaluation.evaluate_trial(candidate,baseline,PROTOCOL)['intent_passed']
