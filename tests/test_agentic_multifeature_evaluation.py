import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
m=load('multifeature',ROOT/'experiments/agentic/evaluate_multifeature.py')
fixtures=load('old_fixture',ROOT/'tests/test_agentic_evaluation.py')
P=dict(parameter_names=m.PARAMETER_NAMES,bounds=[[.35,1.6]]*4+[[-.3,.3]],feature_windows_s=[[.8,1.6],[2.4,3.2]],
       accent_window_s=[1.8,3.2],feature_tolerances=[.08]*4+[.1],physical_gate=dict(min_root_height_m=.6,max_mean_joint_rmse_rad=.35))
G=dict(feature_targets=[.7]*4+[2.5])

def trial(path,amplitude=1.):
    directory=fixtures.trial(path,amplitude)
    with np.load(directory/'reference_cache.npz') as archive: cache=dict(archive)
    cache['committed__dof_vel']=np.zeros((201,29))
    np.savez(directory/'reference_cache.npz',**cache); fixtures.rehash(directory)
    return directory

def peak_fixture(path,amplitude=1.):
    directory=trial(path,amplitude)
    velocity=np.load(directory/'qvel.npy'); times=np.arange(201)*.02
    velocity[:,6:]=np.exp(-((times-2.5)/.12)**2)[:,None]*2; velocity[0]=0
    np.save(directory/'qvel.npy',velocity)
    with np.load(directory/'policy_inputs.npz') as archive: feeds=dict(archive)
    for i in range(1,201): feeds[f'call_{i:04d}__current_dof_vel']=velocity[i-1,6:][None].astype('float32')
    np.savez(directory/'policy_inputs.npz',**feeds); fixtures.rehash(directory)
    return directory

def test_physical_features_not_reference_and_order(tmp_path):
    baseline=peak_fixture(tmp_path/'base'); candidate=peak_fixture(tmp_path/'candidate',.7)
    result=m.evaluate_multifeature(candidate,baseline,P,G)
    assert result['integrity_passed'],result
    assert result['physical_gate_passed'] and result['intent_passed']
    assert np.allclose(result['features'],[.7]*4+[2.5])
    assert [r['name'] for r in result['dimensions']]==['left_ratio_a','right_ratio_a','left_ratio_b','right_ratio_b']
    assert all(row['reference_rms_rad']==0 for row in result['dimensions'])
    assert all(name.startswith('left_') for name in result['dimensions'][0]['joint_names'])

def test_missing_array_fails_closed(tmp_path):
    baseline=peak_fixture(tmp_path/'base'); candidate=peak_fixture(tmp_path/'candidate')
    (candidate/'qvel.npy').unlink(); fixtures.rehash(candidate)
    result=m.evaluate_multifeature(candidate,baseline,P,G)
    assert not result['integrity_passed'] and not result['physical_gate_passed']

def test_absent_peak_cannot_pass_intent(tmp_path):
    baseline=trial(tmp_path/'base'); candidate=trial(tmp_path/'candidate',.7)
    result=m.evaluate_multifeature(candidate,baseline,P,G)
    assert result['integrity_passed'] and result['physical_gate_passed']
    assert not result['intent_passed'] and result['features'][4] is None and result['utility'] is None

def test_physics_separate_from_intent(tmp_path):
    baseline=peak_fixture(tmp_path/'base'); candidate=peak_fixture(tmp_path/'candidate',.7)
    strict={**P,'physical_gate':dict(min_root_height_m=.9,max_mean_joint_rmse_rad=.35)}
    result=m.evaluate_multifeature(candidate,baseline,strict,G)
    assert result['intent_passed'] and not result['physical_gate_passed']

def test_jacobian_linear_inverse_bounds_and_budget():
    history=[]; matrix=np.diag([.7,.8,.9,1.,1.])
    target=[.6,.65,.8,.9,2.55]; goal=dict(feature_targets=target)
    def response(x): return matrix@np.asarray(x)+[0,0,0,0,2.5]
    for _ in range(8):
        x=m.next_jacobian_candidate(history,P,goal)
        assert np.all(np.asarray(x)>=np.asarray(P['bounds'])[:,0])
        assert np.all(np.asarray(x)<=np.asarray(P['bounds'])[:,1])
        history.append(dict(parameters=x,features=response(x).tolist()))
    assert np.max(np.abs(np.asarray(history[6]['features'])-target))<1e-5
    with pytest.raises(ValueError,match='budget'): m.next_jacobian_candidate(history,P,goal)

def test_nested_manifest_verified_and_tampering_rejected(tmp_path):
    baseline=peak_fixture(tmp_path/'base'); candidate=peak_fixture(tmp_path/'candidate',.7)
    for directory in (baseline,candidate):
        nested=directory/'common_reference'; nested.mkdir(); fixtures.write_json(nested/'manifest.json',{'schema':'fixture'})
        manifest=__import__('json').loads((directory/'manifest.json').read_text())
        manifest['artifact_hashes']={str(p.relative_to(directory)):m._legacy._sha(p) for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
        fixtures.write_json(directory/'manifest.json',manifest)
    assert m.evaluate_multifeature(candidate,baseline,P,G)['integrity_passed']
    (candidate/'common_reference'/'manifest.json').write_text('{}')
    assert not m.evaluate_multifeature(candidate,baseline,P,G)['integrity_passed']

def test_optimizer_rejects_missing_observation_and_clips_unreachable_goal():
    history=[]
    for _ in range(7):
        x=m.next_jacobian_candidate(history,P,{'feature_targets':[100]*5})
        history.append(dict(parameters=x,features=x))
    assert np.all(np.asarray(history[-1]['parameters'])<=np.asarray(P['bounds'])[:,1])
    history[0]['features']=None
    with pytest.raises(ValueError): m.next_jacobian_candidate(history,P,G)

def test_new_schema_requires_fresh_fk_observation_match(tmp_path,monkeypatch):
    directory=peak_fixture(tmp_path/'new')
    xml_name='research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml'
    xml=directory/'source_snapshot'/xml_name; xml.parent.mkdir(parents=True); xml.write_text('<mujoco/>')
    manifest=__import__('json').loads((directory/'manifest.json').read_text())
    manifest.update(schema='pulsemorph.agentic_multifeature_trial.v1',control_boundary_forward_count=200)
    manifest['preexecution_input_hashes'][xml_name]=m._legacy._sha(xml)
    fixtures.write_json(directory/'preexecution_inputs.json',manifest['preexecution_input_hashes'])
    fixtures.write_json(directory/'manifest.json',manifest); fixtures.rehash(directory)
    monkeypatch.setattr(m,'_fresh_fk_orientations',lambda *args:np.tile([0,0,0,1.],(200,1)))
    assert m._audit(directory)['fresh_anchor_audit']['max_rotation_error_rad']==0
    monkeypatch.setattr(m,'_fresh_fk_orientations',lambda *args:np.tile([0,0,.01,np.sqrt(1-.01**2)],(200,1)))
    with pytest.raises(ValueError,match='same-time'): m._audit(directory)
    manifest['control_boundary_forward_count']=199
    fixtures.write_json(directory/'manifest.json',manifest); fixtures.rehash(directory)
    with pytest.raises(ValueError,match='forward count'): m._audit(directory)

def test_declared_prominence_and_committed_derivative_are_used(tmp_path):
    baseline=peak_fixture(tmp_path/'base'); candidate=peak_fixture(tmp_path/'candidate',.7)
    with np.load(candidate/'reference_cache.npz') as archive: cache=dict(archive)
    times=np.arange(201)*.02
    cache['committed__dof_vel']=np.exp(-((times-2.7)/.1)**2)[:,None]*np.ones((1,29))*3
    np.savez(candidate/'reference_cache.npz',**cache); fixtures.rehash(candidate)
    result=m.evaluate_multifeature(candidate,baseline,P,G)
    assert result['accent']['reference']['peak_time_s']==2.7
    result=m.evaluate_multifeature(candidate,baseline,{**P,'accent_minimum_prominence':4},G)
    assert result['integrity_passed'] and not result['accent']['available']
