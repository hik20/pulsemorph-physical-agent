"""Two-tool workflow isolation, provenance and parity; no physical execution."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
driver=importlib.import_module('tool_selection_driver_v2')
REAL_VALIDATE_CALIBRATIONS=driver.validate_calibrations


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


@pytest.fixture
def pilot(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'verify_code',lambda protocol:None)
    source={'path':'trusted.motion','sha256':'a'*64}
    tasks=[dict(task_id=f'{s}_{v}',source_id=s,source_motion=source,instruction='controlled task',
                goal={'feature_targets':[.8,1.2,1.2,.8,2.35]},music_observation={'fresh':True})
           for s in ['wave','high5'] for v in ['a','b']]
    protocol=dict(schema='pulsemorph.agentic_tool_selection_protocol.v2',tasks=tasks,anchor=[1.,1.,1.,1.,0.],bounds=[[.35,1.6]]*4+[[-.3,.3]],
                  parameter_names=driver.NAMES,feature_names=['lA','rA','lB','rB','accent'],feature_tolerances=[.08]*4+[.1],
                  physical_gate={},selection={},section_windows_s=[[.4,2],[2,3.6]],feature_windows_s=[[.8,1.6],[2.4,3.2]])
    save(tmp_path/'protocol.json',protocol)
    monkeypatch.setattr(driver,'validate_calibrations',lambda out:[])
    for source_id in ['wave','high5']:
        save(tmp_path/f'evaluations/calibration_{source_id}.json',
             dict(candidate_id='calibration_'+source_id,control_tool=driver.TOOLS[0],features=[1,1,1,1,2.5],integrity_passed=True,physical_gate_passed=True))
    return tmp_path


def respond(out,task,method,index,tool=None,values=None):
    save(driver.response_path(out,task,method,index),dict(control_tool=tool or driver.TOOLS[1],
         parameters=values or [.8,1.2,1.2,.8,.1],
         only_observation_path=str(driver.observation_path(out,task,method,index).resolve()),brief_reason='test'))


def test_free_first_tool_and_parameters_and_runtime_producer(pilot):
    driver.observation(pilot,'wave_a','llm_feedback_off',0)
    respond(pilot,'wave_a','llm_feedback_off',0)
    proposal=driver.commit_model(pilot,'wave_a','llm_feedback_off',0)
    assert proposal['control_tool']==driver.TOOLS[1]
    assert list(proposal['parameters'].values())==[.8,1.2,1.2,.8,.1]
    assert driver.observation_data(pilot,'wave_a','llm_feedback_off',1)['forced_first_parameters'] is None
    runtime=importlib.import_module('execute_tool_selection')
    producer=runtime.proposal_producer(proposal)
    assert producer['model']=='gpt-6.1-sol'
    assert driver.compact({'control_tool':driver.TOOLS[1]})['control_tool']==driver.TOOLS[1]


@pytest.mark.parametrize('change',[
    {'control_tool':'unknown'}, {'parameters':[[1,1,1,1,0]]}, {'parameters':[True,1,1,1,0]},
    {'parameters':[1,1,1,1,.9]}, {'extra':1}, {'brief_reason':None},
])
def test_strict_schema_aborts_preserving_raw_response(pilot,change):
    driver.observation(pilot,'wave_a','llm_feedback',0)
    respond(pilot,'wave_a','llm_feedback',0)
    path=driver.response_path(pilot,'wave_a','llm_feedback',0)
    raw=driver.read(path);raw.update(change);save(path,raw)
    original=path.read_bytes()
    with pytest.raises(ValueError):driver.commit_model(pilot,'wave_a','llm_feedback',0)
    assert path.read_bytes()==original
    assert driver.read(pilot/'abort.json')['slot_consumed'] is True
    assert not (pilot/'proposals/wave_a__llm_feedback__0.json').exists()
    with pytest.raises(RuntimeError,match='aborted'):driver.observation(pilot,'high5_a','llm_feedback',0)


def test_off_never_reads_candidate_outcomes_and_only_own_history(pilot,monkeypatch):
    driver.observation(pilot,'wave_a','llm_feedback_off',0)
    respond(pilot,'wave_a','llm_feedback_off',0)
    driver.commit_model(pilot,'wave_a','llm_feedback_off',0)
    original=driver.read
    def guarded(path):
        if 'evaluations' in str(path) and 'calibration_' not in str(path):
            raise AssertionError('OFF accessed candidate outcome')
        if 'high5_a__' in str(path):raise AssertionError('sibling history accessed')
        return original(path)
    monkeypatch.setattr(driver,'read',guarded)
    obs=driver.observation_data(pilot,'wave_a','llm_feedback_off',1)
    assert obs['prior_trial_results']==[]
    assert len(obs['prior_proposals'])==1
    assert obs['prior_proposals'][0]['control_tool']==driver.TOOLS[1]
    assert [t['control_tool'] for t in obs['tools']]==list(driver.TOOLS)


def test_all32_off_committed_and_revalidated_before_candidate_physics(pilot,monkeypatch):
    protocol=driver.read(pilot/'protocol.json')
    with pytest.raises(FileNotFoundError):driver.require_precommit(pilot,protocol)
    for task in protocol['tasks']:
        name=task['task_id'];driver.observation(pilot,name,'llm_feedback_off',0)
        for i in range(8):
            respond(pilot,name,'llm_feedback_off',i)
            driver.commit_model(pilot,name,'llm_feedback_off',i)
    validations=[]
    monkeypatch.setattr(driver,'validate_calibrations',lambda out:validations.append(out))
    driver.require_precommit(pilot,protocol)
    assert validations==[pilot]
    path=driver.response_path(pilot,'high5_b','llm_feedback_off',7)
    raw=driver.read(path);raw['control_tool']=driver.TOOLS[0];save(path,raw)
    monkeypatch.setattr(driver,'execute_trial',lambda *a:pytest.fail('physics must not run'))
    with pytest.raises(ValueError,match='artifact'):driver.run(pilot,'wave_a','surrogate_search',0)
    assert driver.read(pilot/'abort.json')['retry_allowed'] is False


def test_off_commit_after_candidate_directory_rejected(pilot):
    driver.observation(pilot,'wave_a','llm_feedback_off',0)
    respond(pilot,'wave_a','llm_feedback_off',0)
    (pilot/'trials'/'wave_b__surrogate_search__0').mkdir(parents=True)
    with pytest.raises(ValueError,match='precede'):driver.commit_model(pilot,'wave_a','llm_feedback_off',0)


def test_cross_stream_observation_and_hidden_outcomes_rejected(pilot):
    driver.observation(pilot,'wave_a','llm_feedback_off',0)
    respond(pilot,'wave_a','llm_feedback_off',0)
    obs=driver.observation_path(pilot,'wave_a','llm_feedback_off',0)
    raw=driver.read(obs);raw['prior_trial_results']=[{'candidate_id':'old_label'}];save(obs,raw)
    with pytest.raises(ValueError,match='history'):driver.commit_model(pilot,'wave_a','llm_feedback_off',0)


def make_calibrations(out):
    """Tiny complete fresh fixture; parity validator itself remains unmocked."""
    protocol=driver.read(out/'protocol.json')
    for source in ['wave','high5']:
        task=next(t for t in protocol['tasks'] if t['source_id']==source)
        for identity,tool in zip(driver.calibration_ids(source),driver.TOOLS):
            d=out/'trials'/identity;d.mkdir(parents=True)
            p=dict(source_motion=task['source_motion'],parameters=dict(zip(driver.NAMES,protocol['anchor'])),
                   control_tool=tool,protocol_sha256=driver.sha256(out/'protocol.json'))
            save(out/f'proposals/{identity}.json',p);save(d/'proposal.json',p)
            for name in ['qpos','qvel','state_times','applied_joint_targets']:
                np.save(d/(name+'.npy'),np.ones((3,2)))
            (d/'common_reference').mkdir()
            for name in ['reference_cache.npz','policy_inputs.npz','common_reference/arrays.npz']:
                np.savez_compressed(d/name,q=np.ones((3,2)),dq=np.zeros((3,2)),anchor=np.ones((3,4)))
            pins={'code':'b'*64};save(d/'preexecution_inputs.json',pins)
            manifest=dict(status='success',control_tool=tool,preexecution_input_hashes=pins,
                          onnx_sha256='c'*64,motion_sha256=task['source_motion']['sha256'],physics_contract={},versions={})
            manifest['artifact_hashes']={str(p.relative_to(d)):driver.sha256(p) for p in d.rglob('*') if p.is_file()}
            save(d/'manifest.json',manifest)
            save(out/f'evaluations/{identity}.json',dict(candidate_id=identity,control_tool=tool,protocol_sha256=driver.sha256(out/'protocol.json'),
                execution_manifest_sha256=driver.sha256(d/'manifest.json'),integrity_passed=True,physical_gate_passed=True))


def repin_fixture(out,identity):
    d=out/'trials'/identity;m=driver.read(d/'manifest.json')
    m['artifact_hashes']={str(p.relative_to(d)):driver.sha256(p) for p in d.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save(d/'manifest.json',m)
    e=driver.read(out/f'evaluations/{identity}.json');e['execution_manifest_sha256']=driver.sha256(d/'manifest.json');save(out/f'evaluations/{identity}.json',e)


def test_calibrations_require_full_bytes_inputs_and_common_parity(pilot,monkeypatch):
    make_calibrations(pilot)
    # Recover the real function despite the workflow fixture's speed stub.
    actual=REAL_VALIDATE_CALIBRATIONS
    monkeypatch.setattr(driver,'verify_code',lambda protocol:None)
    assert actual(pilot)['calibration_trials']==4
    np.save(pilot/'trials/calibration_local_wave/qvel.npy',np.zeros((3,2)))
    repin_fixture(pilot,'calibration_local_wave')
    with pytest.raises(ValueError,match='byte parity'):actual(pilot)


@pytest.mark.parametrize('name',['policy_inputs.npz','reference_cache.npz','common_reference/arrays.npz'])
def test_calibration_common_and_actual_inputs_cannot_differ(pilot,monkeypatch,name):
    make_calibrations(pilot);actual=REAL_VALIDATE_CALIBRATIONS
    monkeypatch.setattr(driver,'verify_code',lambda protocol:None)
    np.savez_compressed(pilot/'trials/calibration_local_wave'/name,q=np.zeros((3,2)),dq=np.zeros((3,2)),anchor=np.ones((3,4)))
    repin_fixture(pilot,'calibration_local_wave')
    with pytest.raises(ValueError,match='array parity'):actual(pilot)


def test_calibration_all_pins_must_match_even_with_same_physical_arrays(pilot,monkeypatch):
    make_calibrations(pilot);actual=REAL_VALIDATE_CALIBRATIONS
    monkeypatch.setattr(driver,'verify_code',lambda protocol:None)
    d=pilot/'trials/calibration_local_wave'
    m=driver.read(d/'manifest.json');m['preexecution_input_hashes']['extra_tool_code']='d'*64
    save(d/'preexecution_inputs.json',m['preexecution_input_hashes']);save(d/'manifest.json',m)
    repin_fixture(pilot,'calibration_local_wave')
    with pytest.raises(ValueError,match='full preexecution'):actual(pilot)


def test_numeric_uses_only_fresh_calibration_stores_tool_decision(pilot,monkeypatch):
    monkeypatch.setattr(driver,'_require_precommit',lambda *args:None)
    baseline_path=pilot/'evaluations/calibration_wave.json'
    malicious=driver.read(baseline_path)
    malicious.update(feature_errors=['wrong A goal']*5,per_dimension_pass=[False]*5,
        intent_passed=False,utility=999,utility_l2=99999,wall_seconds=123)
    save(baseline_path,malicious)
    def choose(history,protocol,goal,calibration):
        assert history==[]
        assert calibration['features']==[1,1,1,1,2.5]
        assert calibration['provenance']==driver.artifact(pilot,pilot/'evaluations/calibration_wave.json')
        return dict(control_tool=driver.TOOLS[1],parameters=[.8,1.2,1.2,.8,.1],decision={'strategy':'test','prediction_is_physical_evidence':False})
    monkeypatch.setattr(driver,'next_candidate',choose)
    monkeypatch.setattr(driver,'execute_trial',lambda *args:{'status':'success'})
    monkeypatch.setattr(driver,'record_evaluation',lambda *args:{'integrity_passed':True})
    driver.run(pilot,'wave_a','surrogate_search',0)
    p=driver.read(pilot/'proposals/wave_a__surrogate_search__0.json')
    o=driver.read(pilot/'observations/wave_a__surrogate_search__0.json')
    assert p['control_tool']==o['control_tool']==driver.TOOLS[1]
    assert p['search_decision']==o['search_decision']
    assert [t['control_tool'] for t in o['tools']]==list(driver.TOOLS)
    assert o['baseline']==driver.calibration_view(malicious)
    assert {'feature_errors','per_dimension_pass','intent_passed','utility','utility_l2','wall_seconds'}.isdisjoint(o['baseline'])
    assert p['response_artifact'] is None


def test_failed_runtime_consumes_slot_and_blocks_all_later_work(pilot,monkeypatch):
    monkeypatch.setattr(driver,'_require_precommit',lambda *args:None)
    driver.observation(pilot,'wave_a','llm_feedback',0);respond(pilot,'wave_a','llm_feedback',0)
    driver.commit_model(pilot,'wave_a','llm_feedback',0)
    monkeypatch.setattr(driver,'execute_trial',lambda *args:(_ for _ in ()).throw(RuntimeError('executor failed')))
    with pytest.raises(RuntimeError,match='executor failed'):driver.run(pilot,'wave_a','llm_feedback',0)
    assert driver.read(pilot/'abort.json')['candidate_id']=='wave_a__llm_feedback__0'
    assert (pilot/'proposals/wave_a__llm_feedback__0.json').is_file()
    with pytest.raises(RuntimeError,match='aborted'):driver.run(pilot,'high5_a','llm_feedback',0)


def test_missing_fresh_calibration_blocks_observation(pilot,monkeypatch):
    monkeypatch.setattr(driver,'validate_calibrations',REAL_VALIDATE_CALIBRATIONS)
    with pytest.raises(FileNotFoundError):driver.observation(pilot,'wave_a','llm_feedback_off',0)


def test_imported_labels_are_rejected_even_before_history(pilot):
    p=driver.read(pilot/'protocol.json');p['imported_prior_trial_records']=['old-run'];save(pilot/'protocol.json',p)
    with pytest.raises(ValueError,match='no imported'):driver.context(pilot,'wave_a')


def test_real_a_b_goals_share_only_goal_free_calibration(pilot):
    """Real task metadata, adversarial A-scored shared calibration, no labels."""
    original=driver.read(driver.ROOT/'artifacts/runs/agentic/generalization_music_v1/protocol.json')
    protocol=driver.read(pilot/'protocol.json')
    protocol['tasks']=original['tasks']
    save(pilot/'protocol.json',protocol)
    calibration_path=pilot/'evaluations/calibration_wave.json'
    measured=driver.read(calibration_path)
    measured.update(parameters=[1,1,1,1,0],feature_errors=['A_goal_errors']*5,
        per_dimension_pass=[False]*5,intent_passed=False,utility=999,utility_l2=99999,
        wall_seconds=123,accent={'available':True,'peak_time_s':2.5,'peak_rms_speed_rad_s':1.2,
        'event_times_s':[2.5],'event_count':1,'window_event_count':1,'reference':{'utility':'hidden'}})
    save(calibration_path,measured)
    a=driver.observation_data(pilot,'wave_a','llm_feedback_off',0)
    b=driver.observation_data(pilot,'wave_b','llm_feedback_off',0)
    assert a['schema']==b['schema']=='pulsemorph.agentic_tool_selection_observation.v2'
    assert a['canonical_goal']!=b['canonical_goal']
    assert a['canonical_goal']==next(t['goal'] for t in original['tasks'] if t['task_id']=='wave_a')
    assert b['canonical_goal']==next(t['goal'] for t in original['tasks'] if t['task_id']=='wave_b')
    assert a['baseline']==b['baseline']
    assert a['baseline']['features']==measured['features']
    assert set(a['baseline'])=={'candidate_id','parameters','control_tool','features','integrity_passed',
        'physical_gate_passed','mean_joint_rmse_rad','min_root_height_m','accent'}
    forbidden={'feature_errors','per_dimension_pass','utility','utility_l2','intent_passed','wall_seconds','reference'}
    assert forbidden.isdisjoint(a['baseline'])
    assert forbidden.isdisjoint(a['baseline']['accent'])
    assert a['prior_trial_results']==b['prior_trial_results']==[]


def test_calibration_projection_exact_fields_preserves_measurements():
    source=dict(candidate_id='fresh',parameters=[1,1,1,1,0],control_tool=driver.TOOLS[0],features=[1,1,1,1,2.5],
        integrity_passed=True,physical_gate_passed=True,mean_joint_rmse_rad=.2,min_root_height_m=.8,
        feature_errors=[-.2]*5,per_dimension_pass=[False]*5,intent_passed=False,utility=99,utility_l2=999,wall_seconds=10)
    view=driver.calibration_view(source)
    assert set(view)=={'candidate_id','parameters','control_tool','features','integrity_passed',
        'physical_gate_passed','mean_joint_rmse_rad','min_root_height_m'}
    assert view['features']==source['features']
    assert driver.compact(source)['feature_errors']==source['feature_errors']


def test_v2_context_rejects_v1_protocol(pilot):
    protocol=driver.read(pilot/'protocol.json');protocol['schema']='pulsemorph.agentic_tool_selection_protocol.v1'
    save(pilot/'protocol.json',protocol)
    with pytest.raises(ValueError,match='v2 required'):driver.context(pilot,'wave_a')
