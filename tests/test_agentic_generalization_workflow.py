"""Workflow isolation tests; no model, ONNX, or physics execution."""
import importlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
driver=importlib.import_module('generalization_driver')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')

@pytest.fixture
def pilot(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'verify_code',lambda protocol:None)
    source={'path':'trusted.motion','sha256':'a'*64}
    tasks=[dict(task_id=f'{s}_{v}',source_id=s,source_motion=source,instruction='controlled task',
                goal={'feature_targets':[.7,1.15,.7,1.15,2.35]},music_observation={}) for s in ['wave','high5'] for v in ['a','b']]
    protocol=dict(tasks=tasks,anchor=[1.,1.,1.,1.,0.],bounds=[[0,2]]*4+[[-.45,.45]],
                  feature_names=['lA','rA','lB','rB','accent'],feature_tolerances=[.1]*5,
                  physical_gate={},selection={},section_windows_s=[[.4,2],[2,3.6]],feature_windows_s=[[.8,1.6],[2.4,3.2]])
    save(tmp_path/'protocol.json',protocol)
    for source_id in ['wave','high5']:
        save(tmp_path/f'evaluations/calibration_{source_id}.json',{'candidate_id':'calibration_'+source_id,'features':[1]*5})
    return tmp_path

def respond(out,task,method,index,values):
    save(driver.response_path(out,task,method,index),dict(parameters=values,
         only_observation_path=str(driver.observation_path(out,task,method,index).resolve()),brief_reason='test'))

def test_off_observation_never_reads_candidate_outcomes(pilot,monkeypatch):
    driver.observation(pilot,'wave_a','llm_feedback_off',0)
    respond(pilot,'wave_a','llm_feedback_off',0,[1,1,1,1,0])
    driver.commit_model(pilot,'wave_a','llm_feedback_off',0)
    original=driver.read
    def guarded(path):
        if 'evaluations' in str(path) and 'calibration_' not in str(path):
            raise AssertionError('OFF accessed a candidate outcome')
        return original(path)
    monkeypatch.setattr(driver,'read',guarded)
    obs=driver.observation_data(pilot,'wave_a','llm_feedback_off',1)
    assert obs['prior_trial_results']==[]
    assert len(obs['prior_proposals'])==1
    assert obs['prior_proposals'][0]['parameters']==[1,1,1,1,0]

def test_cross_stream_response_rejected(pilot):
    driver.observation(pilot,'wave_a','llm_feedback',0)
    respond(pilot,'wave_a','llm_feedback',0,[1,1,1,1,0])
    path=driver.response_path(pilot,'wave_a','llm_feedback',0)
    raw=driver.read(path)
    raw['only_observation_path']=str(driver.observation_path(pilot,'high5_a','llm_feedback',0).resolve())
    save(path,raw)
    with pytest.raises(ValueError,match='observation mismatch'):
        driver.commit_model(pilot,'wave_a','llm_feedback',0)
    assert not (pilot/'proposals/wave_a__llm_feedback__0.json').exists()

def test_history_revision_and_budget_bounds(pilot):
    with pytest.raises(ValueError,match='budget'):
        driver.observation(pilot,'wave_a','llm_feedback_off',8)
    with pytest.raises(FileNotFoundError):
        driver.observation(pilot,'wave_a','llm_feedback_off',1)
    driver.observation(pilot,'wave_a','llm_feedback_off',0)
    respond(pilot,'wave_a','llm_feedback_off',0,[1,1,1,1,0])
    obs=driver.observation_path(pilot,'wave_a','llm_feedback_off',0)
    raw=driver.read(obs);raw['prior_trial_results']=[{'candidate_id':'sibling'}];save(obs,raw)
    with pytest.raises(ValueError,match='history'):
        driver.commit_model(pilot,'wave_a','llm_feedback_off',0)

def test_all_off_precommit_required_before_execution(pilot,monkeypatch):
    monkeypatch.setattr(driver,'execute_trial',lambda *a:pytest.fail('physics must not run'))
    with pytest.raises(FileNotFoundError):
        driver.run(pilot,'wave_a','surrogate_search',0)

def test_selector_gate_worst_l2_identifier_and_incomplete_summary(pilot):
    def row(name,worst,l2,gate=True):
        return dict(candidate_id=name,utility=worst,utility_l2=l2,integrity_passed=True,physical_gate_passed=gate)
    rows=[row('failed',0,0,False),row('z',1,2),row('b',1,1),row('a',1,1),row('lower_l2_but_worse',2,0)]
    assert driver.select(rows)['candidate_id']=='a'
    with pytest.raises(FileNotFoundError):driver.summarize(pilot)

def test_one_flat_vector_only_and_anchor(pilot):
    driver.observation(pilot,'wave_a','llm_feedback_off',0)
    respond(pilot,'wave_a','llm_feedback_off',0,[[1,1,1,1,0]])
    with pytest.raises(ValueError,match='five numeric'):
        driver.commit_model(pilot,'wave_a','llm_feedback_off',0)
    respond(pilot,'wave_a','llm_feedback_off',0,[.9,1,1,1,0])
    with pytest.raises(ValueError,match='anchor'):
        driver.commit_model(pilot,'wave_a','llm_feedback_off',0)
