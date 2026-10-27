"""Protocol/commit isolation checks; no model calls or physics execution."""
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/agentic'))
import musical_tasks as music
import multifeature_driver as driver


def put(out,name,value):
    path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8');return path


@pytest.fixture
def out(tmp_path):
    put(tmp_path,'protocol.json',dict(tasks=[{'task_id':'a','goal':{'feature_targets':[.7,1.15,1.15,.7,2.53]}},{'task_id':'b','goal':{'feature_targets':[1.15,.7,.7,1.15,3.03]}}],bounds=[[.35,1.6]]*4+[[-.3,.3]],
        anchor=[1.,1.,1.,1.,0.],section_windows_s=[[.4,2.],[2.,3.6]],trial_budget_per_method_per_task=8))
    (tmp_path/'trials/calibration_0').mkdir(parents=True)
    return tmp_path


def response(out,task='a',method='llm_feedback',index=0,vectors=None):
    obs=put(out,f'observations/{task}__{method}__{index}.json',dict(observation_id=driver.identifier(task,method,index),
        task_id=task,method=method,trial_index=index,protocol_sha256=music.sha256(out/'protocol.json'),
        canonical_goal=driver.case(music.read(out/'protocol.json'),task)['goal'],
        prior_trial_results=[{'candidate_id':driver.identifier(task,method,i)} for i in range(index)]))
    vectors=vectors or [[1.,1.,1.,1.,0.]]*(8 if method.endswith('_off') else 1)
    raw=put(out,f'responses/{task}__{method}__{index}.json',dict(only_observation_path=str(obs),parameters=vectors,brief_reason='Test fixture.'))
    return obs,raw


def test_waveform_measurement_changes_with_pcm_not_authored_metadata(tmp_path):
    times=[]
    for cue in [2.5,3.0]:
        pcm,_=music.synthesize(cue,71)
        path=tmp_path/f'{cue}.wav'
        with wave.open(str(path),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(music.RATE);f.writeframes(pcm.tobytes())
        observed=music.analyze_waveform(path)
        event=next(e for e in observed['measured_bass_events'] if e['event_id']==observed['strongest_bass_event_id'])
        times.append(event['time_s'])
        assert abs(event['time_s']-cue)<.08
        assert observed['waveform_sha256']==music.sha256(path)
        assert 'authored_primary_cue_s' not in observed
    assert .4<times[1]-times[0]<.6


@pytest.mark.parametrize('values',[[True,1,1,1,0],[float('nan'),1,1,1,0],[.34,1,1,1,0],[1,1,1,1,.31],[1,1]])
def test_invalid_bounds_rejected_without_commit(out,values):
    with pytest.raises(ValueError):driver.commit(out,'a','jacobian_search',0,values)
    assert not (out/'proposals').exists()


@pytest.mark.parametrize('index',[-1,8,100,True,1.5])
def test_budget_boundary(out,index):
    with pytest.raises(ValueError):driver.commit(out,'a','jacobian_search',index,[1.,1.,1.,1.,0.])


def test_exclusive_write_and_parameter_response_binding(out):
    obs,raw=response(out)
    driver.commit(out,'a','llm_feedback',0,[1.,1.,1.,1.,0.],obs,raw)
    saved=(out/'proposals/a__llm_feedback__0.json').read_bytes()
    with pytest.raises((FileExistsError,ValueError)):driver.commit(out,'a','llm_feedback',0,[1.,1.,1.,1.,0.],obs,raw)
    assert (out/'proposals/a__llm_feedback__0.json').read_bytes()==saved
    obs,raw=response(out,index=1,vectors=[[.8,1.,1.,1.,0.]])
    with pytest.raises(ValueError):driver.commit(out,'a','llm_feedback',1,[.9,1.,1.,1.,0.],obs,raw)


def test_wrong_observation_path_task_and_revision(out):
    obs,raw=response(out)
    data=music.read(raw);data['only_observation_path']=str(out/'wrong.json');put(out,str(raw.relative_to(out)),data)
    with pytest.raises(ValueError):driver.commit(out,'a','llm_feedback',0,[1.,1.,1.,1.,0.],obs,raw)
    obs,raw=response(out,task='b')
    with pytest.raises(ValueError):driver.commit(out,'a','llm_feedback',0,[1.,1.,1.,1.,0.],obs,raw)
    obs,raw=response(out)
    driver.commit(out,'a','llm_feedback',0,[1.,1.,1.,1.,0.],obs,raw)
    with pytest.raises(ValueError):driver.commit(out,'a','llm_feedback',1,[1.,1.,1.,1.,0.],obs,raw)


def test_off_all_eight_and_both_tasks_required_before_execution(out,monkeypatch):
    called=[]
    monkeypatch.setattr(driver,'execute_trial',lambda *a:called.append(a))
    for task in ['a','b']:
        response(out,task=task,method='llm_feedback_off')
    # Run boundary must reject even when one task's complete stream exists.
    for i in range(8):
        obs,raw=out/'observations/a__llm_feedback_off__0.json',out/'responses/a__llm_feedback_off__0.json'
        driver.commit(out,'a','llm_feedback_off',i,[1.,1.,1.,1.,0.],obs,raw)
    with pytest.raises(ValueError):driver.run(out,'a','llm_feedback_off',0)
    assert called==[]
    # Any target trial directory disallows late no-feedback commitments.
    (out/'trials/any_candidate').mkdir()
    obs,raw=out/'observations/b__llm_feedback_off__0.json',out/'responses/b__llm_feedback_off__0.json'
    with pytest.raises(ValueError):driver.commit(out,'b','llm_feedback_off',0,[1.,1.,1.,1.,0.],obs,raw)


def test_semantic_scoring_uses_canonical_targets_not_gain_parameters(out,monkeypatch):
    # A reversed semantic result must fail, even though all tool anchors equal one.
    protocol=music.read(out/'protocol.json')
    goal=dict(feature_targets=[.7,1.15,1.15,.7,2.53],cue_event_id='bass_peak_05')
    protocol['tasks']=[dict(task_id='a',goal=goal)]
    put(out,'protocol.json',protocol)
    put(out,'responses/semantic_interpretation.json',dict(results=[dict(task_id='a',feature_targets=[1.15,.7,.7,1.15,2.53],cue_event_id='bass_peak_05')]))
    put(out,'observations/semantic_tasks.json',dict(glossary={'dar':.7,'geniş':1.15},instruction='İlk bölüm sol dar, sağ geniş; sonra değiştir.'))
    monkeypatch.setattr(driver,'record_evaluation',lambda *a:None)
    monkeypatch.setattr(driver,'observation',lambda *a:None)
    driver.prepare(out)
    row=music.read(out/'semantic_evaluation.json')['checks'][0]
    assert not row['feature_targets_match']
    assert row['cue_event_matches']


@pytest.mark.parametrize('corruption',['protocol','history_length','sibling_history','observation_id','goal'])
def test_stale_protocol_and_cross_stream_history_rejected(out,corruption):
    obs,raw=response(out)
    driver.commit(out,'a','llm_feedback',0,[1.,1.,1.,1.,0.],obs,raw)
    obs,raw=response(out,index=1)
    data=music.read(obs)
    if corruption=='protocol':data['protocol_sha256']='0'*64
    elif corruption=='history_length':data['prior_trial_results']=[]
    elif corruption=='sibling_history':data['prior_trial_results']=[{'candidate_id':'b__llm_feedback__0'}]
    elif corruption=='observation_id':data['observation_id']='a__llm_feedback__0'
    else:data['canonical_goal']={'feature_targets':[1,1,1,1,0]}
    put(out,str(obs.relative_to(out)),data)
    with pytest.raises(ValueError):driver.commit(out,'a','llm_feedback',1,[1.,1.,1.,1.,0.],obs,raw)
    assert not (out/'proposals/a__llm_feedback__1.json').exists()


def test_jacobian_repair_ignores_unsafe_perfect_objective_center():
    from evaluate_multifeature import next_jacobian_candidate
    anchor=np.array([1.,1.,1.,1.,0.])
    protocol=dict(parameter_names=music.NAMES,bounds=[[.35,1.6]]*4+[[-.3,.3]],
        feature_tolerances=[.08]*4+[.1],jacobian_trust_radius=[.01]*5)
    # Linear fixture: probe1 hits the target exactly, but fails physical screening.
    goal={'feature_targets':(anchor+np.array([.15,0,0,0,0])).tolist()}
    history=[]
    for i in range(6):
        x=anchor.copy()
        if i:x[i-1]+=.1 if i==5 else .15
        history.append(dict(parameters=x.tolist(),features=x.tolist(),integrity_passed=True,physical_gate_passed=i!=1))
    repaired=np.array(next_jacobian_candidate(history,protocol,goal))
    assert 1.<repaired[0]<=1.010001
    assert np.max(np.abs(repaired-anchor))<=.010001
    for row in history:row['physical_gate_passed']=False
    with pytest.raises(ValueError,match='no physically valid'):next_jacobian_candidate(history,protocol,goal)


def test_flat_single_vector_normalization_preserves_values_raw_bytes_and_hash(out):
    obs,raw=response(out)
    data=music.read(raw);data['parameters']=[1.,1.,1.,1.,0.]
    put(out,str(raw.relative_to(out)),data)
    before=raw.read_bytes();digest=music.sha256(raw)
    parsed=driver.model_response(raw)
    assert parsed['parameters']==[[1.,1.,1.,1.,0.]]
    assert parsed['parameter_layout_normalization']=='wrapped_single_flat_vector'
    proposal=driver.commit(out,'a','llm_feedback',0,parsed['parameters'][0],obs,raw)
    assert [proposal['parameters'][name] for name in music.NAMES]==data['parameters']
    assert proposal['response_parameter_layout_normalization']=='wrapped_single_flat_vector'
    assert proposal['response_artifact']['sha256']==digest
    assert raw.read_bytes()==before and music.sha256(raw)==digest


@pytest.mark.parametrize('parameters',[[True,1.,1.,1.,0.],[1.,1.,1.,1.],[1.,1.,1.,1.,0.,0.],
    [[1.,1.,1.,1.,0.],[1.,1.,1.,1.,0.]],['1',1.,1.,1.,0.],[[1.,1.,1.,1.,0.],None]])
def test_normalization_cannot_turn_malformed_or_extra_candidates_into_one(out,parameters):
    obs,raw=response(out)
    data=music.read(raw);data['parameters']=parameters
    put(out,str(raw.relative_to(out)),data)
    before=raw.read_bytes()
    parsed=driver.model_response(raw)
    assert parsed['parameters']==parameters
    assert parsed['parameter_layout_normalization']=='none'
    with pytest.raises((ValueError,TypeError,IndexError)):
        driver.commit(out,'a','llm_feedback',0,[1.,1.,1.,1.,0.],obs,raw)
    assert raw.read_bytes()==before
    assert not (out/'proposals/a__llm_feedback__0.json').exists()


def test_eight_feedback_off_vectors_are_unchanged_by_parser(out):
    vectors=[[1.,1.,1.,1.,0.]]+[[.5+i*.1,1.,1.,1.,0.] for i in range(7)]
    obs,raw=response(out,method='llm_feedback_off',vectors=vectors)
    before=raw.read_bytes();digest=music.sha256(raw)
    parsed=driver.model_response(raw)
    assert parsed['parameters']==vectors
    assert parsed['parameter_layout_normalization']=='none'
    for index,values in enumerate(vectors):
        proposal=driver.commit(out,'a','llm_feedback_off',index,values,obs,raw)
        assert proposal['response_parameter_layout_normalization']=='none'
        assert proposal['response_artifact']['sha256']==digest
    assert raw.read_bytes()==before
