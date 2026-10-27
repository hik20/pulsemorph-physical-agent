"""Audit contract tests without simulations or model calls."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/agentic'))
import audit_tool_selection_v2 as audit
import execute_tool_selection as runtime
from pulsemorph.agentic.common_player import materialize_reference
from pulsemorph.motion.contracts import load_reference, save_reference
from test_agentic_reference import fixture


def metadata_bundle(tmp_path,tool,method):
    """Use frozen runtime producer AND actual common save/load, no invented IDs."""
    p=dict(control_tool=tool,method=method,parameters=dict(left_gain_a=1.,right_gain_a=1.,left_gain_b=1.,right_gain_b=1.,timing_shift_s=0.))
    raw,old=fixture()
    generator=runtime.build_generator(p,raw,joint_names=old.joint_names,body_names=old.body_names,
        source_start_s=1.,control_steps=200,source_provenance={'sha256':'fixture'})
    producer=dict(runtime.proposal_producer(p),control_tool=tool)
    reference=materialize_reference(generator,profile={'body_sha256':'a'*64,'yaml_sha256':'b'*64},
        candidate_id='fixture',observation_id='fixture-observation',task_id='fixture-task',runtime_session_id='fixture-runtime',
        source_path='fixture.motion',source_sha256='c'*64,audio_observed_through_sample=88200,audio_duration_s=4.,producer_kind=producer['kind'])
    reference=replace(reference,metadata={**reference.metadata,'producer':producer,'control_tool':tool,'experiment_kind':'tool_selection_v1'})
    save_reference(reference,tmp_path/'common')
    loaded=load_reference(tmp_path/'common')
    manifest=dict(schema='pulsemorph.agentic_generalization_trial.v1',producer=producer,control_tool=tool,experiment_kind='tool_selection_v1')
    draft=dict(control_tool=tool,producer=producer)
    _,adapter=generator.draft_motion_reference(0);adapter['control_tool']=tool
    return p,manifest,loaded.metadata,draft,adapter


@pytest.mark.parametrize('tool',audit.TOOLS)
@pytest.mark.parametrize('method',['calibration','surrogate_search','llm_feedback','llm_feedback_off'])
def test_real_frozen_producer_roundtrip(tmp_path,tool,method):
    bundle=metadata_bundle(tmp_path,tool,method)
    audit.check_control_provenance(*bundle)
    assert bundle[1]['producer']['id']=='tool-selection-'+method
    assert bundle[1]['producer']['version']=='1'


@pytest.mark.parametrize('position',[1,2,3,4])
def test_wrong_tool_in_any_metadata_rejected(tmp_path,position):
    bundle=list(metadata_bundle(tmp_path,audit.TOOLS[0],'llm_feedback'))
    bundle[position]=dict(bundle[position],control_tool=audit.TOOLS[1])
    with pytest.raises(ValueError,match='control_tool'):
        audit.check_control_provenance(*bundle)


def test_old_producer_id_rejected_even_when_kind_matches(tmp_path):
    p,m,ref,draft,adapter=metadata_bundle(tmp_path,audit.TOOLS[1],'calibration')
    m['producer']=dict(m['producer'],id='local-capability-calibration',version='2')
    with pytest.raises(ValueError,match='exact runtime producer'):
        audit.check_control_provenance(p,m,ref,draft,adapter)


def scope_bundle():
    old=audit.read(ROOT/'artifacts/runs/agentic/generalization_music_v1/protocol.json')
    p=copy.deepcopy(old)
    p.update(schema='pulsemorph.agentic_tool_selection_protocol.v2',phase='development_comparison',training_admission=False,
        imported_prior_trial_records=[],optimizer={'old_feature_labels_imported':False},control_tools=list(audit.TOOLS),
        trial_budget_per_method_per_task=8,new_candidate_budget=8,
        stopping_policy='All8 candidate slots; no success early stop. No confirmation. no free repair, retry or silent replacement',
        planned_budget=dict(candidate_trials=96,calibration_trials=4,physical_trials=100,control_steps=20000,physics_steps=400000,model_proposal_responses=64))
    comparison=dict(schema='pulsemorph.agentic_tool_selection_comparison.v1',development_only=True)
    return p,comparison,old


def test_unchanged_development_scope():
    audit.check_development_scope(*scope_bundle())


@pytest.mark.parametrize('change',['phase','goals','waveform','oldlabels','stopping','budget'])
def test_scope_leakage_or_early_stopping_rejected(change):
    p,c,old=scope_bundle()
    if change=='phase':p['phase']='held_out_main'
    if change=='goals':p['tasks'][0]['goal']['feature_targets'][0]+=.01
    if change=='waveform':p['tasks'][0]['music_observation']['waveform_sha256']='0'*64
    if change=='oldlabels':p['imported_prior_trial_records']=['old_label']
    if change=='stopping':p['stopping_policy']='stop at success'
    if change=='budget':p['new_candidate_budget']=9
    with pytest.raises(ValueError):audit.check_development_scope(p,c,old)


def rows_and_summary():
    rows=[dict(candidate_id=f'candidate_{i}',control_tool=audit.TOOLS[i%2],parameters=[1,1,1,1,0],
        integrity_passed=True,physical_gate_passed=i!=0,utility=float(8-i),utility_l2=float(8-i)**2) for i in range(8)]
    summary=dict(selected_tool=audit.select(rows)['control_tool'],tool_usage={tool:4 for tool in audit.TOOLS},physical_trials=8,
        prefix_best=[dict(budget=i,selected=audit.select(rows[:i])) for i in range(1,9)])
    return rows,summary


def test_summary_prefix_and_tool_accounting():
    rows,summary=rows_and_summary();audit.check_summary_tools(summary,rows)


@pytest.mark.parametrize('change',['tool','usage','early_stop','prefix'])
def test_summary_tampering_rejected(change):
    rows,summary=rows_and_summary()
    if change=='tool':summary['selected_tool']=audit.TOOLS[0]
    if change=='usage':summary['tool_usage'][audit.TOOLS[0]]=3
    if change=='early_stop':summary['physical_trials']=7
    if change=='prefix':summary['prefix_best'][0]['selected']=rows[0]
    with pytest.raises(ValueError):audit.check_summary_tools(summary,rows)


def test_audit_missing_completion_does_not_make_output(tmp_path):
    pilot=tmp_path/'pilot';pilot.mkdir()
    output=tmp_path/'audit'
    with pytest.raises(FileNotFoundError):audit.audit(pilot,output)
    assert not output.exists()

def observation_fixture(tmp_path,monkeypatch,method='llm_feedback_off'):
    import tool_selection_driver_v2 as driver
    p,_,_=scope_bundle();task=p['tasks'][0]
    pilot=tmp_path/'pilot';pilot.mkdir()
    for name in ('observations','responses','proposals','evaluations'):(pilot/name).mkdir()
    def save(path,value):path.write_text(json.dumps(value),encoding='utf8')
    save(pilot/'protocol.json',p)
    baseline=dict(candidate_id='calibration_'+task['source_id'],parameters=p['anchor'],features=[1.,1.,1.,1.,2.],
        integrity_passed=True,physical_gate_passed=True,control_tool=audit.TOOLS[0],utility=None,utility_l2=None,intent_passed=None)
    save(pilot/'evaluations'/(baseline['candidate_id']+'.json'),baseline)
    monkeypatch.setattr(driver,'validate_calibrations',lambda out:None)
    identity=audit.cid(task['task_id'],method,0)
    obs_path=pilot/'observations'/(identity+'.json')
    save(obs_path,driver.observation_data(pilot,task['task_id'],method,0))
    raw=dict(control_tool=audit.TOOLS[1],parameters=[1.,1.,1.,1.,.1],only_observation_path=str(obs_path),brief_reason='unit fixture')
    response=pilot/'responses'/(identity+'.json');save(response,raw)
    proposal=dict(candidate_id=identity,task_id=task['task_id'],method=method,trial_index=0,observation_id=identity,
        observation_artifact=driver.artifact(pilot,obs_path),response_artifact=driver.artifact(pilot,response),
        control_tool=raw['control_tool'],parameters=dict(zip(p['parameter_names'],raw['parameters'])),search_decision=None,
        brief_reason=raw['brief_reason'])
    save(pilot/'proposals'/(identity+'.json'),proposal)
    return pilot,proposal,p,task,{baseline['candidate_id']:baseline}


@pytest.mark.parametrize('method',['llm_feedback','llm_feedback_off'])
def test_actual_driver_observation_matches_independent_audit(tmp_path,monkeypatch,method):
    bundle=observation_fixture(tmp_path,monkeypatch,method)
    audit.check_observation(*bundle)


def test_feedback_off_injected_outcome_rejected(tmp_path,monkeypatch):
    pilot,proposal,p,task,results=observation_fixture(tmp_path,monkeypatch)
    path=pilot/proposal['observation_artifact']['path']
    obs=audit.read(path);obs['prior_trial_results']=[{'candidate_id':'sibling','utility':0}]
    path.write_text(json.dumps(obs),encoding='utf8')
    proposal['observation_artifact']['sha256']=audit.sha(path)
    with pytest.raises(ValueError,match='exact isolated model observation'):
        audit.check_observation(pilot,proposal,p,task,results)

def test_a_b_goal_relative_fields_cannot_change_calibration_view():
    import tool_selection_driver_v2 as driver
    baseline=dict(candidate_id='calibration_wave',parameters=[1,1,1,1,0],control_tool='global_clock_v1',
        features=[1,1,1,1,2.1],integrity_passed=True,physical_gate_passed=True,
        mean_joint_rmse_rad=.1,min_root_height_m=.8,
        accent=dict(available=True,peak_time_s=2.1,peak_rms_speed_rad_s=3.,event_times_s=[2.1],event_count=1,window_event_count=1))
    a=dict(baseline,feature_errors=[.2,-.2,.2,-.2,-.3],per_dimension_pass=[False]*5,
           intent_passed=False,utility=3.,utility_l2=20.,wall_seconds=12.)
    b=dict(baseline,feature_errors=[-.2,.2,-.2,.2,.1],per_dimension_pass=[False,False,False,False,True],
           intent_passed=False,utility=2.,utility_l2=8.,wall_seconds=15.)
    expected=audit.calibration_measurements(a)
    assert expected==audit.calibration_measurements(b)==driver.calibration_view(a)==driver.calibration_view(b)
    assert set(expected)=={'candidate_id','parameters','control_tool','features','integrity_passed',
        'physical_gate_passed','mean_joint_rmse_rad','min_root_height_m','accent'}
    assert set(expected['accent'])=={'available','peak_time_s','peak_rms_speed_rad_s','event_times_s','event_count','window_event_count'}
    audit.check_calibration_view(expected,a)


@pytest.mark.parametrize('field',['feature_errors','per_dimension_pass','utility','utility_l2','intent_passed','wall_seconds'])
def test_contradictory_goal_or_wall_field_in_baseline_rejected(field):
    baseline=dict(candidate_id='calibration_high5',features=[1,1,1,1,2.1])
    observed=audit.calibration_measurements(baseline)
    observed[field]=[.1]*5 if field=='feature_errors' else False
    with pytest.raises(ValueError,match='target-independent'):
        audit.check_calibration_view(observed,baseline)


def test_nested_accent_cannot_smuggle_target_error():
    baseline=dict(accent=dict(available=True,peak_time_s=2.1))
    observed=audit.calibration_measurements(baseline)
    observed['accent']['timing_error_s']=.2
    with pytest.raises(ValueError,match='target-independent'):
        audit.check_calibration_view(observed,baseline)

def quarantine_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    folder=tmp_path/'artifacts/runs/agentic/tool_selection_development_v1';folder.mkdir(parents=True)
    path=folder/'abort.json'
    path.write_text(json.dumps(dict(schema='pulsemorph.agentic_tool_selection_abort.v1',stage='shared_calibration_goal_context_error',
        retry_allowed=False,slot_consumed=True,created_at_utc='2026-10-26T19:18:46+00:00')))
    return path,dict(supersedes_quarantined=dict(path=str(path),sha256=audit.sha(path)),imported_prior_trial_records=[])


def test_quarantined_predecessor_exact_hash_stage(tmp_path,monkeypatch):
    path,p=quarantine_fixture(tmp_path,monkeypatch)
    result=audit.check_quarantined_predecessor(p)
    assert result['stage']=='shared_calibration_goal_context_error' and result['old_trial_labels_imported'] is False


@pytest.mark.parametrize('change',['hash','stage','oldlabels'])
def test_quarantined_predecessor_rejects_changed_evidence_or_reuse(tmp_path,monkeypatch,change):
    path,p=quarantine_fixture(tmp_path,monkeypatch)
    if change=='hash':p['supersedes_quarantined']['sha256']='0'*64
    if change=='stage':
        value=audit.read(path);value['stage']='other';path.write_text(json.dumps(value));p['supersedes_quarantined']['sha256']=audit.sha(path)
    if change=='oldlabels':p['imported_prior_trial_records']=['oldlabel']
    with pytest.raises(ValueError):audit.check_quarantined_predecessor(p)
