"""Commit proposals, execute physical trials, record evidence, and compare budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

from workflow import ROOT, read, stamp, submit, observe, simple_amplitude, write_new_json, sha256
from execute_trial import execute_trial
from evaluate_trial import evaluate_trial


def metrics(out, candidate):
    return read(out/f'evaluations/{candidate}.json')


def proposal_checks(out, candidate):
    """Bind proposed, permitted and executed artifacts before using their scores."""
    original=out/f'proposals/{candidate}.json'; executed=out/f'trials/{candidate}/proposal.json'
    protocol=read(out/'protocol.json');proposal=read(executed)
    if sha256(original)!=sha256(executed) or proposal['candidate_id']!=candidate:
        raise ValueError('executed proposal differs from committed candidate')
    if proposal['protocol_sha256']!=sha256(out/'protocol.json'):
        raise ValueError('proposal protocol mismatch')
    for key in ['source_rate','source_start_s','duration_s','edit_window_s']:
        if proposal[key]!=protocol['tool'][key]:
            raise ValueError('proposal outside fixed tool context: '+key)
    if candidate!='calibration_0':
        if proposal['method'] not in protocol['methods'] or candidate!=f'{proposal["method"]}_{proposal["trial_index"]}':
            raise ValueError('candidate method/index mismatch')
        if not 0<=proposal['trial_index']<3 or not 0<=proposal['arm_amplitude']<=1:
            raise ValueError('proposal outside fixed budget/domain')
    else:
        if proposal['method']!='calibration' or proposal['arm_amplitude']!=1:
            raise ValueError('invalid calibration proposal')
    for field in ['observation_artifact','response_artifact']:
        item=proposal.get(field)
        if item and sha256(out/item['path'])!=item['sha256']:
            raise ValueError('proposal linked artifact changed: '+field)


def commit_off(out):
    response=out/'responses/llm_feedback_off_initial.json'
    data=read(response)
    for i, amplitude in enumerate(data['arm_amplitudes']):
        submit(out,'llm_feedback_off',i,amplitude,out/'observations/llm_feedback_off_initial.json',data['brief_reason'],response)


def evaluate_and_record(out, candidate):
    trial=out/f'trials/{candidate}'; baseline=out/'trials/calibration_0'
    proposal=read(trial/'proposal.json'); protocol=read(out/'protocol.json')
    result=evaluate_trial(trial,baseline,protocol)
    try:
        proposal_checks(out,candidate)
    except (ValueError,OSError,KeyError) as error:
        result.update(integrity_passed=False,physical_gate_passed=False,intent_passed=False)
        result['integrity_errors'].append(str(error))
    result.update(candidate_id=candidate, arm_amplitude=proposal['arm_amplitude'], method=proposal['method'])
    result['evaluator_sha256']=sha256(Path(__file__).with_name('evaluate_trial.py'))
    result['protocol_sha256']=sha256(out/'protocol.json')
    result['execution_manifest_sha256']=sha256(trial/'manifest.json')
    write_new_json(out/f'evaluations/{candidate}.json',result)
    record=dict(schema='pulsemorph.agentic_trial_record_draft.v1', common_trial_record_validated=False,
        candidate_id=candidate, recorded_at_utc=stamp(), task=protocol['task'], data_split='exploratory_development',
        observation=proposal.get('observation_artifact'), model_response=proposal.get('response_artifact'),
        proposal={'path':f'proposals/{candidate}.json','sha256':sha256(out/f'proposals/{candidate}.json')},
        requested_reference={'path':f'trials/{candidate}/reference_cache.npz','available':(trial/'reference_cache.npz').is_file(),
                             'sha256':sha256(trial/'reference_cache.npz') if (trial/'reference_cache.npz').is_file() else None},
        applied_action={'path':f'trials/{candidate}/applied_joint_targets.npy','meaning':'PD joint position targets; not reference or realized motion'},
        realized={'qpos':f'trials/{candidate}/qpos.npy','qvel':f'trials/{candidate}/qvel.npy','clock':f'trials/{candidate}/state_times.npy','telemetry':f'trials/{candidate}/substep_telemetry.npz'},
        execution_manifest={'path':f'trials/{candidate}/manifest.json','sha256':sha256(trial/'manifest.json')},
        independent_evaluation={'path':f'evaluations/{candidate}.json','sha256':sha256(out/f'evaluations/{candidate}.json')},
        audio_access=protocol['audio'], initialization=protocol['initialization'], execution_scope='complete new offline take',
        simulated_control_steps=read(trial/'manifest.json')['actual_control_steps'],
        model_tokens=None if proposal['method'].startswith('llm') else 0,
        provider_billed_cost=None if proposal['method'].startswith('llm') else 0.,
        hardware_energy_cost=None, human_preference=None, semantic_human_label=None,
        model_self_assessment_is_ground_truth=False,
        status='physically_screened' if result['physical_gate_passed'] else 'rejected_or_invalid')
    write_new_json(out/f'trial_records/{candidate}.json',record)
    print(json.dumps({k:result[k] for k in ['candidate_id','arm_amplitude','integrity_passed','integrity_errors','intent_ratio','intent_abs_error','physical_gate_passed','mean_joint_rmse_rad','min_root_height_m']},allow_nan=False),flush=True)
    return result


def run_one(out, method, index):
    candidate=f'{method}_{index}'
    if method=='simple_search':
        previous=[metrics(out,f'{method}_{i}') for i in range(index)]
        observation=out/f'observations/{candidate}.json'
        write_new_json(observation,dict(observation_id=candidate,method=method,prior_trial_results=previous,baseline_intent_ratio=1.))
        submit(out,method,index,simple_amplitude(previous),observation,'Bounded secant using only own earlier actual ratios and shared unchanged calibration.')
    elif method=='llm_feedback':
        suffix='initial' if index==0 else f'after_{index}'
        response=out/f'responses/{method}_{suffix}.json';data=read(response)
        submit(out,method,index,data['arm_amplitudes'][0],out/f'observations/{method}_{suffix}.json',data['brief_reason'],response)
    elif method!='llm_feedback_off':
        raise ValueError('unknown method')
    elif not all((out/f'proposals/{method}_{i}.json').is_file() for i in range(3)):
        raise ValueError('all three no-feedback proposals must be committed before execution')
    manifest=execute_trial(out/f'proposals/{candidate}.json',out/f'trials/{candidate}')
    result=evaluate_and_record(out,candidate)
    if method=='llm_feedback' and index<2:
        # Only same-method observed outcomes are sent back to the model.
        previous=[metrics(out,f'{method}_{i}') for i in range(index+1)]
        observe(out,method,previous)
    if manifest['status']!='success':
        raise RuntimeError('failed physical trial retained: '+candidate)
    return result


def summarize(out):
    protocol=read(out/'protocol.json')
    calibration=metrics(out,'calibration_0')
    audit=[]
    baseline_cache=dict(np.load(out/'trials/calibration_0/reference_cache.npz'))
    for candidate in ['calibration_0']+[f'{method}_{i}' for method in protocol['methods'] for i in range(3)]:
        proposal_checks(out,candidate)
        current=evaluate_trial(out/f'trials/{candidate}',out/'trials/calibration_0',protocol)
        previous=metrics(out,candidate)
        cache=dict(np.load(out/f'trials/{candidate}/reference_cache.npz'))
        lower_body_unchanged=all(np.array_equal(cache[key][...,:15],baseline_cache[key][...,:15])
            for key in ['committed__dof_pos','committed__dof_vel','forecast__dof_pos','forecast__dof_vel'])
        audit.append(dict(candidate_id=candidate,integrity_passed=current['integrity_passed'],errors=current['integrity_errors'],
            lower_body_reference_identical=lower_body_unchanged,
            metric_reproduction_exact=all(current[k]==previous[k] for k in ['intent_ratio','mean_joint_rmse_rad','min_root_height_m','physical_gate_passed'])))
    write_new_json(out/'final_independent_audit.json',dict(evaluator_sha256=sha256(Path(__file__).with_name('evaluate_trial.py')),checks=audit))
    methods=[]
    for method in protocol['methods']:
        rows=[metrics(out,f'{method}_{i}') for i in range(3)]
        eligible=[r for r in rows if r['physical_gate_passed']]
        chosen=min(eligible,key=lambda r:(r['intent_abs_error'],r['candidate_id'])) if eligible else None
        manifests=[read(out/f'trials/{method}_{i}/manifest.json') for i in range(3)]
        methods.append(dict(method=method, trials=rows, selected=chosen,
            physical_trials=len(manifests), control_steps=sum(m['actual_control_steps'] for m in manifests),
            physics_steps=sum(m['actual_physics_steps'] for m in manifests),
            model_proposal_calls={'simple_search':0,'llm_feedback':3,'llm_feedback_off':1}[method],
            generated_candidates=3, token_count=0 if method=='simple_search' else None,
            billed_model_cost=0 if method=='simple_search' else None,
            total_trial_wall_seconds=sum(r['latency'].get('wall_seconds',0) for r in rows)))
    before=read(out/'protected_hashes_before.json')
    protected={p:sha256(ROOT/p)==digest for p,digest in before.items()}
    result=dict(schema='pulsemorph.agentic_pilot_comparison.v1',protocol_sha256=sha256(out/'protocol.json'),
        created_at_utc=stamp(),calibration=calibration, methods=methods,
        comparison_valid=all(r['integrity_passed'] and r['metric_reproduction_exact'] and r['lower_body_reference_identical'] for r in audit),
        protected_files_unchanged=protected,
        total_comparison_physical_trials=1+sum(m['physical_trials'] for m in methods),
        total_control_steps=read(out/'trials/calibration_0/manifest.json')['actual_control_steps']+sum(m['control_steps'] for m in methods),
        total_physics_steps=read(out/'trials/calibration_0/manifest.json')['actual_physics_steps']+sum(m['physics_steps'] for m in methods),
        planned_budget=dict(comparison_trials=10,control_steps=2000,physics_steps=40000),
        paired_first_candidate_trajectories_identical=all(np.array_equal(np.load(out/f'trials/{method}_0/{name}.npy'),np.load(out/f'trials/simple_search_0/{name}.npy'))
            for method in ['llm_feedback','llm_feedback_off'] for name in ['qpos','qvel','applied_joint_targets']),
        extra_engineering_smoke=dict(path='artifacts/runs/agentic/runner_smoke_20261026_v1',control_steps=2,physics_steps=40,counted_in_method_budget=False),
        limitations=protocol['limitations']+['LLM token and billed cost receipts unavailable; unknown is not zero.','Agent deliberation and orchestration latency not isolated from human/host tool dispatch.','Model-call counts cover experiment proposals only; engineering/research model work is extra and unmetered.','Observation access restricted by explicit subagent instructions and supplied files, not an operating-system sandbox.','One task does not estimate general LLM benefit.'])
    write_new_json(out/'comparison.json',result)
    snapshot=out/'evaluation_source';snapshot.mkdir()
    for path in [Path(__file__),Path(__file__).with_name('workflow.py'),Path(__file__).with_name('evaluate_trial.py'),ROOT/'experiments/music_control/accents.py',ROOT/'experiments/music_control/evaluate.py']:
        target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    write_new_json(out/'evaluation_source_hashes.json',{str(p.relative_to(snapshot)):sha256(p) for p in snapshot.rglob('*.py')})
    print(json.dumps({'comparison_valid':result['comparison_valid'],'protected_files_unchanged':all(protected.values()),'selected':[{ 'method':m['method'],'candidate_id':m['selected']['candidate_id'] if m['selected'] else None,'intent_ratio':m['selected']['intent_ratio'] if m['selected'] else None,'intent_passed':m['selected']['intent_passed'] if m['selected'] else False} for m in methods]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--action',choices=['commit-off','evaluate-calibration','run','summarize'],required=True)
    p.add_argument('--method');p.add_argument('--index',type=int)
    a=p.parse_args();out=a.pilot.resolve()
    if a.action=='commit-off':commit_off(out)
    elif a.action=='evaluate-calibration':evaluate_and_record(out,'calibration_0')
    elif a.action=='run':run_one(out,a.method,a.index)
    else:summarize(out)
