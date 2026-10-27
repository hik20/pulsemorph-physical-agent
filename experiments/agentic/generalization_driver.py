"""Immutable source-held-out workflow.

CLI: python experiments/agentic/generalization_driver.py --pilot PATH --action
evaluate-calibration --task wave_a; observe/commit-model/run additionally take
--task TASK --method METHOD --index 0..7. Model responses are saved separately
at responses/TASK__METHOD__INDEX.json before commit-model. Commit all 32 OFF
responses before run. summarize requires all 96 candidate evaluations and both
calibrations. No model provider is called by this driver.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from musical_tasks import ROOT, NAMES, read, sha256, write_new_json
from multifeature_driver import compact, validate_parameters, stamp
from execute_generalization import execute_trial
from evaluate_generalization import evaluate_generalization
from generalization_search import next_candidate

METHODS = ('surrogate_search', 'llm_feedback', 'llm_feedback_off')
BUDGET = 8

def identifier(task, method, index): return f'{task}__{method}__{index}'
def observation_path(out, task, method, index): return out/'observations'/f'{identifier(task,method,index)}.json'
def response_path(out, task, method, index): return out/'responses'/f'{identifier(task,method,index)}.json'
def case(protocol, task):
    matches = [t for t in protocol['tasks'] if t['task_id'] == task]
    if len(matches) != 1: raise ValueError('unknown or duplicate task')
    return matches[0]

def context(out, task_id, method=None, index=None):
    protocol = read(out/'protocol.json'); task = case(protocol, task_id)
    if method is not None and (method not in METHODS or type(index) is not int or not 0 <= index < BUDGET):
        raise ValueError('method/index outside budget')
    return protocol, task

def verify_code(protocol):
    pins = protocol.get('execution_code_sha256')
    if not isinstance(pins, dict) or not pins: raise ValueError('execution code pins required')
    for path, digest in pins.items():
        if sha256(ROOT/path) != digest: raise ValueError('execution code changed: '+path)

def artifact(out, path):
    path = Path(path).resolve()
    return dict(path=path.relative_to(out.resolve()).as_posix(), sha256=sha256(path))

def histories(out, task, method, index, outcomes=True):
    protocol=read(out/'protocol.json')
    proposals=[]; results=[]
    for i in range(index):
        identity=identifier(task,method,i); proposal=read(out/f'proposals/{identity}.json')
        if (proposal['candidate_id'],proposal['task_id'],proposal['method'],proposal['trial_index']) != (identity,task,method,i):
            raise ValueError('cross-stream proposal history')
        if proposal['protocol_sha256']!=sha256(out/'protocol.json'): raise ValueError('stale proposal history')
        values=validate_parameters([proposal['parameters'][n] for n in NAMES],protocol)
        proposals.append(dict(candidate_id=identity,parameters=[proposal['parameters'][n] for n in NAMES],
                              proposal_sha256=sha256(out/f'proposals/{identity}.json')))
        if outcomes:
            result=read(out/f'evaluations/{identity}.json')
            if result['candidate_id'] != identity: raise ValueError('cross-stream outcome history')
            results.append(compact(result))
    return proposals, results

def observation_data(out, task_id, method, index):
    protocol,task=context(out,task_id,method,index)
    if method == 'surrogate_search': raise ValueError('observe is for recorded model calls')
    proposals,results=histories(out,task_id,method,index,method=='llm_feedback')
    baseline=read(out/f'evaluations/calibration_{task["source_id"]}.json')
    return dict(schema='pulsemorph.agentic_generalization_observation.v1',
        observation_id=identifier(task_id,method,index),task_id=task_id,method=method,trial_index=index,
        protocol_sha256=sha256(out/'protocol.json'),source_id=task['source_id'],source_motion=task['source_motion'],
        user_instruction=task['instruction'],canonical_goal=task['goal'],baseline=compact(baseline),
        parameter_names=NAMES,bounds=protocol['bounds'],feature_names=protocol['feature_names'],
        feature_tolerances=protocol['feature_tolerances'],physical_gate=protocol['physical_gate'],selection=protocol['selection'],
        music_observation=task['music_observation'],prior_proposals=proposals,prior_trial_results=results,
        trials_remaining=BUDGET-index,forced_first_parameters=protocol['anchor'] if index==0 else None,
        tool=dict(section_windows_s=protocol['section_windows_s'],measurement_windows_s=protocol['feature_windows_s'],
            gains='Independent per-side initial-posture displacement gains with sin-squared section envelopes. Achieved RMS ratios differ from gains.',
            clock='source_time=t+timing_shift_s*sin(pi*t/4)^2 inside 0<t<4; identity outside; analytic chain-rule velocity.',
            output='Validated MotionReference -> fixed pretrained G1 ONNX -> PD -> MuJoCo. One offline take; no learning.'),
        access='Use only this observation and your own previous supplied observations. No sibling files or code. Feedback-off has no candidate outcomes.',
        response_schema={'parameters':'one flat list of five numbers','only_observation_path':str(observation_path(out,task_id,method,index).resolve()),'brief_reason':'short decision summary'},
        instructions='Return exactly one next vector. First vector must be forced anchor. No extra candidates.')

def observation(out,task,method,index):
    path=observation_path(out,task,method,index)
    write_new_json(path,observation_data(out,task,method,index)); return path

def any_candidates(out):
    return any(p.is_dir() and not p.name.startswith('calibration_') for p in (out/'trials').glob('*'))

def commit(out,task_id,method,index,values,obs=None,response=None,reason=''):
    protocol,task=context(out,task_id,method,index);verify_code(protocol)
    values=validate_parameters(values,protocol)
    if index==0 and values != protocol['anchor']: raise ValueError('first candidate must equal common anchor')
    if method=='llm_feedback_off' and any_candidates(out): raise ValueError('OFF proposals must all precede candidates')
    histories(out,task_id,method,index,method!='llm_feedback_off')
    if method.startswith('llm'):
        expected=observation_path(out,task_id,method,index).resolve()
        if obs is None or response is None or Path(obs).resolve()!=expected or Path(response).resolve()!=response_path(out,task_id,method,index).resolve():
            raise ValueError('wrong response/observation path')
        observed=read(obs);raw=read(response)
        if observed != observation_data(out,task_id,method,index): raise ValueError('stale or cross-stream observation/history')
        if Path(raw['only_observation_path']).resolve()!=expected: raise ValueError('response observation mismatch')
        if validate_parameters(raw['parameters'],protocol)!=values: raise ValueError('response vector mismatch')
        reason=raw.get('brief_reason','')
    candidate=identifier(task_id,method,index)
    proposal=dict(candidate_id=candidate,task_id=task_id,method=method,trial_index=index,
        source_id=task['source_id'],source_motion=task['source_motion'],parameters=dict(zip(NAMES,values)),
        duration_s=4.,source_start_s=0.,section_windows_s=protocol['section_windows_s'],
        audio={'duration_s':4.,'observed_through_sample':88200},protocol_sha256=sha256(out/'protocol.json'),
        created_at_utc=stamp(),observation_id=candidate,
        observation_artifact=artifact(out,obs) if obs else None,response_artifact=artifact(out,response) if response else None,brief_reason=reason)
    write_new_json(out/f'proposals/{candidate}.json',proposal)
    return proposal

def commit_model(out,task,method,index):
    if method not in ('llm_feedback','llm_feedback_off'): raise ValueError('model method required')
    response=response_path(out,task,method,index); raw=read(response)
    result=commit(out,task,method,index,raw['parameters'],observation_path(out,task,method,index),response)
    if method=='llm_feedback_off' and index<BUDGET-1: observation(out,task,method,index+1)
    return result

def require_precommit(out,protocol):
    for task in protocol['tasks']:
        for i in range(BUDGET):
            identity=identifier(task['task_id'],'llm_feedback_off',i)
            proposal=read(out/f'proposals/{identity}.json')
            if proposal['protocol_sha256']!=sha256(out/'protocol.json'): raise ValueError('stale OFF commitment')
            if (proposal['candidate_id'],proposal['task_id'],proposal['method'],proposal['trial_index'],proposal['source_motion']) != (identity,task['task_id'],'llm_feedback_off',i,task['source_motion']):
                raise ValueError('OFF commitment identity/source mismatch')
            values=validate_parameters([proposal['parameters'][n] for n in NAMES],protocol)
            if i==0 and values!=protocol['anchor']: raise ValueError('OFF anchor mismatch')
            for key in ('observation_artifact','response_artifact'):
                ref=proposal[key]
                if sha256(out/ref['path'])!=ref['sha256']: raise ValueError('changed OFF commitment artifact')
            obs=observation_path(out,task['task_id'],'llm_feedback_off',i)
            response=response_path(out,task['task_id'],'llm_feedback_off',i)
            if proposal['observation_artifact']!=artifact(out,obs) or proposal['response_artifact']!=artifact(out,response):
                raise ValueError('OFF linked path mismatch')
            if read(obs)!=observation_data(out,task['task_id'],'llm_feedback_off',i): raise ValueError('OFF observation changed')
            raw=read(response)
            if Path(raw['only_observation_path']).resolve()!=obs.resolve() or validate_parameters(raw['parameters'],protocol)!=values:
                raise ValueError('OFF response changed')

def record_evaluation(out,candidate,task_id,calibration=False):
    protocol,task=context(out,task_id);verify_code(protocol)
    trial=out/f'trials/{candidate}'; proposal_path=out/f'proposals/{candidate}.json'; proposal=read(proposal_path)
    if sha256(trial/'proposal.json')!=sha256(proposal_path): raise ValueError('executed proposal mismatch')
    if proposal['protocol_sha256']!=sha256(out/'protocol.json'): raise ValueError('proposal protocol mismatch')
    if proposal['source_motion']!=task['source_motion']: raise ValueError('proposal source mismatch')
    for key in ('observation_artifact','response_artifact'):
        ref=proposal.get(key)
        if ref and sha256(out/ref['path'])!=ref['sha256']: raise ValueError('changed proposal artifact')
    baseline=trial if calibration else out/f'trials/calibration_{task["source_id"]}'
    result=evaluate_generalization(trial,baseline,protocol,task['goal']); manifest=read(trial/'manifest.json')
    result.update(candidate_id=candidate,task_id=task_id,method='calibration' if calibration else proposal['method'],
        parameters=[proposal['parameters'][n] for n in NAMES],wall_seconds=manifest['wall_seconds'],
        executor_status=manifest['status'],control_steps=manifest['actual_control_steps'],physics_steps=manifest['actual_physics_steps'],
        protocol_sha256=sha256(out/'protocol.json'),execution_manifest_sha256=sha256(trial/'manifest.json'),
        evaluator_sha256=sha256(Path(__file__).with_name('evaluate_generalization.py')))
    if calibration: result.update(intent_passed=None,utility=None,utility_l2=None,calibration_only=True)
    write_new_json(out/f'evaluations/{candidate}.json',result)
    refs={key:artifact(out,trial/name) for key,name in [('execution_manifest','manifest.json'),('actual_qpos','qpos.npy'),('actual_qvel','qvel.npy'),('actual_policy_inputs','policy_inputs.npz'),('applied_pd_targets','applied_joint_targets.npy')]}
    record=dict(schema='pulsemorph.agentic_generalization_trial_record.v1',candidate_id=candidate,
        task_id=task_id,source_group=task['source_motion']['sha256'],source_motion=task['source_motion'],
        split='unassigned',quality='unassessed',training_admission=False,shared_corpus_imported=False,
        proposal=artifact(out,proposal_path),evaluation=artifact(out,out/f'evaluations/{candidate}.json'),
        observation=proposal.get('observation_artifact'),model_response=proposal.get('response_artifact'),
        reference=artifact(out,trial/'common_reference/manifest.json'),provider_tokens=None if proposal['method'].startswith('llm') else 0,
        provider_billed_cost=None if proposal['method'].startswith('llm') else 0,
        protocol=artifact(out,out/'protocol.json'),model_reason_is_measurement=False,**refs)
    write_new_json(out/f'trial_records/{candidate}.json',record)
    print(json.dumps(compact(result),allow_nan=False),flush=True);return result

def run(out,task_id,method,index):
    protocol,task=context(out,task_id,method,index);verify_code(protocol);require_precommit(out,protocol)
    candidate=identifier(task_id,method,index)
    if (out/f'trials/{candidate}').exists() or (out/f'evaluations/{candidate}.json').exists(): raise ValueError('trial slot already consumed')
    previous=histories(out,task_id,method,index,True)[1]
    proposal_path=out/f'proposals/{candidate}.json'
    if method=='surrogate_search':
        started=time.perf_counter(); values=next_candidate(previous,protocol,task['goal'])
        obs=observation_path(out,task_id,method,index)
        write_new_json(obs,dict(observation_id=candidate,task_id=task_id,method=method,prior_trial_results=previous,
                               protocol_sha256=sha256(out/'protocol.json'),proposal_compute_seconds=time.perf_counter()-started))
        commit(out,task_id,method,index,values,obs,reason='Deterministic bounded surrogate using only own measured history.')
    elif method=='llm_feedback' and not proposal_path.exists(): commit_model(out,task_id,method,index)
    proposal=read(proposal_path)
    if proposal['protocol_sha256']!=sha256(out/'protocol.json') or proposal['source_motion']!=task['source_motion']: raise ValueError('stale proposal')
    for key in ('observation_artifact','response_artifact'):
        ref=proposal.get(key)
        if ref and sha256(out/ref['path'])!=ref['sha256']: raise ValueError('changed committed artifact')
    manifest=execute_trial(proposal_path,out/f'trials/{candidate}')
    result=record_evaluation(out,candidate,task_id)
    if method=='llm_feedback' and index<BUDGET-1: observation(out,task_id,method,index+1)
    if manifest['status']!='success' or not result['integrity_passed']: raise RuntimeError('failed consumed slot: '+candidate)
    return result

def select(rows):
    eligible=[r for r in rows if r['integrity_passed'] and r['physical_gate_passed'] and r.get('utility') is not None and r.get('utility_l2') is not None]
    return min(eligible,key=lambda r:(r['utility'],r['utility_l2'],r['candidate_id'])) if eligible else None

def summarize(out):
    protocol=read(out/'protocol.json');verify_code(protocol);require_precommit(out,protocol)
    groups=[];allrows=[]
    for task in protocol['tasks']:
        methods=[]
        for method in METHODS:
            rows=[read(out/f'evaluations/{identifier(task["task_id"],method,i)}.json') for i in range(BUDGET)]
            for i,row in enumerate(rows):
                if row['candidate_id']!=identifier(task['task_id'],method,i) or row['protocol_sha256']!=sha256(out/'protocol.json'): raise ValueError('invalid completion stream')
            allrows.extend(rows)
            methods.append(dict(method=method,trials=rows,selected=select(rows),physical_trials=len(rows),
                model_proposal_calls=8 if method.startswith('llm') else 0,provider_tokens=None if method.startswith('llm') else 0,
                provider_billed_cost=None if method.startswith('llm') else 0))
        groups.append(dict(task_id=task['task_id'],source_id=task['source_id'],goal=task['goal'],methods=methods))
    calibrations=[read(out/f'evaluations/calibration_{source}.json') for source in sorted({t['source_id'] for t in protocol['tasks']})]
    rows=allrows+calibrations
    if len(allrows)!=96 or len(calibrations)!=2: raise ValueError('expected 96 candidates and two calibrations')
    expected={row['candidate_id'] for row in rows}
    actual={path.name for path in (out/'trials').iterdir() if path.is_dir()}
    if expected!=actual: raise ValueError('actual trial directories differ from completed budget')
    for row in rows:
        trial=out/'trials'/row['candidate_id'];manifest=read(trial/'manifest.json')
        if sha256(trial/'manifest.json')!=row['execution_manifest_sha256']: raise ValueError('completed execution manifest changed')
        if (row['control_steps'],row['physics_steps'])!=(manifest['actual_control_steps'],manifest['actual_physics_steps']):
            raise ValueError('actual execution count mismatch')
    summary=dict(schema='pulsemorph.agentic_generalization_comparison.v1',protocol_sha256=sha256(out/'protocol.json'),
        tasks=groups,calibrations=calibrations,comparison_valid=all(r['integrity_passed'] and r['executor_status']=='success' for r in rows),
        physical_trials=len(rows),control_steps=sum(r['control_steps'] for r in rows),physics_steps=sum(r['physics_steps'] for r in rows),
        planned_budget={'physical_trials':98,'candidate_trials':96,'calibration_trials':2},
        limitations=protocol.get('limitations',[])+['Canonical goals supplied; no semantic-stage score.','Same source family/body/motor; no unseen-human claim.','Provider token usage and billing unavailable.'])
    write_new_json(out/'comparison.json',summary);return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',required=True,type=Path)
    parser.add_argument('--action',required=True,choices=['observe','commit-model','run','evaluate-calibration','summarize'])
    parser.add_argument('--task');parser.add_argument('--method');parser.add_argument('--index',type=int)
    args=parser.parse_args();out=args.pilot.resolve()
    if args.action=='observe': observation(out,args.task,args.method,args.index)
    elif args.action=='commit-model': commit_model(out,args.task,args.method,args.index)
    elif args.action=='run': run(out,args.task,args.method,args.index)
    elif args.action=='evaluate-calibration':
        protocol,task=context(out,args.task);record_evaluation(out,'calibration_'+task['source_id'],args.task,True)
    else: print(json.dumps(summarize(out),indent=2,allow_nan=False))

if __name__=='__main__': main()
