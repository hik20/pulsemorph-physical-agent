"""Immutable DEVELOPMENT workflow for two equally available clock tools.

CLI: python experiments/agentic/tool_selection_driver_v2.py --pilot PATH --action
evaluate-calibration --task wave_a; observe/commit-model/run additionally take
--task TASK --method METHOD --index 0..7. Model responses are saved separately
at responses/TASK__METHOD__INDEX.json before commit-model. Commit all 32 OFF
responses before run. summarize requires all 96 candidate evaluations and both
global/local calibrations per source (100 total physics). No provider is called.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from musical_tasks import ROOT, NAMES, read, sha256, write_new_json
from multifeature_driver import compact as legacy_compact, validate_parameters, stamp
from execute_tool_selection import execute_trial
from evaluate_generalization import evaluate_generalization
from tool_selection_search import next_candidate, TOOLS

METHODS = ('surrogate_search', 'llm_feedback', 'llm_feedback_off')
BUDGET = 8

def compact(result):
    return dict(legacy_compact(result), control_tool=result.get('control_tool'))

def calibration_view(result):
    """Shared measured calibration, independent of the task used to audit it.

    Never expose calibration goal scores as if they belonged to another task.
    Candidate history continues to use its own task-specific compact scores.
    """
    keys=('candidate_id','parameters','control_tool','features','integrity_passed',
          'physical_gate_passed','mean_joint_rmse_rad','min_root_height_m')
    value={key:result.get(key) for key in keys}
    if isinstance(result.get('accent'),dict):
        observed=('available','peak_time_s','peak_rms_speed_rad_s','event_times_s',
                  'event_count','window_event_count')
        value['accent']={key:result['accent'][key] for key in observed if key in result['accent']}
    return value

def validate_tool(tool):
    if tool not in TOOLS: raise ValueError('unknown control_tool')
    return tool

def validate_response(raw, protocol):
    if not isinstance(raw, dict) or set(raw) != {'control_tool','parameters','only_observation_path','brief_reason'}:
        raise ValueError('response requires exactly control_tool, parameters, only_observation_path, brief_reason')
    validate_tool(raw['control_tool'])
    validate_parameters(raw['parameters'], protocol)
    if not isinstance(raw['only_observation_path'], str) or not isinstance(raw['brief_reason'], str):
        raise ValueError('response path/reason must be strings')

def calibration_ids(source):
    return ('calibration_'+source, 'calibration_local_'+source)

def _array_parity(first, second):
    """Compare decompressed payload bytes, dtype, shape and full key inventory."""
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if set(a.files) != set(b.files): raise ValueError('calibration archive key mismatch')
        for key in a.files:
            if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape or a[key].tobytes() != b[key].tobytes():
                raise ValueError('calibration array parity mismatch: '+key)

def validate_calibrations(out):
    """Recheck all four fresh calibrations before observations/candidate physics."""
    protocol=read(out/'protocol.json'); verify_code(protocol)
    sources=sorted({t['source_id'] for t in protocol['tasks']})
    if sources != ['high5','wave']: raise ValueError('expected wave/high5 development sources')
    reports=[]
    for source in sources:
        ids=calibration_ids(source); dirs=[out/'trials'/identity for identity in ids]
        manifests=[read(d/'manifest.json') for d in dirs]
        evaluations=[read(out/f'evaluations/{identity}.json') for identity in ids]
        task=next(t for t in protocol['tasks'] if t['source_id']==source)
        for identity, tool, directory, manifest, evaluation in zip(ids, TOOLS, dirs, manifests, evaluations):
            proposal=read(out/f'proposals/{identity}.json')
            if (proposal['source_motion'] != task['source_motion'] or proposal['parameters'] != dict(zip(NAMES,protocol['anchor']))
                    or proposal['control_tool'] != tool or proposal['protocol_sha256'] != sha256(out/'protocol.json')):
                raise ValueError('fresh calibration proposal mismatch')
            if sha256(directory/'proposal.json') != sha256(out/f'proposals/{identity}.json'):
                raise ValueError('fresh calibration executed proposal mismatch')
            if (evaluation['candidate_id'] != identity or evaluation['control_tool'] != tool
                    or evaluation['protocol_sha256'] != sha256(out/'protocol.json')
                    or evaluation['execution_manifest_sha256'] != sha256(directory/'manifest.json')
                    or not evaluation['integrity_passed'] or not evaluation['physical_gate_passed']
                    or manifest['status'] != 'success' or manifest['control_tool'] != tool):
                raise ValueError('fresh calibration integrity/gate mismatch')
            hashes=manifest['artifact_hashes']
            actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p != directory/'manifest.json'}
            if actual != set(hashes): raise ValueError('calibration artifact inventory mismatch')
            for path,digest in hashes.items():
                if sha256(directory/path) != digest: raise ValueError('calibration artifact changed: '+path)
            if read(directory/'preexecution_inputs.json') != manifest['preexecution_input_hashes']:
                raise ValueError('calibration input pins mismatch')
        for key in ('preexecution_input_hashes','onnx_sha256','motion_sha256','physics_contract','versions'):
            if manifests[0][key] != manifests[1][key]: raise ValueError('calibration full preexecution parity: '+key)
        for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy','state_times.npy'):
            if (dirs[0]/name).read_bytes() != (dirs[1]/name).read_bytes():
                raise ValueError('calibration byte parity mismatch: '+name)
        for name in ('policy_inputs.npz','reference_cache.npz','common_reference/arrays.npz'):
            _array_parity(dirs[0]/name,dirs[1]/name)
        reports.append({'source_id':source,'global':ids[0],'local':ids[1],'byte_and_array_parity':True})
    return dict(schema='pulsemorph.agentic_tool_selection_calibration_parity.v1',
        protocol_sha256=sha256(out/'protocol.json'),calibration_trials=4,parity_passed=True,sources=reports)

def identifier(task, method, index): return f'{task}__{method}__{index}'
def observation_path(out, task, method, index): return out/'observations'/f'{identifier(task,method,index)}.json'
def response_path(out, task, method, index): return out/'responses'/f'{identifier(task,method,index)}.json'
def case(protocol, task):
    matches = [t for t in protocol['tasks'] if t['task_id'] == task]
    if len(matches) != 1: raise ValueError('unknown or duplicate task')
    return matches[0]

def context(out, task_id, method=None, index=None):
    if (out/'abort.json').exists(): raise RuntimeError('experiment aborted; no retry')
    protocol = read(out/'protocol.json'); task = case(protocol, task_id)
    if protocol.get('schema') != 'pulsemorph.agentic_tool_selection_protocol.v2':
        raise ValueError('tool selection protocol v2 required')
    if sorted(t['task_id'] for t in protocol['tasks']) != ['high5_a','high5_b','wave_a','wave_b']:
        raise ValueError('fixed four development tasks required')
    if (protocol.get('new_candidate_budget',8) != 8 or protocol.get('trial_budget_per_method_per_task',8) != 8
            or protocol.get('methods',list(METHODS)) != list(METHODS)
            or protocol.get('control_tools',list(TOOLS)) != list(TOOLS)
            or protocol.get('imported_prior_trial_records',[]) != []):
        raise ValueError('fixed methods/tools/budget and no imported prior labels required')
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
        validate_tool(proposal['control_tool'])
        if proposal['source_motion'] != case(protocol,task)['source_motion']: raise ValueError('history source mismatch')
        for key in ('observation_artifact','response_artifact'):
            ref=proposal.get(key)
            if ref and artifact(out,out/ref['path']) != ref: raise ValueError('changed own history artifact')
        proposals.append(dict(candidate_id=identity,control_tool=proposal['control_tool'],parameters=[proposal['parameters'][n] for n in NAMES],
                              proposal_sha256=sha256(out/f'proposals/{identity}.json')))
        if outcomes:
            result=read(out/f'evaluations/{identity}.json')
            if (result['candidate_id'] != identity or result['control_tool'] != proposal['control_tool']
                    or result['protocol_sha256'] != sha256(out/'protocol.json')): raise ValueError('cross-stream outcome history')
            trial=out/'trials'/identity
            if (sha256(trial/'manifest.json') != result['execution_manifest_sha256']
                    or sha256(trial/'proposal.json') != sha256(out/f'proposals/{identity}.json')):
                raise ValueError('own outcome provenance mismatch')
            results.append(compact(result))
    return proposals, results

def observation_data(out, task_id, method, index):
    validate_calibrations(out)
    return _observation_data(out,task_id,method,index)

def _observation_data(out, task_id, method, index):
    """Pure assembly; public callers must first revalidate fresh calibration bytes."""
    protocol,task=context(out,task_id,method,index)
    if method == 'surrogate_search': raise ValueError('observe is for recorded model calls')
    proposals,results=histories(out,task_id,method,index,method=='llm_feedback')
    baseline=read(out/f'evaluations/calibration_{task["source_id"]}.json')
    return dict(schema='pulsemorph.agentic_tool_selection_observation.v2',
        observation_id=identifier(task_id,method,index),task_id=task_id,method=method,trial_index=index,
        protocol_sha256=sha256(out/'protocol.json'),source_id=task['source_id'],source_motion=task['source_motion'],
        user_instruction=task['instruction'],canonical_goal=task['goal'],baseline=calibration_view(baseline),
        parameter_names=NAMES,bounds=protocol['bounds'],feature_names=protocol['feature_names'],
        feature_tolerances=protocol['feature_tolerances'],physical_gate=protocol['physical_gate'],selection=protocol['selection'],
        music_observation=task['music_observation'],prior_proposals=proposals,prior_trial_results=results,
        trials_remaining=BUDGET-index,forced_first_parameters=None,
        tools=tool_descriptions(protocol),
        calibration_provenance=artifact(out,out/f'evaluations/calibration_{task["source_id"]}.json'),
        access='Use only this observation and your own previous supplied observations. No sibling files or code. Feedback-off has no candidate outcomes.',
        response_schema={'control_tool':list(TOOLS),'parameters':'one flat list of five numbers','only_observation_path':str(observation_path(out,task_id,method,index).resolve()),'brief_reason':'short decision summary'},
        instructions='Return exactly one tool and one vector in the exact response schema. Every slot is freely chosen; identity calibration is already supplied. No extra candidates or keys.')

def tool_descriptions(protocol):
    common=dict(section_windows_s=protocol['section_windows_s'],measurement_windows_s=protocol['feature_windows_s'],
        gains='Independent per-side initial-posture displacement gains with sin-squared section envelopes. Achieved RMS ratios differ from gains.',
        output='Validated MotionReference -> fixed pretrained G1 ONNX -> PD -> MuJoCo. One offline take; no learning.')
    return [dict(common,control_tool=tool,clock=clock) for tool,clock in zip(TOOLS,[
        'source_time=t+timing_shift_s*sin(pi*t/4)^2 for 0<t<4; identity outside; analytic chain-rule velocity.',
        'source_time=t+timing_shift_s*sin(pi*(t-1.4)/1.4)^2 for 1.4<t<2.8; identity outside; analytic chain-rule velocity.'])]

def observation(out,task,method,index):
    path=observation_path(out,task,method,index)
    write_new_json(path,observation_data(out,task,method,index)); return path

def any_candidates(out):
    return any(p.is_dir() and not p.name.startswith('calibration_') for p in (out/'trials').glob('*'))

def commit(out,task_id,method,index,values,obs=None,response=None,reason='',control_tool=None,search_decision=None):
    validate_calibrations(out)
    return _commit(out,task_id,method,index,values,obs,response,reason,control_tool,search_decision)

def _commit(out,task_id,method,index,values,obs=None,response=None,reason='',control_tool=None,search_decision=None):
    protocol,task=context(out,task_id,method,index);verify_code(protocol)
    values=validate_parameters(values,protocol)
    validate_tool(control_tool)
    if method=='llm_feedback_off' and any_candidates(out): raise ValueError('OFF proposals must all precede candidates')
    histories(out,task_id,method,index,method!='llm_feedback_off')
    if method.startswith('llm'):
        expected=observation_path(out,task_id,method,index).resolve()
        if obs is None or response is None or Path(obs).resolve()!=expected or Path(response).resolve()!=response_path(out,task_id,method,index).resolve():
            raise ValueError('wrong response/observation path')
        observed=read(obs);raw=read(response)
        validate_response(raw,protocol)
        if observed != _observation_data(out,task_id,method,index): raise ValueError('stale or cross-stream observation/history')
        if Path(raw['only_observation_path']).resolve()!=expected: raise ValueError('response observation mismatch')
        if validate_parameters(raw['parameters'],protocol)!=values: raise ValueError('response vector mismatch')
        if raw['control_tool'] != control_tool: raise ValueError('response tool mismatch')
        reason=raw.get('brief_reason','')
    candidate=identifier(task_id,method,index)
    proposal=dict(schema='pulsemorph.agentic_tool_selection_proposal.v1',candidate_id=candidate,task_id=task_id,method=method,trial_index=index,
        control_tool=control_tool,search_decision=search_decision,
        source_id=task['source_id'],source_motion=task['source_motion'],parameters=dict(zip(NAMES,values)),
        duration_s=4.,source_start_s=0.,section_windows_s=protocol['section_windows_s'],
        audio={'duration_s':4.,'observed_through_sample':88200},protocol_sha256=sha256(out/'protocol.json'),
        created_at_utc=stamp(),observation_id=candidate,
        observation_artifact=artifact(out,obs) if obs else None,response_artifact=artifact(out,response) if response else None,brief_reason=reason)
    write_new_json(out/f'proposals/{candidate}.json',proposal)
    return proposal

def commit_model(out,task,method,index):
    validate_calibrations(out)
    return _commit_model(out,task,method,index)

def _commit_model(out,task,method,index):
    if method not in ('llm_feedback','llm_feedback_off'): raise ValueError('model method required')
    response=response_path(out,task,method,index)
    try:
        raw=read(response)
        protocol,_=context(out,task,method,index);validate_response(raw,protocol)
        result=_commit(out,task,method,index,raw['parameters'],observation_path(out,task,method,index),response,control_tool=raw['control_tool'])
    except Exception as error:
        abort(out,identifier(task,method,index),error,'invalid_model_response')
        raise
    if method=='llm_feedback_off' and index<BUDGET-1:
        write_new_json(observation_path(out,task,method,index+1),_observation_data(out,task,method,index+1))
    return result

def abort(out,candidate,error,stage):
    if not (out/'abort.json').exists():
        write_new_json(out/'abort.json',dict(schema='pulsemorph.agentic_tool_selection_abort.v1',candidate_id=candidate,
            stage=stage,error=str(error),slot_consumed=True,retry_allowed=False,created_at_utc=stamp()))

def require_precommit(out,protocol):
    validate_calibrations(out)
    return _require_precommit(out,protocol)

def _require_precommit(out,protocol):
    for task in protocol['tasks']:
        for i in range(BUDGET):
            identity=identifier(task['task_id'],'llm_feedback_off',i)
            proposal=read(out/f'proposals/{identity}.json')
            if proposal['protocol_sha256']!=sha256(out/'protocol.json'): raise ValueError('stale OFF commitment')
            if (proposal['candidate_id'],proposal['task_id'],proposal['method'],proposal['trial_index'],proposal['source_motion']) != (identity,task['task_id'],'llm_feedback_off',i,task['source_motion']):
                raise ValueError('OFF commitment identity/source mismatch')
            values=validate_parameters([proposal['parameters'][n] for n in NAMES],protocol)
            validate_tool(proposal['control_tool'])
            for key in ('observation_artifact','response_artifact'):
                ref=proposal[key]
                if sha256(out/ref['path'])!=ref['sha256']: raise ValueError('changed OFF commitment artifact')
            obs=observation_path(out,task['task_id'],'llm_feedback_off',i)
            response=response_path(out,task['task_id'],'llm_feedback_off',i)
            if proposal['observation_artifact']!=artifact(out,obs) or proposal['response_artifact']!=artifact(out,response):
                raise ValueError('OFF linked path mismatch')
            if read(obs)!=_observation_data(out,task['task_id'],'llm_feedback_off',i): raise ValueError('OFF observation changed')
            raw=read(response)
            validate_response(raw,protocol)
            if raw['control_tool'] != proposal['control_tool']: raise ValueError('OFF tool changed')
            if Path(raw['only_observation_path']).resolve()!=obs.resolve() or validate_parameters(raw['parameters'],protocol)!=values:
                raise ValueError('OFF response changed')

def record_evaluation(out,candidate,task_id,calibration=False):
    protocol,task=context(out,task_id);verify_code(protocol)
    trial=out/f'trials/{candidate}'; proposal_path=out/f'proposals/{candidate}.json'; proposal=read(proposal_path)
    if sha256(trial/'proposal.json')!=sha256(proposal_path): raise ValueError('executed proposal mismatch')
    if proposal['protocol_sha256']!=sha256(out/'protocol.json'): raise ValueError('proposal protocol mismatch')
    if proposal['source_motion']!=task['source_motion']: raise ValueError('proposal source mismatch')
    validate_tool(proposal['control_tool'])
    if calibration:
        if candidate != calibration_ids(task['source_id'])[TOOLS.index(proposal['control_tool'])]:
            raise ValueError('calibration id/tool mismatch')
        if proposal['parameters'] != dict(zip(NAMES,protocol['anchor'])): raise ValueError('calibration must be identity')
    for key in ('observation_artifact','response_artifact'):
        ref=proposal.get(key)
        if ref and sha256(out/ref['path'])!=ref['sha256']: raise ValueError('changed proposal artifact')
    baseline=trial if calibration else out/f'trials/calibration_{task["source_id"]}'
    result=evaluate_generalization(trial,baseline,protocol,task['goal']); manifest=read(trial/'manifest.json')
    if manifest.get('control_tool') != proposal['control_tool']: raise ValueError('execution control_tool mismatch')
    result.update(candidate_id=candidate,task_id=task_id,method='calibration' if calibration else proposal['method'],
        control_tool=validate_tool(proposal['control_tool']),
        parameters=[proposal['parameters'][n] for n in NAMES],wall_seconds=manifest['wall_seconds'],
        executor_status=manifest['status'],control_steps=manifest['actual_control_steps'],physics_steps=manifest['actual_physics_steps'],
        protocol_sha256=sha256(out/'protocol.json'),execution_manifest_sha256=sha256(trial/'manifest.json'),
        evaluator_sha256=sha256(Path(__file__).with_name('evaluate_generalization.py')))
    if calibration: result.update(intent_passed=None,utility=None,utility_l2=None,calibration_only=True)
    write_new_json(out/f'evaluations/{candidate}.json',result)
    refs={key:artifact(out,trial/name) for key,name in [('execution_manifest','manifest.json'),('actual_qpos','qpos.npy'),('actual_qvel','qvel.npy'),('actual_policy_inputs','policy_inputs.npz'),('applied_pd_targets','applied_joint_targets.npy')]}
    record=dict(schema='pulsemorph.agentic_tool_selection_trial_record.v1',candidate_id=candidate,control_tool=proposal['control_tool'],
        task_id=task_id,source_group=task['source_motion']['sha256'],source_motion=task['source_motion'],
        split='unassigned',quality='unassessed',training_admission=False,shared_corpus_imported=False,
        proposal=artifact(out,proposal_path),evaluation=artifact(out,out/f'evaluations/{candidate}.json'),
        observation=proposal.get('observation_artifact'),model_response=proposal.get('response_artifact'),
        reference=artifact(out,trial/'common_reference/manifest.json'),provider_tokens=None if proposal['method'].startswith('llm') else 0,
        provider_billed_cost=None if proposal['method'].startswith('llm') else 0,
        protocol=artifact(out,out/'protocol.json'),model_reason_is_measurement=False,**refs)
    write_new_json(out/f'trial_records/{candidate}.json',record)
    if calibration and (manifest['status'] != 'success' or not result['integrity_passed'] or not result['physical_gate_passed']):
        abort(out,candidate,'failed fresh calibration','calibration_execution_or_integrity')
        raise RuntimeError('failed fresh calibration: '+candidate)
    print(json.dumps(compact(result),allow_nan=False),flush=True);return result

def _run(out,task_id,method,index):
    protocol,task=context(out,task_id,method,index);verify_code(protocol);validate_calibrations(out);_require_precommit(out,protocol)
    candidate=identifier(task_id,method,index)
    if (out/f'trials/{candidate}').exists() or (out/f'evaluations/{candidate}.json').exists(): raise ValueError('trial slot already consumed')
    previous=histories(out,task_id,method,index,True)[1]
    proposal_path=out/f'proposals/{candidate}.json'
    if method=='surrogate_search':
        baseline_path=out/f'evaluations/calibration_{task["source_id"]}.json'
        baseline=read(baseline_path)
        calibration=dict(parameters=protocol['anchor'],features=baseline['features'],integrity_passed=True,
            physical_gate_passed=True,provenance=artifact(out,baseline_path),control_tool=TOOLS[0])
        started=time.perf_counter(); choice=next_candidate(previous,protocol,task['goal'],calibration)
        values=choice['parameters'];validate_tool(choice['control_tool'])
        obs=observation_path(out,task_id,method,index)
        write_new_json(obs,dict(schema='pulsemorph.agentic_tool_selection_search_observation.v1',observation_id=candidate,task_id=task_id,method=method,prior_trial_results=previous,
                               canonical_goal=task['goal'],music_observation=task['music_observation'],baseline=calibration_view(baseline),
                               calibration_provenance=artifact(out,baseline_path),tools=tool_descriptions(protocol),
                               search_decision=choice['decision'],control_tool=choice['control_tool'],
                               protocol_sha256=sha256(out/'protocol.json'),proposal_compute_seconds=time.perf_counter()-started))
        _commit(out,task_id,method,index,values,obs,reason='Deterministic mixed-tool search using only fresh calibration and own measured history.',
            control_tool=choice['control_tool'],search_decision=choice['decision'])
    elif method=='llm_feedback' and not proposal_path.exists(): _commit_model(out,task_id,method,index)
    proposal=read(proposal_path)
    validate_tool(proposal['control_tool'])
    if proposal['protocol_sha256']!=sha256(out/'protocol.json') or proposal['source_motion']!=task['source_motion']: raise ValueError('stale proposal')
    for key in ('observation_artifact','response_artifact'):
        ref=proposal.get(key)
        if ref and sha256(out/ref['path'])!=ref['sha256']: raise ValueError('changed committed artifact')
    manifest=execute_trial(proposal_path,out/f'trials/{candidate}')
    result=record_evaluation(out,candidate,task_id)
    if manifest['status']!='success' or not result['integrity_passed']: raise RuntimeError('failed consumed slot: '+candidate)
    if method=='llm_feedback' and index<BUDGET-1:
        write_new_json(observation_path(out,task_id,method,index+1),_observation_data(out,task_id,method,index+1))
    return result

def run(out,task_id,method,index):
    try:
        return _run(out,task_id,method,index)
    except Exception as error:
        abort(out,identifier(task_id,method,index),error,'candidate_execution_or_integrity')
        raise

def select(rows):
    eligible=[r for r in rows if r['integrity_passed'] and r['physical_gate_passed'] and r.get('utility') is not None and r.get('utility_l2') is not None]
    return min(eligible,key=lambda r:(r['utility'],r['utility_l2'],r['candidate_id'])) if eligible else None

def summarize(out):
    if (out/'abort.json').exists(): raise RuntimeError('experiment aborted; no completion summary')
    protocol=read(out/'protocol.json');verify_code(protocol);parity=validate_calibrations(out);_require_precommit(out,protocol)
    groups=[];allrows=[]
    for task in protocol['tasks']:
        methods=[]
        for method in METHODS:
            rows=[read(out/f'evaluations/{identifier(task["task_id"],method,i)}.json') for i in range(BUDGET)]
            for i,row in enumerate(rows):
                if row['candidate_id']!=identifier(task['task_id'],method,i) or row['protocol_sha256']!=sha256(out/'protocol.json'): raise ValueError('invalid completion stream')
                proposal_path=out/f'proposals/{row["candidate_id"]}.json';proposal=read(proposal_path)
                if row['control_tool'] != validate_tool(proposal['control_tool']): raise ValueError('completed tool mismatch')
                if sha256(out/'trials'/row['candidate_id']/'proposal.json') != sha256(proposal_path): raise ValueError('completed proposal changed')
                if method.startswith('llm'):
                    receipt=read(out/f'model_calls/{row["candidate_id"]}.json')
                    if ((receipt['candidate_id'],receipt['task_id'],receipt['method'],receipt['trial_index']) != (row['candidate_id'],task['task_id'],method,i)
                            or receipt['requested_model'] != 'gpt-6.1-sol' or receipt['fork_turns'] != 'none'
                            or receipt['observation_sha256'] != sha256(observation_path(out,task['task_id'],method,i))
                            or receipt['response_sha256'] != sha256(response_path(out,task['task_id'],method,i))):
                        raise ValueError('invalid model call receipt')
            allrows.extend(rows)
            winner=select(rows)
            methods.append(dict(method=method,trials=rows,selected=winner,selected_tool=winner['control_tool'] if winner else None,
                tool_usage={tool:sum(r['control_tool']==tool for r in rows) for tool in TOOLS},
                prefix_best=[dict(budget=i,selected=select(rows[:i])) for i in range(1,BUDGET+1)],physical_trials=len(rows),
                model_proposal_calls=8 if method.startswith('llm') else 0,provider_tokens=None if method.startswith('llm') else 0,
                provider_billed_cost=None if method.startswith('llm') else 0))
        groups.append(dict(task_id=task['task_id'],source_id=task['source_id'],goal=task['goal'],methods=methods))
    calibrations=[read(out/f'evaluations/{identity}.json') for source in sorted({t['source_id'] for t in protocol['tasks']}) for identity in calibration_ids(source)]
    rows=allrows+calibrations
    if len(allrows)!=96 or len(calibrations)!=4: raise ValueError('expected 96 candidates and four calibrations')
    expected_calls={r['candidate_id'] for r in allrows if r['method'].startswith('llm')}
    if {p.stem for p in (out/'model_calls').glob('*.json')} != expected_calls: raise ValueError('expected exactly 64 model call receipts')
    expected={row['candidate_id'] for row in rows}
    actual={path.name for path in (out/'trials').iterdir() if path.is_dir()}
    if expected!=actual: raise ValueError('actual trial directories differ from completed budget')
    for row in rows:
        trial=out/'trials'/row['candidate_id'];manifest=read(trial/'manifest.json')
        if sha256(trial/'manifest.json')!=row['execution_manifest_sha256']: raise ValueError('completed execution manifest changed')
        if (row['control_steps'],row['physics_steps'])!=(manifest['actual_control_steps'],manifest['actual_physics_steps']):
            raise ValueError('actual execution count mismatch')
    summary=dict(schema='pulsemorph.agentic_tool_selection_comparison.v1',protocol_sha256=sha256(out/'protocol.json'),
        development_only=True,calibration_parity=parity,model_proposal_calls=64,
        tasks=groups,calibrations=calibrations,comparison_valid=all(r['integrity_passed'] and r['executor_status']=='success' for r in rows),
        physical_trials=len(rows),control_steps=sum(r['control_steps'] for r in rows),physics_steps=sum(r['physics_steps'] for r in rows),
        planned_budget={'physical_trials':100,'candidate_trials':96,'calibration_trials':4},
        limitations=protocol.get('limitations',[])+['Canonical goals supplied; no semantic-stage score.','Same source family/body/motor; no unseen-human claim.','Provider token usage and billing unavailable.'])
    write_new_json(out/'comparison.json',summary);return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',required=True,type=Path)
    parser.add_argument('--action',required=True,choices=['observe','commit-model','run','evaluate-calibration','validate-calibrations','summarize'])
    parser.add_argument('--task');parser.add_argument('--method');parser.add_argument('--index',type=int);parser.add_argument('--control-tool',choices=TOOLS,default=TOOLS[0])
    args=parser.parse_args();out=args.pilot.resolve()
    if args.action=='observe': observation(out,args.task,args.method,args.index)
    elif args.action=='commit-model': commit_model(out,args.task,args.method,args.index)
    elif args.action=='run': run(out,args.task,args.method,args.index)
    elif args.action=='evaluate-calibration':
        protocol,task=context(out,args.task);record_evaluation(out,calibration_ids(task['source_id'])[TOOLS.index(args.control_tool)],args.task,True)
    elif args.action=='validate-calibrations': print(json.dumps(validate_calibrations(out)))
    else: print(json.dumps(summarize(out),indent=2,allow_nan=False))

if __name__=='__main__': main()
