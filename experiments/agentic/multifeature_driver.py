"""Observable, bounded multi-feature planning trials with independent screening."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from musical_tasks import ROOT, NAMES, METHODS, read, sha256, write_new_json
from execute_multifeature import execute_trial
from evaluate_multifeature import evaluate_multifeature, next_jacobian_candidate


def stamp():return datetime.now(timezone.utc).isoformat()


def case(protocol,task_id):
    return next(t for t in protocol['tasks'] if t['task_id']==task_id)


def identifier(task_id,method,index):return f'{task_id}__{method}__{index}'


def response_path(out,task_id,method,index):
    return out/f'responses/{task_id}__{method}__{index}.json'


def observation_path(out,task_id,method,index):
    return out/f'observations/{task_id}__{method}__{index}.json'


def validate_parameters(values,protocol):
    if not isinstance(values,list) or len(values)!=5 or any(isinstance(v,bool) or not isinstance(v,(float,int)) for v in values):
        raise ValueError('five numeric ordered parameters required')
    x=np.asarray(values,float);bounds=np.asarray(protocol['bounds'])
    if not np.isfinite(x).all() or np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]):raise ValueError('parameters outside declared tool domain')
    return x.tolist()


def model_response(path):
    """Lossless shape repair only: a single flat five-vector is one proposal.

    Preserve the original response artifact and disclose normalization. This
    cannot add or change a candidate or request another model generation.
    """
    raw=read(path);values=raw.get('parameters')
    flat=isinstance(values,list) and len(values)==5 and all(type(v) in (int,float) for v in values)
    return {**raw,'parameters':[values] if flat else values,
            'parameter_layout_normalization':'wrapped_single_flat_vector' if flat else 'none'}


def compact(result):
    keys=['candidate_id','parameters','integrity_passed','integrity_errors','physical_gate_passed','intent_passed',
          'features','feature_errors','per_dimension_pass','utility','utility_l2','mean_joint_rmse_rad','min_root_height_m','wall_seconds']
    value={k:result.get(k) for k in keys}
    if 'accent' in result:
        value['accent']={k:result['accent'][k] for k in ['available','peak_time_s','peak_rms_speed_rad_s','event_times_s','event_count','window_event_count']}
    return value


def observation(out,task_id,method,index):
    protocol=read(out/'protocol.json'); task=case(protocol,task_id)
    baseline=read(out/'evaluations/calibration_0.json')
    history=[compact(read(out/f'evaluations/{identifier(task_id,method,i)}.json')) for i in range(index)]
    data=dict(schema='pulsemorph.agentic_multifeature_observation.v1',observation_id=identifier(task_id,method,index),
        task_id=task_id,method=method,trial_index=index,protocol_sha256=sha256(out/'protocol.json'),
        user_instruction=task['instruction'],canonical_goal=task['goal'],
        goal_source='The same authored canonical goal is given to every numerical method after separate language interpretation scoring.',
        feature_names=protocol['feature_names'],feature_tolerances=protocol['feature_tolerances'],
        music_observation=task['music_observation'],parameter_names=NAMES,bounds=protocol['bounds'],
        tool=dict(section_windows_s=protocol['section_windows_s'],measurement_windows_s=protocol['feature_windows_s'],
            gains='Independent per-side scale of joint displacement from initial source posture; sin² envelope in each section, identity outside. Gain is not equal to achieved RMS ratio.',
            clock='source_time=t+timing_shift_s*sin(pi*t/4)^2 for 0<t<4; identity outside; positive shift generally advances a source event. Joint velocities include the full analytic chain rule.',
            output='Shared validated MotionReference consumed by pretrained G1 ONNX/PD/MuJoCo; torque and realized pose remain separate.',
            scope='One offline four-second take, no persistent weights or live state branch.'),
        physical_gate=protocol['physical_gate'],selection=protocol['selection'],
        baseline=compact(baseline),prior_trial_results=history,
        trials_remaining=8-index,forced_first_parameters=protocol['anchor'] if index==0 else None,
        access='Only this supplied observation and your own previous supplied observations; do not inspect sibling outcomes or code.',
        response_schema={'parameters':[NAMES] if method=='llm_feedback' else ['eight ordered vectors, first exactly forced anchor'],
                         'brief_reason':'short decision summary; no private reasoning',
                         'only_observation_path':str(observation_path(out,task_id,method,index))},
        instructions='Propose one next vector using only these observations.' if method=='llm_feedback' else 'Commit all eight vectors now before candidate outcomes; no later proposal revision allowed.')
    path=observation_path(out,task_id,method,index);write_new_json(path,data)
    return path


def record_evaluation(out,candidate,task_id=None):
    protocol=read(out/'protocol.json');task=case(protocol,task_id or protocol['tasks'][0]['task_id'])
    trial=out/f'trials/{candidate}';proposal=read(out/f'proposals/{candidate}.json')
    result=evaluate_multifeature(trial,out/'trials/calibration_0',protocol,task['goal'])
    try:
        if sha256(trial/'proposal.json')!=sha256(out/f'proposals/{candidate}.json'):raise ValueError('executed proposal differs from committed proposal')
        if proposal['protocol_sha256']!=sha256(out/'protocol.json'):raise ValueError('proposal protocol mismatch')
        for key in ['observation_artifact','response_artifact']:
            item=proposal.get(key)
            if item and sha256(out/item['path'])!=item['sha256']:raise ValueError('proposal linked artifact changed')
    except (ValueError,OSError) as error:
        result.update(integrity_passed=False,physical_gate_passed=False,intent_passed=False)
        result['integrity_errors'].append(str(error))
    manifest=read(trial/'manifest.json')
    result.update(candidate_id=candidate,task_id=proposal['task_id'],method=proposal['method'],
        parameters=[proposal['parameters'][n] for n in NAMES],wall_seconds=manifest['wall_seconds'],
        executor_status=manifest['status'],control_steps=manifest['actual_control_steps'],physics_steps=manifest['actual_physics_steps'],
        evaluator_sha256=sha256(Path(__file__).with_name('evaluate_multifeature.py')),
        protocol_sha256=sha256(out/'protocol.json'),execution_manifest_sha256=sha256(trial/'manifest.json'))
    calls=read(trial/'policy_calls.json')
    online=[r['elapsed_ms'] for r in calls if r['role']=='control']
    result['online_onnx_ms_p50_p95_p99']=np.percentile(online,[50,95,99]).tolist() if online else None
    write_new_json(out/f'evaluations/{candidate}.json',result)
    def ref(path):
        p=out/path
        return dict(path=path,available=p.is_file(),sha256=sha256(p) if p.is_file() else None)
    record=dict(schema='pulsemorph.agentic_trial_record_draft.v2',candidate_id=candidate,
        task_id=proposal['task_id'],split='unassigned',quality='unassessed',group='same_upstream_output_dance_source',
        instruction=task['instruction'] if task_id else 'Unchanged shared calibration',
        proposal=ref(f'proposals/{candidate}.json'),observation=proposal.get('observation_artifact'),model_response=proposal.get('response_artifact'),
        reference=ref(f'trials/{candidate}/common_reference/manifest.json'),
        applied_pd_targets=ref(f'trials/{candidate}/applied_joint_targets.npy'),
        actual_qpos=ref(f'trials/{candidate}/qpos.npy'),actual_qvel=ref(f'trials/{candidate}/qvel.npy'),
        actual_policy_inputs=ref(f'trials/{candidate}/policy_inputs.npz'),
        execution_manifest=ref(f'trials/{candidate}/manifest.json'),evaluation=ref(f'evaluations/{candidate}.json'),
        audio_observation=task['music_observation'] if task_id else None,
        observed_physics_alignment=manifest.get('observation_alignment'),
        provider_tokens=None if proposal['method'].startswith('llm') else 0,
        provider_billed_cost=None if proposal['method'].startswith('llm') else 0,
        human_preference=None,shared_corpus_imported=False,model_reason_is_measurement=False)
    write_new_json(out/f'trial_records/{candidate}.json',record)
    print(json.dumps(compact(result),allow_nan=False),flush=True)
    return result


def prepare(out):
    protocol=read(out/'protocol.json')
    response=read(out/'responses/semantic_interpretation.json')
    results={r['task_id']:r for r in response['results']}
    checks=[]
    for task in protocol['tasks']:
        parsed=results[task['task_id']];goal=task['goal']
        checks.append(dict(task_id=task['task_id'],parsed=parsed,canonical=goal,
            feature_targets_match=bool(np.allclose(parsed['feature_targets'],goal['feature_targets'],atol=1e-9,rtol=0)),
            cue_event_matches=parsed['cue_event_id']==goal['cue_event_id']))
    write_new_json(out/'semantic_evaluation.json',dict(checks=checks,response_sha256=sha256(out/'responses/semantic_interpretation.json'),
        observation_sha256=sha256(out/'observations/semantic_tasks.json'),scope='Two explicit-glossary development instructions; not unconstrained semantic generalization.'))
    record_evaluation(out,'calibration_0')
    for task in protocol['tasks']:
        for method in ['llm_feedback','llm_feedback_off']:observation(out,task['task_id'],method,0)


def commit(out,task_id,method,index,values,obs=None,response=None,reason=''):
    protocol=read(out/'protocol.json');case(protocol,task_id)
    if method not in METHODS or type(index)is not int or not 0<=index<8:raise ValueError('method/index outside budget')
    values=validate_parameters(values,protocol)
    if index==0 and values!=protocol['anchor']:raise ValueError('first candidate must equal common anchor')
    if index and not (out/f'proposals/{identifier(task_id,method,index-1)}.json').exists():raise ValueError('prior commitment missing')
    if method=='llm_feedback_off' and any(p.name!='calibration_0' for p in (out/'trials').iterdir()):
        raise ValueError('all feedback-off proposals must precede any candidate execution')
    if method.startswith('llm'):
        if not obs or not response:raise ValueError('LLM observation and recorded response required')
        observed=read(obs);raw=model_response(response)
        if Path(raw['only_observation_path']).resolve()!=Path(obs).resolve() or observed['task_id']!=task_id or observed['method']!=method:
            raise ValueError('response/observation identity mismatch')
        revision=index if method=='llm_feedback' else 0
        if (observed.get('protocol_sha256')!=sha256(out/'protocol.json') or
                observed.get('observation_id')!=identifier(task_id,method,revision) or
                observed.get('trial_index')!=revision or
                observed.get('canonical_goal')!=case(protocol,task_id)['goal']):
            raise ValueError('stale protocol, goal or observation revision')
        if len(observed.get('prior_trial_results',[]))!=revision:
            raise ValueError('observation history revision mismatch')
        if any(r['candidate_id']!=identifier(task_id,method,i) for i,r in enumerate(observed.get('prior_trial_results',[]))):
            raise ValueError('observation includes another decision stream')
        expected=raw['parameters'][index] if method=='llm_feedback_off' else raw['parameters'][0]
        if expected!=values or (method=='llm_feedback' and observed['trial_index']!=index):raise ValueError('response or revision mismatch')
        if len(raw['parameters'])!=(8 if method=='llm_feedback_off' else 1):raise ValueError('wrong response candidate count')
    candidate=identifier(task_id,method,index)
    def item(path):return {'path':str(Path(path).relative_to(out)),'sha256':sha256(Path(path))} if path else None
    proposal=dict(candidate_id=candidate,task_id=task_id,method=method,trial_index=index,
        parameters=dict(zip(NAMES,values)),duration_s=4.,source_start_s=0.,section_windows_s=protocol['section_windows_s'],
        audio={'duration_s':4.,'observed_through_sample':88200},
        protocol_sha256=sha256(out/'protocol.json'),created_at_utc=stamp(),observation_id=read(obs)['observation_id'] if obs else candidate,
        observation_artifact=item(obs),response_artifact=item(response),brief_reason=reason)
    if response:proposal['response_parameter_layout_normalization']=model_response(response)['parameter_layout_normalization']
    write_new_json(out/f'proposals/{candidate}.json',proposal)
    return proposal


def commit_off(out):
    protocol=read(out/'protocol.json')
    for task in protocol['tasks']:
        task_id=task['task_id'];method='llm_feedback_off';rpath=response_path(out,task_id,method,0);data=read(rpath)
        # Validate the entire response before creating any commitment.
        if len(data['parameters'])!=8:raise ValueError('eight precommitted proposals required')
        for values in data['parameters']:validate_parameters(values,protocol)
        for i,values in enumerate(data['parameters']):
            commit(out,task_id,method,i,values,observation_path(out,task_id,method,0),rpath,data['brief_reason'])


def run(out,task_id,method,index):
    protocol=read(out/'protocol.json');task=case(protocol,task_id)
    if not all((out/f'proposals/{identifier(t["task_id"],"llm_feedback_off",i)}.json').exists() for t in protocol['tasks'] for i in range(8)):
        raise ValueError('both feedback-off streams must be committed before any candidates')
    candidate=identifier(task_id,method,index)
    if method=='jacobian_search':
        previous=[read(out/f'evaluations/{identifier(task_id,method,i)}.json') for i in range(index)]
        started=time.perf_counter();values=next_jacobian_candidate(previous,protocol,task['goal']);elapsed=time.perf_counter()-started
        obs=observation_path(out,task_id,method,index)
        write_new_json(obs,dict(observation_id=candidate,method=method,task_id=task_id,canonical_goal=task['goal'],prior_trial_results=[compact(r) for r in previous],proposal_compute_seconds=elapsed))
        commit(out,task_id,method,index,values,obs,reason='Eight-trial deterministic Jacobian/Broyden search from only own observed features.')
    elif method=='llm_feedback':
        rpath=response_path(out,task_id,method,index);data=model_response(rpath)
        commit(out,task_id,method,index,data['parameters'][0],observation_path(out,task_id,method,index),rpath,data['brief_reason'])
    elif method!='llm_feedback_off':raise ValueError('unknown method')
    manifest=execute_trial(out/f'proposals/{candidate}.json',out/f'trials/{candidate}')
    result=record_evaluation(out,candidate,task_id)
    if method=='llm_feedback' and index<7:observation(out,task_id,method,index+1)
    if manifest['status']!='success' or not result['integrity_passed']:
        raise RuntimeError('failed execution/integrity retained as a consumed slot: '+candidate)
    return result


def summarize(out):
    protocol=read(out/'protocol.json');groups=[];allresults=[]
    for task in protocol['tasks']:
        methods=[]
        for method in METHODS:
            rows=[read(out/f'evaluations/{identifier(task["task_id"],method,i)}.json') for i in range(8)]
            eligible=[r for r in rows if r['integrity_passed'] and r['physical_gate_passed'] and r['utility'] is not None]
            selected=min(eligible,key=lambda r:(r['utility'],r['utility_l2'],r['candidate_id'])) if eligible else None
            methods.append(dict(method=method,trials=rows,selected=selected,physical_trials=len(rows),
                control_steps=sum(r['control_steps'] for r in rows),physics_steps=sum(r['physics_steps'] for r in rows),
                trial_wall_seconds=sum(r['wall_seconds'] for r in rows),model_proposal_calls=8 if method=='llm_feedback' else 1 if method=='llm_feedback_off' else 0,
                provider_tokens=None if method.startswith('llm') else 0,provider_billed_cost=None if method.startswith('llm') else 0))
            allresults.extend(rows)
        groups.append(dict(task_id=task['task_id'],instruction=task['instruction'],goal=task['goal'],methods=methods))
    before=read(out/'protected_hashes_before.json');frozen={p:sha256(ROOT/p)==h for p,h in before.items()}
    baseline=read(out/'evaluations/calibration_0.json')
    summary=dict(schema='pulsemorph.agentic_multifeature_comparison.v1',protocol_sha256=sha256(out/'protocol.json'),
        tasks=groups,calibration=baseline,semantic_evaluation=read(out/'semantic_evaluation.json'),
        comparison_valid=baseline['integrity_passed'] and all(r['integrity_passed'] for r in allresults),
        physical_trials=1+len(allresults),control_steps=baseline['control_steps']+sum(r['control_steps'] for r in allresults),
        physics_steps=baseline['physics_steps']+sum(r['physics_steps'] for r in allresults),protected_files_unchanged=frozen,
        planned_budget={'physical_trials':49,'control_steps':9800,'physics_steps':196000},
        limitations=protocol['limitations']+['Two explicit-glossary tasks, no held-out semantic or musical generalization.','Reported proposal-call counts exclude separate semantic and engineering model work.','LLM tokens, provider billing and isolated model latency unavailable.'])
    write_new_json(out/'comparison.json',summary)
    print(json.dumps({'comparison_valid':summary['comparison_valid'],'physical_trials':summary['physical_trials'],
        'selected':[{'task':t['task_id'],'method':m['method'],'candidate':m['selected']['candidate_id'] if m['selected'] else None,
                     'features':m['selected']['features'] if m['selected'] else None,'passed':m['selected']['intent_passed'] if m['selected'] else False} for t in groups for m in t['methods']]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True,type=Path);p.add_argument('--action',required=True,choices=['prepare','commit-off','run','summarize'])
    p.add_argument('--task');p.add_argument('--method');p.add_argument('--index',type=int);a=p.parse_args();out=a.pilot.resolve()
    if a.action=='prepare':prepare(out)
    elif a.action=='commit-off':commit_off(out)
    elif a.action=='run':run(out,a.task,a.method,a.index)
    else:summarize(out)
