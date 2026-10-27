"""Immutable warm-start repair workflow; subprocesses may run separate streams.

Every state trial has independent start/complete event files. All OFF responses
precede candidate physics; all six winners precede confirmation physics.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from musical_tasks import ROOT, read, sha256 as _sha256, write_new_json
from repair_plan import METHODS,TASKS,DEV,initialize,link,feature_row
from execute_repair import execute_trial, validate_parameters
from evaluate_repair import evaluate_repair,aggregate_conditions,selection_key,bridge_pins
from evaluate_generalization import _audit
from run_tool_compatibility_smoke import trace_parity
from tool_selection_driver_v2 import tool_descriptions
from repair_search import next_candidate


def require(value,message):
    if not value:raise ValueError(message)


def sha256(path):return _sha256(Path(path))


def verify_link(record):
    require(set(record)=={'path','sha256'} and sha256(record['path'])==record['sha256'],'artifact link changed')


def event(out,kind,identity,**fields):
    write_new_json(out/'events'/(kind+'__'+identity+'.json'),dict(event=kind,identity=identity,
        time_utc=datetime.now(timezone.utc).isoformat(),**fields))


def context(out):
    require(not (out/'abort.json').exists(),'panel is aborted')
    p=read(out/'protocol.json')
    for name,digest in p['execution_code_sha256'].items():require(sha256(ROOT/name)==digest,'frozen source changed: '+name)
    for item in p['shared_priors'].values():require(sha256(item['path'])==item['sha256'],'shared prior changed')
    if (out/'gate.json').exists():
        gate=read(out/'gate.json');require(gate['protocol_sha256']==sha256(out/'protocol.json'),'gate/protocol identity')
    return p


def task_for(p,task):return next(t for t in p['tasks'] if t['task_id']==task)
def identifier(task,method,index):
    require(task in TASKS and method in METHODS and type(index) is int and 0<=index<8,'invalid stream/slot')
    return f'{task}__{method}__{index}'


def history(out,task,method,index,results=True):
    proposals=[]; measured=[]
    for i in range(index):
        identity=identifier(task,method,i);row=read(out/'proposals'/(identity+'.json'))
        require((row['task_id'],row['method'],row['trial_index'])==(task,method,i),'cross-stream history')
        require(row['protocol_sha256']==sha256(out/'protocol.json'),'history protocol changed')
        for field in ('observation','response'):verify_link(row[field])
        proposals.append(dict(candidate_id=identity,control_tool=row['control_tool'],parameters=row['parameters']))
        if results:
            result=read(out/'candidates'/(identity+'.json'))
            require(result['candidate_id']==identity and result['task_id']==task and result['method']==method,'cross-stream result')
            verify_link(result['provenance']['proposal'])
            for record in result['condition_records'].values():verify_link(record['evaluation'])
            measured.append(result)
    return proposals,measured


def observation_data(out,task,method,index):
    p=context(out);require(read(out/'gate.json')['passed'] is True,'four-case gate required')
    t=task_for(p,task);proposed,measured=history(out,task,method,index,method!='llm_feedback_off')
    prior=read(p['shared_priors'][task]['path'])
    return dict(schema='pulsemorph.agentic_repair_observation.v1',candidate_id=identifier(task,method,index),
        task_id=task,method=method,trial_index=index,protocol_sha256=sha256(out/'protocol.json'),
        user_instruction=t['instruction'],canonical_goal=t['goal'],music_observation=t['music_observation'],
        feature_names=p['feature_names'],feature_tolerances=p['feature_tolerances'],physical_gate=p['physical_gate'],
        parameter_names=p['parameter_names'],bounds=p['bounds'],tools=tool_descriptions(p),
        development_condition_ids=p['development_condition_ids'],scenarios=p['scenarios'],selection=p['selection'],
        shared_prior=prior,prior_proposals=proposed,prior_trial_results=measured,
        response_schema=dict(control_tool=p['control_tools'],parameters='flat list of5 finite numbers',
            only_observation_path=str((out/'observations'/(identifier(task,method,index)+'.json')).resolve()),brief_reason='short decision summary'),
        access='Only this observation and arithmetic on its contents; no sibling files, internet, saved code or extra physics. OFF has no NEW candidate feedback. One vector per turn.',
        scope='Known warm-start development; fixed nominal denominator and frozen dominant-peak goal. Prior model-generated parameters shared equally.')


def observe(out,task,method,index):
    path=out/'observations'/(identifier(task,method,index)+'.json')
    write_new_json(path,observation_data(out,task,method,index));return path


def commit(out,task,method,index):
    p=context(out);identity=identifier(task,method,index);obs=out/'observations'/(identity+'.json')
    require(read(obs)==observation_data(out,task,method,index),'stale/cross-stream observation')
    if method=='llm_feedback_off':require(not list((out/'events').glob('trial_start__candidate__*.json')),'OFF must precede candidate physics')
    response=out/'responses'/(identity+'.json');raw=read(response)
    require(set(raw)=={'control_tool','parameters','only_observation_path','brief_reason'},'exact response schema')
    require(raw['control_tool'] in p['control_tools'] and isinstance(raw['brief_reason'],str),'response tool/reason')
    require(Path(raw['only_observation_path']).resolve()==obs.resolve(),'response observation path')
    require(isinstance(raw['parameters'],list) and len(raw['parameters'])==5,'five response parameters')
    validate_parameters(dict(zip(p['parameter_names'],raw['parameters'])))
    if method.startswith('llm'):
        call=read(out/'model_calls'/(identity+'.json'))
        require(call['model']=='gpt-6.1-sol' and call['fork_turns']=='none' and call['candidate_id']==identity,'model call receipt identity')
        require(call['observation']==link(obs) and call['response']==link(response),'model call artifact receipt')
    proposal=dict(candidate_id=identity,task_id=task,method=method,trial_index=index,
        control_tool=raw['control_tool'],parameters=raw['parameters'],observation=link(obs),response=link(response),
        protocol_sha256=sha256(out/'protocol.json'),brief_reason=raw['brief_reason'])
    if method=='surrogate_search':proposal['search_diagnostics']=read(out/'search_decisions'/(identity+'.json'))['search_diagnostics']
    write_new_json(out/'proposals'/(identity+'.json'),proposal)
    event(out,'proposal_committed',identity,proposal=link(out/'proposals'/(identity+'.json')))
    return proposal


def precommit_off(out):
    for task in TASKS:
        for i in range(8):
            identity=identifier(task,'llm_feedback_off',i);proposal=read(out/'proposals'/(identity+'.json'))
            for name in ('observation','response'):
                require(sha256(proposal[name]['path'])==proposal[name]['sha256'],'OFF commitment changed')
            committed=read(out/'events'/('proposal_committed__'+identity+'.json'))
            require(committed['proposal']==link(out/'proposals'/(identity+'.json')),'OFF proposal commitment changed')
            receipt=read(out/'model_calls'/(identity+'.json'))
            require(receipt['candidate_id']==identity and receipt['model']=='gpt-6.1-sol' and receipt['fork_turns']=='none'
                and receipt['observation']==proposal['observation'] and receipt['response']==proposal['response'],'OFF call receipt changed')
    return True


def trial_proposal(out,p,candidate,scenario,identity,seed=None):
    seed=seed or p['seed_proposals'][candidate['task_id']]
    path=Path(seed['path']);path=path if path.is_absolute() else ROOT/path
    require(sha256(path)==seed['sha256'],'seed proposal changed')
    raw=read(path);names=p['parameter_names'];raw=copy.deepcopy(raw)
    raw.update(candidate_id=identity,method=candidate['method'],task_id=candidate['task_id'],
        control_tool=candidate['control_tool'],parameters=dict(zip(names,candidate['parameters'])),
        execution_kind='robust_initial_state_repair',parameter_source=seed,initial_perturbation=scenario['initial_perturbation'],
        protocol_sha256=sha256(out/'protocol.json'),observation_id=identity+':observation',
        runtime_session_id=sha256(out/'protocol.json')+':'+identity,observation_artifact=None,response_artifact=None,
        repair_candidate_id=candidate['candidate_id'])
    write_new_json(out/'trial_proposals'/(identity+'.json'),raw)
    return out/'trial_proposals'/(identity+'.json')


def run_trial(out,p,candidate,condition,phase,seed=None,identity=None,score=True):
    identity=identity or phase+'__'+candidate['candidate_id']+'__'+condition
    scenario=next(s for s in p['scenarios'] if s['scenario_id']==condition)
    proposal=trial_proposal(out,p,candidate,scenario,identity,seed)
    event(out,'trial_start',identity,phase=phase,task_id=candidate['task_id'],method=candidate['method'],
        candidate_id=candidate['candidate_id'],scenario_id=condition,proposal=link(proposal))
    trial=out/'trials'/identity;manifest=execute_trial(proposal,trial)
    event(out,'trial_complete',identity,phase=phase,manifest=link(trial/'manifest.json'),status=manifest['status'],
        control_steps=manifest.get('actual_control_steps',0),physics_steps=manifest.get('actual_physics_steps',0))
    require(manifest['status']=='success',identity+': '+str(manifest.get('error')))
    require(manifest['strict_g1_admission_passed'] and manifest['strict_g1_preflight_count']==2,'strict admission')
    require((manifest['actual_control_steps'],manifest['actual_physics_steps'])==(200,4000),'trial counts')
    result=None
    if score:
        task=task_for(p,candidate['task_id']);base=Path(p['baseline_panel'])/'trials'
        result=evaluate_repair(trial,base/('nominal__calibration_'+task['source_id']),
            base/(condition+'__calibration_'+task['source_id']),p,task['goal'])
        write_new_json(out/'evaluations'/(identity+'.json'),result)
        require(result['integrity_passed'],str(result['integrity_errors']))
    print(json.dumps(dict(trial=identity,phase=phase,status='audited')),flush=True)
    return identity,result


def gate(out):
    p=context(out);base=Path(p['baseline_panel']);cases=[]
    for role in ('calibration_wave','calibration_high5','selected_wave_a','selected_high5_b'):
        old=base/'trials'/('nominal__'+role);raw=read(old/'proposal.json')
        seed=dict(path=(old/'proposal.json').relative_to(ROOT).as_posix(),sha256=sha256(old/'proposal.json'))
        candidate=dict(candidate_id='gate_'+role,task_id=raw['task_id'],method='surrogate_search',control_tool=raw['control_tool'],
            parameters=[raw['parameters'][n] for n in p['parameter_names']])
        identity,_=run_trial(out,p,candidate,'nominal','gate',seed,identity='gate__'+role,score=False)
        new=out/'trials'/identity;audit=_audit(new)
        bridge_pins(audit['manifest']['preexecution_input_hashes'],read(old/'manifest.json')['preexecution_input_hashes'],p['repair_wrapper_pin_switch'])
        require(audit['mean_joint_rmse_rad']<=p['physical_gate']['max_mean_joint_rmse_rad'] and float(audit['sub'][:,2].min())>=p['physical_gate']['min_root_height_m'],'gate physical failure')
        cases.append(dict(role=role,trial_id=identity,old_trial=str(old),parity=trace_parity(old,new),manifest=link(new/'manifest.json')))
    write_new_json(out/'gate.json',dict(passed=True,cases=cases,protocol_sha256=sha256(out/'protocol.json')))
    event(out,'gate_passed','all4',gate=link(out/'gate.json'))


def run_candidate(out,task,method,index):
    p=context(out);precommit_off(out);require(read(out/'gate.json')['passed'],'gate required')
    identity=identifier(task,method,index);_,measured=history(out,task,method,index,True)
    if method=='surrogate_search':
        obs=observe(out,task,method,index);prior=read(p['shared_priors'][task]['path'])
        decision=next_candidate(measured,p,task_for(p,task)['goal'],prior['observations'])
        write_new_json(out/'search_decisions'/(identity+'.json'),decision)
        write_new_json(out/'responses'/(identity+'.json'),dict(control_tool=decision['control_tool'],parameters=decision['parameters'],
            only_observation_path=str(obs.resolve()),brief_reason='Deterministic weighted affine minimax repair using shared prior and own measured history.'))
        commit(out,task,method,index)
    proposal=read(out/'proposals'/(identity+'.json'));results={};links={}
    require((proposal['candidate_id'],proposal['task_id'],proposal['method'],proposal['trial_index'])==(identity,task,method,index),'candidate stream identity')
    require(proposal['protocol_sha256']==sha256(out/'protocol.json'),'candidate protocol identity')
    for name in ('observation','response'):verify_link(proposal[name])
    for condition in p['development_condition_ids']:
        trial_id,result=run_trial(out,p,proposal,condition,'candidate')
        results[condition]=result;links[condition]=dict(trial_id=trial_id,evaluation=link(out/'evaluations'/(trial_id+'.json')))
    row=dict(candidate_id=identity,task_id=task,method=method,control_tool=proposal['control_tool'],parameters=proposal['parameters'],
        state_evaluations=[feature_row(results[s],s) for s in p['development_condition_ids']],
        aggregate=aggregate_conditions(results,p['development_condition_ids']),condition_records=links,
        provenance=dict(proposal=link(out/'proposals'/(identity+'.json'))),search_diagnostics=proposal.get('search_diagnostics'))
    write_new_json(out/'candidates'/(identity+'.json'),row);return row


def incumbent(out,p,task):
    prior=read(p['shared_priors'][task]['path']);base=Path(p['baseline_panel'])
    row=copy.deepcopy(next(r for r in prior['observations'] if r['candidate_id']=='incumbent'))
    results={s:read(base/'evaluations'/(s+'__selected_'+task+'.json')) for s in p['development_condition_ids']}
    row.update(task_id=task,aggregate=aggregate_conditions(results,p['development_condition_ids']),
        condition_records={s:dict(trial_id=s+'__selected_'+task,evaluation=link(base/'evaluations'/(s+'__selected_'+task+'.json')),cached_prior=True) for s in results})
    return row


def select(out):
    p=context(out);winners=[]
    for task in TASKS:
        for method in METHODS:
            rows=[incumbent(out,p,task)]+[read(out/'candidates'/(identifier(task,method,i)+'.json')) for i in range(8)]
            for row in rows:
                for record in row['condition_records'].values():verify_link(record['evaluation'])
            winner=copy.deepcopy(min(rows,key=lambda r:selection_key(r['aggregate'],r['candidate_id'])))
            winner.update(method=method,task_id=task)
            winners.append(winner)
    write_new_json(out/'selection.json',dict(winners=winners,protocol_sha256=sha256(out/'protocol.json'),confirmation_used_for_selection=False))
    event(out,'selection_frozen','all6',selection=link(out/'selection.json'))


def confirm(out):
    p=context(out);selected=read(out/'selection.json');confirmed=[]
    require(len(selected['winners'])==6 and selected['confirmation_used_for_selection'] is False,'all6 frozen winners required')
    require({(w['task_id'],w['method']) for w in selected['winners']}=={(t,m) for t in TASKS for m in METHODS},'exact six distinct task-method winners')
    require(selected['protocol_sha256']==sha256(out/'protocol.json'),'selection protocol identity')
    barrier=read(out/'events'/'selection_frozen__all6.json')
    require(barrier['event']=='selection_frozen' and barrier['selection']==link(out/'selection.json'),'selection freeze receipt')
    for winner in selected['winners']:
        for record in winner['condition_records'].values():verify_link(record['evaluation'])
        results={s:read(r['evaluation']['path']) for s,r in winner['condition_records'].items()}
        references=copy.deepcopy(winner['condition_records'])
        for condition in p['confirmation_condition_ids']:
            identity='confirmation__'+winner['task_id']+'__'+winner['method']+'__'+condition
            trial_id,result=run_trial(out,p,winner,condition,'confirmation',identity=identity)
            results[condition]=result;references[condition]=dict(trial_id=trial_id,evaluation=link(out/'evaluations'/(trial_id+'.json')),fresh_confirmation=True)
        confirmed.append(dict(task_id=winner['task_id'],method=winner['method'],selected_candidate_id=winner['candidate_id'],
            control_tool=winner['control_tool'],parameters=winner['parameters'],development_aggregate=winner['aggregate'],
            full_aggregate=aggregate_conditions(results,[s['scenario_id'] for s in p['scenarios']]),
            primary_passed=sum(r['primary']['joint_success'] for r in results.values()),condition_records=references))
    manifests=[read(x) for x in (out/'trials').glob('*/manifest.json')]
    require(len(manifests)==250,'exact250 trials required')
    summary=dict(schema='pulsemorph.agentic_repair_summary.v1',completed=True,errors=[],winners=confirmed,
        protocol_sha256=sha256(out/'protocol.json'),physical_trials=len(manifests),control_steps=sum(m['actual_control_steps'] for m in manifests),
        physics_steps=sum(m['actual_physics_steps'] for m in manifests),model_proposal_turns=len(list((out/'model_calls').glob('*.json'))),
        training_admission=False,confirmation_is_known_development=True)
    require(summary['model_proposal_turns']==32,'exact32 model receipts required')
    write_new_json(out/'summary.json',summary);event(out,'panel_complete','all',summary=link(out/'summary.json'))


def abort(out,stage,error):
    path=out/'abort.json'
    if not path.exists():write_new_json(path,dict(stage=stage,error=str(error),time_utc=datetime.now(timezone.utc).isoformat(),retry_allowed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--action',choices=['initialize','gate','observe','commit','run','batch-baselines','select','confirm'],required=True)
    parser.add_argument('--task');parser.add_argument('--method');parser.add_argument('--index',type=int)
    args=parser.parse_args();out=args.output.resolve()
    try:
        if args.action=='initialize':initialize(out)
        elif args.action=='gate':gate(out)
        elif args.action=='observe':print(observe(out,args.task,args.method,args.index))
        elif args.action=='commit':commit(out,args.task,args.method,args.index)
        elif args.action=='run':run_candidate(out,args.task,args.method,args.index)
        elif args.action=='select':select(out)
        elif args.action=='confirm':confirm(out)
        else:
            for task in TASKS:
                for method in ('surrogate_search','llm_feedback_off'):
                    for index in range(8):run_candidate(out,task,method,index)
    except Exception as error:
        if out.exists():abort(out,args.action,error)
        traceback.print_exc();raise SystemExit(1)
