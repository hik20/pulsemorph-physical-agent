"""Independent read-only repair acceptance helpers; no model or physics execution.

End-to-end orchestration is bound to the frozen driver schema separately.
Timestamp checks are local provenance checks, not cryptographic attestation.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

METHODS = ('surrogate_search', 'llm_feedback', 'llm_feedback_off')
TASKS = ('wave_a', 'high5_b')
DEVELOPMENT = ('nominal', 'joint_position_minus', 'combined_plus', 'combined_minus')
CONFIRMATION = ('joint_position_plus', 'joint_velocity_plus', 'joint_velocity_minus',
                'root_vx_plus', 'root_vx_minus', 'root_vy_plus', 'root_vy_minus',
                'root_yaw_plus', 'root_yaw_minus')


def require(value, message):
    if not value:
        raise ValueError(message)


def independent_aggregate(evaluations, condition_ids=DEVELOPMENT):
    """Derive selector inputs from signed residuals, never copied utilities."""
    require(set(evaluations) == set(condition_ids), 'exact condition inventory required')
    physical, residuals, condition_squares, available = True, [], [], True
    for condition in condition_ids:
        row = evaluations[condition]
        require(row.get('integrity_passed') is True, 'integrity failure is protocol error')
        require(type(row.get('physical_gate_passed')) is bool, 'explicit physical gate required')
        physical &= row['physical_gate_passed']
        score = row.get('primary')
        require(isinstance(score, dict), 'primary score required')
        errors, tolerances = score['feature_errors'], score['feature_tolerances']
        require(len(errors) == len(tolerances) == 5, 'five features required')
        local_residuals=[]
        for error, tolerance in zip(errors, tolerances):
            require(type(tolerance) in (int, float) and math.isfinite(tolerance) and tolerance > 0,
                    'positive finite tolerance required')
            if error is None:
                available = False
            else:
                require(type(error) in (int, float) and math.isfinite(error), 'finite signed error required')
                residuals.append(error / tolerance)
                local_residuals.append(error / tolerance)
        condition_squares.append(float(np.sum(np.asarray(local_residuals)**2)))
    return dict(integrity_passed=True, physical_gate_passed=physical,
                utility=max(map(abs, residuals)) if available else None,
                utility_l2=sum(condition_squares) if available else None)


def audit_selection(candidates, winner_id, condition_ids=DEVELOPMENT):
    """Candidates maps stable IDs to complete measured condition dictionaries."""
    require(isinstance(candidates, dict) and len(candidates) == 9,
            'incumbent plus eight committed candidates required')
    require(winner_id in candidates, 'winner absent from candidate inventory')
    scores = {key: independent_aggregate(value, condition_ids) for key, value in candidates.items()}
    def key(identity):
        score = scores[identity]
        return (not score['physical_gate_passed'],
                math.inf if score['utility'] is None else score['utility'],
                math.inf if score['utility_l2'] is None else score['utility_l2'], identity)
    selected = min(candidates, key=key)
    require(selected == winner_id, 'winner differs from independent physical/worst/L2/ID selector')
    return dict(winner_id=selected, independent_selector_verified=True, aggregate=scores[selected])


def check_budget(ledger):
    """Each record is a newly executed physical trial, never a reused evaluation."""
    require(len(ledger) == 250, 'exact250 fresh physical trials required')
    require(len({row['trial_id'] for row in ledger}) == 250, 'duplicate trial ID')
    expected = {'runtime_gate': 4, 'candidate': 192, 'confirmation': 54}
    for phase, count in expected.items():
        require(sum(row['phase'] == phase for row in ledger) == count, 'incorrect '+phase+' count')
    require(all(row['phase'] in expected and row.get('fresh_physics') is True for row in ledger),
            'unknown phase or cached row counted as fresh physics')
    require(all(type(row['control_steps']) is int and row['control_steps'] == 200
                and type(row['physics_steps']) is int and row['physics_steps'] == 4000
                for row in ledger), 'per-trial200/4000 execution count')
    for task in TASKS:
        for method in METHODS:
            rows = [r for r in ledger if r['phase']=='candidate' and r['task_id']==task and r['method']==method]
            require(len(rows)==32, 'method/task candidate budget differs')
            pairs = {(r['proposal_index'], r['condition_id']) for r in rows}
            require(pairs == {(i,c) for i in range(8) for c in DEVELOPMENT},
                    'eight complete candidate condition batches required')
            confirm = [r for r in ledger if r['phase']=='confirmation' and r['task_id']==task and r['method']==method]
            require(len(confirm)==9 and len({r['condition_id'] for r in confirm})==9,
                    'nine unique confirmation conditions required')
            require(set(r['condition_id'] for r in confirm)==set(CONFIRMATION),
                    'exact nine remaining known confirmation conditions required')
    return dict(new_physical_trials=250, control_steps=50000, physics_steps=1000000)


def check_temporal_isolation(off_responses, candidate_starts, winner_freezes, confirmation_starts):
    """All values are integer filesystem st_mtime_ns from preserved artifacts."""
    require(len(off_responses)==16 and len(candidate_starts)==192
            and len(winner_freezes)==6 and len(confirmation_starts)==54, 'chronology inventory incomplete')
    values = [*off_responses,*candidate_starts,*winner_freezes,*confirmation_starts]
    require(all(type(value) is int and value>0 for value in values), 'nanosecond chronology required')
    require(max(off_responses) < min(candidate_starts), 'OFF proposal committed after candidate physics began')
    require(max(winner_freezes) < min(confirmation_starts), 'confirmation began before all six winners froze')
    require(max(candidate_starts) < min(winner_freezes), 'winner froze before candidate executions finished starting')
    return dict(off_commit_barrier_verified=True, all_winner_confirmation_barrier_verified=True,
                timestamps_are_local_unsigned_evidence=True)


def check_hash_inventory(root, before):
    """Verify every protected artifact; caller captures full inventory at freeze."""
    root = Path(root).resolve()
    require(isinstance(before, dict) and before, 'protected inventory required')
    for relative, digest in before.items():
        path = (root / relative).resolve()
        require(root in path.parents and path.is_file(), 'protected path missing or outside root')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'protected evidence mutated: '+relative)
    return dict(verified_files=len(before), protected_evidence_unchanged=True)


def audit_plan(protocol):
    """Reject design drift before proposals or new physics are admitted."""
    require(protocol.get('schema')=='pulsemorph.agentic_robust_repair_protocol.v1','repair protocol schema')
    require(protocol.get('methods')==list(METHODS),'exact three methods')
    require([t['task_id'] for t in protocol.get('tasks',[])]==list(TASKS),'exact two development tasks')
    require(protocol.get('budget')==8,'exact eight proposals per arm/task')
    require(protocol.get('development_condition_ids')==list(DEVELOPMENT),'development panel changed')
    require(set(protocol.get('confirmation_condition_ids',[]))==set(CONFIRMATION)
            and len(protocol['confirmation_condition_ids'])==9,'remaining nine condition panel')
    scenarios=protocol.get('scenarios',[])
    require(len(scenarios)==13 and {s['scenario_id'] for s in scenarios}==set(DEVELOPMENT+CONFIRMATION),
            'full13 condition inventory')
    require(protocol.get('training_admission') is False,'repair is not training admission')
    require(protocol.get('planned_budget')==dict(gate_trials=4,candidate_trials=192,
        confirmation_trials=54,physical_trials=250,control_steps=50000,physics_steps=1000000,
        model_proposal_turns=32),'fixed budget mismatch')
    model=protocol.get('model',{})
    require(model.get('model')=='gpt-6.1-sol' and model.get('fork_turns')=='none'
            and model.get('turns_per_task_method')==8,'fixed proposer model protocol')
    require(set(protocol.get('shared_priors',{}))==set(TASKS),'equal common priors absent')
    from robustness_scenarios import build_scenarios
    expected=build_scenarios(scenarios[0]['joint_names'])
    require(scenarios==expected,'scenario numeric definitions changed')
    return dict(plan_admission_passed=True,new_physical_trials=0,new_llm_calls=0)


def audit_trial(trial, nominal_baseline, condition_baseline, protocol, goal, scenario,
                original_nominal, evaluation_path=None):
    """Raw trace audit plus independent injection, RMS and score reconstruction.

    The production evaluator is rerun but its recorded score is not sufficient:
    fixed-center RMS reconstruction independently checks every numeric field.
    """
    import evaluate_repair
    from evaluate_generalization import _audit
    from audit_robustness_v2 import check_initial_state, check_center_definitions
    audited = [_audit(path) for path in (trial, nominal_baseline, condition_baseline)]
    for baseline in audited[1:]:
        evaluate_repair.bridge_pins(audited[0]['manifest']['preexecution_input_hashes'],
            baseline['manifest']['preexecution_input_hashes'],protocol['repair_wrapper_pin_switch'])
    evaluation = evaluate_repair.evaluate_repair(trial,nominal_baseline,condition_baseline,protocol,goal)
    require(evaluation['integrity_passed'] is True,'repair evaluation integrity failed')
    if evaluation_path is not None:
        recorded=json.loads(Path(evaluation_path).read_text(encoding='utf8'))
        # Driver may add provenance fields to the immutable evaluation payload.
        require(all(recorded.get(key)==value for key,value in evaluation.items()),
                'recorded evaluation differs from raw trace recomputation')
    injection=check_initial_state(trial,original_nominal,scenario,audited[0])
    centers=check_center_definitions(evaluation,*(row['arrays'] for row in audited),
                                    audited[0]['names'],protocol,goal)
    return dict(evaluation=evaluation,initial_state=injection,centers=centers,
                new_physical_trials=0,new_llm_calls=0)


def audit(panel, output):
    """Independently accept the complete frozen250-trial development repair panel."""
    import sys
    import copy
    from datetime import datetime
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'src'))
    sys.path.insert(0,str(Path(__file__).parent))
    from audit_robustness_v2 import inventory, check_initial_state, check_center_definitions
    from evaluate_generalization import _audit
    from evaluate_repair import evaluate_repair, bridge_pins, aggregate_conditions
    from repair_search import next_candidate
    from audit_multifeature import common_parity
    from run_tool_compatibility_smoke import trace_parity
    from repair_plan import prior_data
    from repair_driver import observation_data
    panel,output=Path(panel).resolve(),Path(output).resolve()
    require(output!=panel and panel not in output.parents,'audit output must be outside input panel')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='pulsemorph.agentic_repair_audit.v1',audit_passed=False,errors=[],
        input_panel=str(panel),new_physical_trials=0,new_llm_calls=0,trials=[],streams=[])
    before=inventory(panel)
    def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def resolve(value):
        path=Path(value)
        return path if path.is_absolute() else root/path
    def pinned(item):
        require(isinstance(item,dict) and set(item)=={'path','sha256'},'exact linked artifact schema')
        path=resolve(item['path'])
        require(path.is_file() and sha(path)==item['sha256'],'linked artifact changed: '+str(path))
        return path
    def stamp(path):return Path(path).stat().st_mtime_ns
    try:
        require(not (panel/'abort.json').exists(),'aborted experiment cannot be accepted')
        protocol=read(panel/'protocol.json');audit_plan(protocol)
        report['input_protocol_sha256']=sha(panel/'protocol.json')
        for name,digest in protocol['execution_code_sha256'].items():
            require(sha(root/name)==digest and sha(panel/'source_snapshot'/name)==digest,'frozen code changed: '+name)
        for protected in protocol['protected_inventories']:
            require(inventory(resolve(protected['path']))==protected['files'],'old complete artifact inventory changed')
        original=read(pinned(protocol['original_protocol']))
        robust=read(pinned(protocol['robustness_protocol']))
        inherited_protected=robust['protected_inventories']
        for protected in inherited_protected:
            require(inventory(resolve(protected['path']))==protected['files'],'inherited old evidence inventory changed')
        for name in ('quarantine','quarantine_inventory'):
            pinned(robust[name])
        old_audit=read(pinned(protocol['accepted_prior_audit']))
        require(old_audit['audit_passed'] is True
            and old_audit['input_protocol_sha256']==sha(pinned(protocol['robustness_protocol'])),'prior audit not accepted/bound')
        base=Path(protocol['baseline_panel'])
        require(base/'protocol.json'==pinned(protocol['robustness_protocol']),'baseline panel not linked robustness panel')
        for task in protocol['tasks']:
            prior=read(pinned(protocol['shared_priors'][task['task_id']]))
            require(prior==prior_data(task,original),'shared prior differs from all original observed labels')
            require(len(prior['observations'])==27,'26 nominal records plus incumbent required')
        starts={};completes={}
        for path in (panel/'events').glob('*.json'):
            item=read(path)
            require(path.name==item['event']+'__'+item['identity']+'.json','event filename identity')
            if item['event']=='trial_start':
                require(item['identity'] not in starts,'duplicate start');starts[item['identity']]=(item,path)
            elif item['event']=='trial_complete':
                require(item['identity'] not in completes,'duplicate completion');completes[item['identity']]=(item,path)
        require(set(starts)==set(completes),'trial start/complete inventory differs')
        require(set(starts)=={p.parent.name for p in (panel/'trials').glob('*/manifest.json')},'trial/event inventory differs')
        scenarios={s['scenario_id']:s for s in protocol['scenarios']}
        tasks={t['task_id']:t for t in protocol['tasks']}
        raw_audits={};evaluations={};ledger=[];baseline_cache={}
        def baseline_audit(path):
            key=str(path)
            if key not in baseline_cache:baseline_cache[key]=_audit(path)
            return baseline_cache[key]
        for scenario in scenarios:
            for source in ('wave','high5'):
                baseline_audit(base/'trials'/(scenario+'__calibration_'+source))
        for identity in sorted(starts):
            start,startpath=starts[identity];complete,completepath=completes[identity]
            phase=start['phase'];trial=panel/'trials'/identity
            proposal=read(pinned(start['proposal']));manifest=read(pinned(complete['manifest']))
            require(pinned(start['proposal'])==(panel/'trial_proposals'/(identity+'.json')),'wrong trial proposal path')
            require((trial/'proposal.json').read_bytes()==pinned(start['proposal']).read_bytes(),'execution proposal mismatch')
            require(pinned(complete['manifest'])==trial/'manifest.json','wrong complete manifest path')
            require(manifest['candidate_id']==proposal['candidate_id']==identity,'trial candidate identity')
            require((trial/'source_proposal.json').read_bytes()==pinned(proposal['parameter_source']).read_bytes(),
                    'runtime seed snapshot differs from linked prior')
            require(manifest['status']==complete['status']=='success','technical failure in completed panel')
            require(complete['phase']==phase and stamp(startpath)<stamp(completepath),'trial event chronology')
            require(manifest['strict_g1_admission_passed'] is True and manifest['strict_g1_preflight_count']==2,'strict admission absent')
            require(proposal['protocol_sha256']==sha(panel/'protocol.json'),'trial protocol identity changed')
            condition=start['scenario_id'];taskid=start['task_id']
            source=proposal['source_motion']
            audit_raw=_audit(trial);raw_audits[identity]=audit_raw
            require(audit_raw['fresh_anchor_audit']['control_rows']==200,'all200 current-state FK required')
            old_nominal=base/'trials'/('nominal__calibration_'+('wave' if 'wave' in taskid else 'high5'))
            bridge_pins(manifest['preexecution_input_hashes'],read(old_nominal/'manifest.json')['preexecution_input_hashes'],protocol['repair_wrapper_pin_switch'])
            injected=check_initial_state(trial,old_nominal,scenarios[condition],audit_raw)
            common=common_parity(trial,proposal,output/'roundtrip_packages'/identity)
            row=dict(trial_id=identity,phase='runtime_gate' if phase=='gate' else phase,
                task_id=taskid,method=start['method'],condition_id=condition,fresh_physics=True,
                control_steps=manifest['actual_control_steps'],physics_steps=manifest['actual_physics_steps'])
            if phase=='candidate':row['proposal_index']=int(start['candidate_id'].rsplit('__',1)[1])
            ledger.append(row)
            if phase!='gate':
                require(taskid in tasks,'unknown scored task')
                task=tasks[taskid]
                require(proposal['parameter_source']==protocol['seed_proposals'][taskid],'trial uses wrong seed provenance')
                expected_source=task['source_motion']
                require(source==expected_source,'task source changed')
                nom=base/'trials'/('nominal__calibration_'+task['source_id'])
                cond=base/'trials'/(condition+'__calibration_'+task['source_id'])
                result=evaluate_repair(trial,nom,cond,protocol,task['goal'])
                require(result['integrity_passed'] is True,'evaluation integrity')
                require(read(panel/'evaluations'/(identity+'.json'))==result,'recorded evaluation mismatch')
                centers=check_center_definitions(result,audit_raw['arrays'],baseline_audit(nom)['arrays'],baseline_audit(cond)['arrays'],audit_raw['names'],protocol,task['goal'])
                evaluations[identity]=result
            else:
                role=identity.removeprefix('gate__');old=base/'trials'/('nominal__'+role)
                require(proposal['parameter_source']==dict(path=str(old/'proposal.json'),sha256=sha(old/'proposal.json')),
                        'gate seed differs from historical trace proposal')
                require(condition=='nominal','gate condition must be nominal')
                trace_parity(old,trial)
                require(audit_raw['mean_joint_rmse_rad']<=protocol['physical_gate']['max_mean_joint_rmse_rad']
                    and float(audit_raw['sub'][:,2].min())>=protocol['physical_gate']['min_root_height_m'],'gate physical failure')
            report['trials'].append(dict(trial_id=identity,initial_state=injected,common_all200=common))
        report['budget']=check_budget(ledger)
        gate=read(panel/'gate.json')
        expected_roles={'calibration_wave','calibration_high5','selected_wave_a','selected_high5_b'}
        require({identity for identity,(item,_) in starts.items() if item['phase']=='gate'}=={'gate__'+r for r in expected_roles},'runtime gate ID inventory')
        require(gate['passed'] is True and {r['role'] for r in gate['cases']}==expected_roles and len(gate['cases'])==4,'gate report incomplete')
        for record in gate['cases']:
            identity='gate__'+record['role'];old=base/'trials'/('nominal__'+record['role'])
            require(record['trial_id']==identity and Path(record['old_trial'])==old
                and pinned(record['manifest'])==panel/'trials'/identity/'manifest.json'
                and record['parity']==trace_parity(old,panel/'trials'/identity),'gate report parity mismatch')
        require(gate['protocol_sha256']==sha(panel/'protocol.json'),'gate protocol hash')
        gate_completed=panel/'events/gate_passed__all4.json';pinned(read(gate_completed)['gate'])
        require(max(stamp(path) for item,path in completes.values() if item['phase']=='gate')<stamp(gate_completed),'gate report precedes gate physics')
        receipts=list((panel/'model_calls').glob('*.json'))
        expected_calls={f'{t}__{m}__{i}' for t in TASKS for m in METHODS if m.startswith('llm') for i in range(8)}
        require({p.stem for p in receipts}==expected_calls,'exact32 model receipt identity inventory')
        require({p.stem for p in (panel/'model_requests').glob('*.json')}==expected_calls,'exact32 model request identity inventory')
        expected_candidates={f'{t}__{m}__{i}' for t in TASKS for m in METHODS for i in range(8)}
        for folder in ('proposals','observations','responses','candidates'):
            require({p.stem for p in (panel/folder).glob('*.json')}==expected_candidates,'exact48 '+folder+' required')
        candidate_maps={};off_commits=[]
        for taskid in TASKS:
            prior=read(pinned(protocol['shared_priors'][taskid]))
            for method in METHODS:
                measured=[];proposed=[];candidates={}
                for index in range(8):
                    identity=f'{taskid}__{method}__{index}'
                    proposalpath=panel/'proposals'/(identity+'.json');proposal=read(proposalpath)
                    require((proposal['candidate_id'],proposal['task_id'],proposal['method'],proposal['trial_index'])
                        ==(identity,taskid,method,index),'proposal stream identity changed')
                    obs_path=panel/'observations'/(identity+'.json');obs=read(obs_path)
                    require(obs==observation_data(panel,taskid,method,index),'observation data/schema drift')
                    require(obs['shared_prior']==prior and obs['prior_proposals']==proposed,'cross-stream prior/proposal leak')
                    require(obs['prior_trial_results']==([] if method=='llm_feedback_off' else measured),'new cross-stream or OFF feedback leak')
                    require(stamp(gate_completed)<stamp(obs_path),'observation before runtime admission')
                    require(pinned(proposal['observation'])==obs_path,'proposal observation provenance')
                    responsepath=pinned(proposal['response']);response=read(responsepath)
                    require(response['control_tool']==proposal['control_tool'] and response['parameters']==proposal['parameters'],'response vector changed')
                    require(Path(response['only_observation_path']).resolve()==obs_path,'response observation path changed')
                    commitpath=panel/'events'/('proposal_committed__'+identity+'.json')
                    require(pinned(read(commitpath)['proposal'])==proposalpath,'proposal commit link')
                    require(stamp(obs_path)<=stamp(responsepath)<=stamp(proposalpath)<=stamp(commitpath),'proposal artifact chronology')
                    if method.startswith('llm'):
                        callpath=panel/'model_calls'/(identity+'.json');receipt=read(callpath)
                        requestpath=panel/'model_requests'/(identity+'.json');request=read(requestpath)
                        require(receipt['candidate_id']==identity and receipt['model']=='gpt-6.1-sol' and receipt['fork_turns']=='none','model receipt mismatch')
                        require(pinned(receipt['observation'])==obs_path and pinned(receipt['response'])==responsepath,'model receipt links')
                        require(pinned(receipt['request'])==requestpath,'model request receipt link')
                        require(request['candidate_id']==identity and request['model']=='gpt-6.1-sol'
                            and request['fork_turns']=='none' and pinned(request['observation'])==obs_path,'model request identity/context')
                        require(Path(request['expected_response_path']).resolve()==responsepath
                            and isinstance(request['prompt'],str) and str(obs_path) in request['prompt']
                            and isinstance(request['isolation'],str),'model request prompt/response destination')
                        require(receipt['dispatch_requested_at_utc']==request['dispatch_requested_at_utc'],'dispatch receipt timestamp mismatch')
                        dispatch=datetime.fromisoformat(request['dispatch_requested_at_utc'])
                        completion=datetime.fromisoformat(receipt['completion_observed_at_utc'])
                        require(dispatch.utcoffset() is not None and completion.utcoffset() is not None
                            and dispatch<=completion,'UTC model dispatch/completion chronology')
                        require(stamp(obs_path)<=stamp(requestpath)<=stamp(responsepath)<=stamp(callpath)<=stamp(proposalpath),
                            'model request/response/call/proposal file chronology')
                        require(isinstance(receipt['agent_task'],str) and receipt['agent_task']
                            and receipt['status']=='response_observed','model completion provenance')
                        require(all(record.get(name) is None for record in (request,receipt)
                            for name in ('provider_tokens','provider_billed_cost')),'unmeasured provider usage presented as known')
                    else:
                        decision=next_candidate(measured,protocol,tasks[taskid]['goal'],prior['observations'])
                        require(read(panel/'search_decisions'/(identity+'.json'))==decision,'numeric decision replay mismatch')
                        require(decision['parameters']==proposal['parameters'] and decision['control_tool']==proposal['control_tool'],'numeric proposal mismatch')
                        require(proposal['search_diagnostics']==decision['search_diagnostics'],'numeric proposal diagnostics lineage changed')
                    if method=='llm_feedback_off':off_commits.append(stamp(commitpath))
                    elif measured:require(stamp(panel/'candidates'/(proposed[-1]['candidate_id']+'.json'))<=stamp(obs_path),'feedback observed before previous result committed')
                    results={};refs={}
                    for condition in DEVELOPMENT:
                        trialid='candidate__'+identity+'__'+condition
                        require(trialid in starts,'candidate trial missing')
                        require(stamp(commitpath)<stamp(starts[trialid][1]),'candidate execution before proposal commitment')
                        trialproposal=read(panel/'trial_proposals'/(trialid+'.json'))
                        require(trialproposal['repair_candidate_id']==identity and trialproposal['control_tool']==proposal['control_tool']
                            and [trialproposal['parameters'][n] for n in protocol['parameter_names']]==proposal['parameters'],'candidate execution vector mismatch')
                        results[condition]=evaluations[trialid]
                        refs[condition]=dict(trial_id=trialid,evaluation=dict(path=str(panel/'evaluations'/(trialid+'.json')),sha256=sha(panel/'evaluations'/(trialid+'.json'))))
                    candidate=read(panel/'candidates'/(identity+'.json'))
                    require((candidate['candidate_id'],candidate['task_id'],candidate['method'])==(identity,taskid,method),
                        'candidate result stream identity changed')
                    require(candidate['provenance']==dict(proposal=dict(path=str(proposalpath),sha256=sha(proposalpath))),
                        'candidate result proposal provenance changed')
                    require(candidate['search_diagnostics']==proposal.get('search_diagnostics'),
                        'candidate numeric diagnostics lineage changed')
                    require(candidate['condition_records']==refs,'candidate condition links changed')
                    require(candidate['aggregate']==aggregate_conditions(results,DEVELOPMENT),'candidate aggregate changed')
                    expected_states=[dict(scenario_id=s,integrity_passed=results[s]['integrity_passed'],physical_gate_passed=results[s]['physical_gate_passed'],features=results[s]['primary']['features']) for s in DEVELOPMENT]
                    require(candidate['state_evaluations']==expected_states,'candidate feedback feature labels changed')
                    require(candidate['control_tool']==proposal['control_tool'] and candidate['parameters']==proposal['parameters'],'candidate result vector changed')
                    require(max(stamp(completes['candidate__'+identity+'__'+s][1]) for s in DEVELOPMENT)<=stamp(panel/'candidates'/(identity+'.json')),'batch committed before completion')
                    candidates[identity]=results;measured.append(candidate)
                    proposed.append(dict(candidate_id=identity,control_tool=proposal['control_tool'],parameters=proposal['parameters']))
                incumbent={s:read(base/'evaluations'/(s+'__selected_'+taskid+'.json')) for s in DEVELOPMENT}
                candidates['incumbent']=incumbent;candidate_maps[(taskid,method)]=candidates
        selection_path=panel/'selection.json';selection=read(selection_path)
        require(selection['confirmation_used_for_selection'] is False and len(selection['winners'])==6,'selection shape')
        require({(w['task_id'],w['method']) for w in selection['winners']}==set(candidate_maps),'exact six winner streams')
        for winner in selection['winners']:
            taskid,method=winner['task_id'],winner['method']
            checked=audit_selection(candidate_maps[(taskid,method)],winner['candidate_id'])
            checked.update(task_id=taskid,method=method)
            report['streams'].append(checked)
            if winner['candidate_id']=='incumbent':
                prior=read(pinned(protocol['shared_priors'][taskid]))
                expected=copy.deepcopy(next(r for r in prior['observations'] if r['candidate_id']=='incumbent'))
                results=candidate_maps[(taskid,method)]['incumbent']
                expected.update(task_id=taskid,method=method,aggregate=aggregate_conditions(results,DEVELOPMENT),
                    condition_records={s:dict(trial_id=s+'__selected_'+taskid,
                        evaluation=dict(path=str(base/'evaluations'/(s+'__selected_'+taskid+'.json')),
                            sha256=sha(base/'evaluations'/(s+'__selected_'+taskid+'.json'))),cached_prior=True)
                        for s in DEVELOPMENT})
            else:
                expected=read(panel/'candidates'/(winner['candidate_id']+'.json'))
            require(winner==expected,'frozen winner fields differ from selected measured candidate')
        frozenpath=panel/'events/selection_frozen__all6.json'
        require(pinned(read(frozenpath)['selection'])==selection_path,'selection frozen link')
        require(max(stamp(panel/'candidates'/(identity+'.json')) for identity in expected_candidates)<=stamp(selection_path)<=stamp(frozenpath),'all winners not frozen after all results')
        candidate_start=[stamp(path) for item,path in starts.values() if item['phase']=='candidate']
        confirm_start=[stamp(path) for item,path in starts.values() if item['phase']=='confirmation']
        report['chronology']=check_temporal_isolation(off_commits,candidate_start,[stamp(frozenpath)]*6,confirm_start)
        summary=read(panel/'summary.json')
        require(summary['protocol_sha256']==selection['protocol_sha256']==sha(panel/'protocol.json'),'summary/selection protocol hash')
        require(summary['completed'] is True and summary['errors']==[] and summary['training_admission'] is False
            and summary['confirmation_is_known_development'] is True,'summary completion claims')
        require((summary['physical_trials'],summary['control_steps'],summary['physics_steps'],summary['model_proposal_turns'])==(250,50000,1000000,32),'summary accounting')
        require(len(summary['winners'])==6,'summary winner inventory')
        for frozen in selection['winners']:
            taskid,method=frozen['task_id'],frozen['method']
            matching=[w for w in summary['winners'] if (w['task_id'],w['method'])==(taskid,method)]
            require(len(matching)==1,'duplicate/missing summary winner');row=matching[0]
            require(row['selected_candidate_id']==frozen['candidate_id'] and row['control_tool']==frozen['control_tool'] and row['parameters']==frozen['parameters'],'confirmation changed selected output')
            require(set(row['condition_records'])==set(scenarios),'full13 summary condition inventory')
            results={}
            for condition,record in row['condition_records'].items():
                result=read(pinned(record['evaluation']));results[condition]=result
                if condition in DEVELOPMENT:
                    require(record==frozen['condition_records'][condition],'development evidence replaced in confirmation')
                    if frozen['candidate_id']=='incumbent':require(record.get('cached_prior') is True,'incumbent cache origin hidden')
                else:
                    trialid='confirmation__'+taskid+'__'+method+'__'+condition
                    require(record['trial_id']==trialid and record.get('fresh_confirmation') is True,'fresh confirmation identity')
                    require(result==evaluations[trialid],'confirmation eval link changed')
                    trialproposal=read(panel/'trial_proposals'/(trialid+'.json'))
                    require(trialproposal['repair_candidate_id']==frozen['candidate_id'] and trialproposal['control_tool']==frozen['control_tool']
                        and [trialproposal['parameters'][n] for n in protocol['parameter_names']]==frozen['parameters'],'confirmation output drift')
            require(row['development_aggregate']==aggregate_conditions({s:results[s] for s in DEVELOPMENT},DEVELOPMENT),'summary development aggregate')
            require(row['full_aggregate']==aggregate_conditions(results,list(scenarios)),'summary full aggregate')
            require(row['primary_passed']==sum(result['primary']['joint_success'] for result in results.values()),'summary primary count')
            next(r for r in report['streams'] if (r['task_id'],r['method'])==(taskid,method)).update(
                nominal_primary_success=results['nominal']['primary']['joint_success'],
                nonzero_primary_successes=sum(result['primary']['joint_success'] for key,result in results.items() if key!='nominal'),
                nonzero_conditions=12)
        for protected in protocol['protected_inventories']:
            require(inventory(resolve(protected['path']))==protected['files'],'old evidence changed during audit')
        for protected in inherited_protected:
            require(inventory(resolve(protected['path']))==protected['files'],'inherited evidence changed during audit')
        for name in ('quarantine','quarantine_inventory'):pinned(robust[name])
        report.update(audit_passed=True,protected_evidence_unchanged=True,prior_provenance_verified=True,
            numeric_decisions_replayed=True,observation_isolation_verified=True)
    except Exception as error:
        import traceback
        report['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    report['panel_evidence_unchanged']=inventory(panel)==before
    if not report['panel_evidence_unchanged']:
        report['audit_passed']=False;report['errors'].append(dict(error='audit changed input evidence'))
    (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.panel,args.output)
    print(json.dumps({key:result[key] for key in ('audit_passed','errors','new_physical_trials','new_llm_calls')}))
    raise SystemExit(0 if result['audit_passed'] else 1)
