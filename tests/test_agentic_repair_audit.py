import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
from audit_repair import (DEVELOPMENT, CONFIRMATION, METHODS, TASKS, audit_selection, check_budget,
                          check_hash_inventory, check_temporal_isolation, audit_plan)
from robustness_scenarios import build_scenarios


def measurements(error=.2, physical=True):
    return {condition: dict(integrity_passed=True, physical_gate_passed=physical,
        primary=dict(feature_errors=[error]*5, feature_tolerances=[1.]*5)) for condition in DEVELOPMENT}


def ledger():
    result = [dict(trial_id='gate'+str(i), phase='runtime_gate', fresh_physics=True,
                   control_steps=200, physics_steps=4000) for i in range(4)]
    for task in TASKS:
        for method in METHODS:
            for index in range(8):
                for condition in DEVELOPMENT:
                    result.append(dict(trial_id=f'{task}:{method}:{index}:{condition}', phase='candidate',
                        task_id=task,method=method,proposal_index=index,condition_id=condition,
                        fresh_physics=True,control_steps=200,physics_steps=4000))
            for i, condition in enumerate(CONFIRMATION):
                result.append(dict(trial_id=f'{task}:{method}:confirm:{i}', phase='confirmation',
                    task_id=task,method=method,condition_id=condition,fresh_physics=True,
                    control_steps=200,physics_steps=4000))
    return result


class RepairAuditTests(unittest.TestCase):
    def test_plan_admission_rejects_budget_and_scenario_drift(self):
        protocol=dict(schema='pulsemorph.agentic_robust_repair_protocol.v1',methods=list(METHODS),
            tasks=[{'task_id':task} for task in TASKS],budget=8,
            development_condition_ids=list(DEVELOPMENT),confirmation_condition_ids=list(CONFIRMATION),
            scenarios=build_scenarios(['joint'+str(i) for i in range(29)]),training_admission=False,
            planned_budget=dict(gate_trials=4,candidate_trials=192,confirmation_trials=54,
                physical_trials=250,control_steps=50000,physics_steps=1000000,model_proposal_turns=32),
            model=dict(model='gpt-6.1-sol',fork_turns='none',turns_per_task_method=8),
            shared_priors={task:{} for task in TASKS})
        self.assertTrue(audit_plan(protocol)['plan_admission_passed'])
        protocol['planned_budget']['physical_trials']=249
        with self.assertRaisesRegex(ValueError,'budget'):audit_plan(protocol)
        protocol['planned_budget']['physical_trials']=250
        protocol['scenarios'][1]['initial_perturbation']['joint_position_offset_rad'][0]*=2
        with self.assertRaisesRegex(ValueError,'numeric'):audit_plan(protocol)

    def test_selection_recomputes_residuals_and_physical_first(self):
        candidates = {str(i):measurements(.4+i*.01) for i in range(9)}
        candidates['0'] = measurements(.01,False)
        candidates['8'] = measurements(.2)
        self.assertEqual(audit_selection(candidates,'8')['winner_id'],'8')
        with self.assertRaisesRegex(ValueError,'selector'): audit_selection(candidates,'0')
        candidates['8']['nominal']['primary']['utility'] = 999
        self.assertEqual(audit_selection(candidates,'8')['aggregate']['utility'],.2)

    def test_missing_feature_is_losing_candidate_not_protocol_abort(self):
        candidates={str(i):measurements(.2) for i in range(9)}
        candidates['0']['nominal']['primary']['feature_errors'][4]=None
        self.assertEqual(audit_selection(candidates,'1')['winner_id'],'1')
        candidates['1']['nominal']['integrity_passed']=False
        with self.assertRaisesRegex(ValueError,'integrity'): audit_selection(candidates,'2')

    def test_budget_rejects_cached_or_missing_batch(self):
        rows=ledger()
        self.assertEqual(check_budget(rows)['new_physical_trials'],250)
        rows[5]['fresh_physics']=False
        with self.assertRaises(ValueError): check_budget(rows)
        rows=ledger();rows[5]['condition_id']='wrong'
        with self.assertRaises(ValueError): check_budget(rows)
        rows=ledger();rows[-1]['condition_id']=DEVELOPMENT[0]
        with self.assertRaises(ValueError): check_budget(rows)

    def test_barriers(self):
        self.assertTrue(check_temporal_isolation([1]*16,[2]*192,[3]*6,[4]*54)['off_commit_barrier_verified'])
        for values in (([2]*16,[2]*192,[3]*6,[4]*54),([1]*16,[2]*192,[4]*6,[4]*54)):
            with self.assertRaises(ValueError): check_temporal_isolation(*values)

    def test_protected_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); path=root/'old.json';path.write_bytes(b'old')
            before={'old.json':hashlib.sha256(b'old').hexdigest()}
            self.assertTrue(check_hash_inventory(root,before)['protected_evidence_unchanged'])
            path.write_bytes(b'new')
            with self.assertRaises(ValueError): check_hash_inventory(root,before)

if __name__=='__main__': unittest.main()

class RepairAuditOrchestrationTests(unittest.TestCase):
    """A synthetic receipt panel exercises orchestration, never physics math."""
    def make_panel(self, folder):
        import copy
        import os
        import numpy as np
        from evaluate_repair import aggregate_conditions
        root=Path(folder);panel=root/'panel';base=root/'old_panel'
        clock=[1_800_000_000_000_000_000]
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf8')
            clock[0]+=1_000_000;os.utime(path,ns=(clock[0],clock[0]));return path
        def link(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        def event(kind,identity,**fields):
            return write(panel/'events'/(kind+'__'+identity+'.json'),dict(event=kind,identity=identity,**fields))
        tasks=[dict(task_id=t,source_id='wave' if t=='wave_a' else 'high5',source_motion={'path':t,'sha256':'d'*64},goal={'feature_targets':[1.]*5}) for t in TASKS]
        source_by_task={t['task_id']:t['source_motion'] for t in tasks}
        scenarios=build_scenarios(['joint'+str(i) for i in range(29)])
        oldpin='experiments/agentic/execute_robustness.py';newpin='experiments/agentic/execute_repair.py'
        oldpins={oldpin:'a'*64};newpins={newpin:'b'*64}
        quarantine=write(root/'quarantine.json',{'historical':True})
        quarantine_inventory=write(root/'quarantine_inventory.json',{})
        oldproto=write(base/'protocol.json',dict(protected_inventories=[],quarantine=link(quarantine),
            quarantine_inventory=link(quarantine_inventory)))
        original=write(root/'original.json',{})
        accepted=write(root/'old_audit.json',dict(audit_passed=True,input_protocol_sha256=link(oldproto)['sha256']))
        seeds={}
        roles=('calibration_wave','calibration_high5','selected_wave_a','selected_high5_b')
        for condition in DEVELOPMENT+CONFIRMATION:
            for source in ('wave','high5'):
                write(base/'trials'/(condition+'__calibration_'+source)/'manifest.json',dict(preexecution_input_hashes=oldpins))
        for role in roles:
            task='wave_a' if 'wave' in role else 'high5_b'
            old=base/'trials'/('nominal__'+role)
            write(old/'manifest.json',dict(preexecution_input_hashes=oldpins))
            proposal=write(old/'proposal.json',dict(task_id=task,source_motion=source_by_task[task]))
            if role.startswith('selected'):seeds[task]=link(proposal)
        def evaluation(error=.2):
            return dict(integrity_passed=True,physical_gate_passed=True,primary=dict(features=[1.+error]*5,
                feature_errors=[error]*5,feature_tolerances=[1.]*5,joint_success=True,
                utility=error,utility_l2=float(np.sum(np.asarray([error]*5)**2)) ))
        priors={}
        for task in TASKS:
            observations=[dict(candidate_id='prior'+str(i)) for i in range(26)]
            observations.append(dict(candidate_id='incumbent',control_tool='global_clock_v1',parameters=[1.,1.,1.,1.,0.],state_evaluations=[],provenance={'old':True}))
            prior=dict(task_id=task,observations=observations)
            priors[task]=prior;write(panel/'priors'/(task+'.json'),prior)
            for s in DEVELOPMENT+CONFIRMATION:write(base/'evaluations'/(s+'__selected_'+task+'.json'),evaluation(.3))
        protocol=dict(schema='pulsemorph.agentic_robust_repair_protocol.v1',methods=list(METHODS),tasks=tasks,budget=8,
            development_condition_ids=list(DEVELOPMENT),confirmation_condition_ids=list(CONFIRMATION),scenarios=scenarios,
            training_admission=False,planned_budget=dict(gate_trials=4,candidate_trials=192,confirmation_trials=54,
                physical_trials=250,control_steps=50000,physics_steps=1000000,model_proposal_turns=32),
            model=dict(model='gpt-6.1-sol',fork_turns='none',turns_per_task_method=8),
            shared_priors={t:link(panel/'priors'/(t+'.json')) for t in TASKS},execution_code_sha256={},protected_inventories=[],
            original_protocol=link(original),robustness_protocol=link(oldproto),accepted_prior_audit=link(accepted),
            baseline_panel=str(base),seed_proposals=seeds,parameter_names=['left_gain_a','right_gain_a','left_gain_b','right_gain_b','timing_shift_s'],
            repair_wrapper_pin_switch=dict(old=dict(path=oldpin,sha256='a'*64),new=dict(path=newpin,sha256='b'*64)),
            physical_gate=dict(max_mean_joint_rmse_rad=.35,min_root_height_m=.6))
        write(panel/'protocol.json',protocol);phash=link(panel/'protocol.json')['sha256']
        fake_audits={};fake_results={}
        def trial(identity,phase,task,method,candidate,condition,tool,parameters,seed):
            source=source_by_task[task]
            proposal=dict(candidate_id=identity,source_motion=source,parameter_source=seed,protocol_sha256=phash,
                repair_candidate_id=candidate,control_tool=tool,parameters=dict(zip(protocol['parameter_names'],parameters)))
            pp=write(panel/'trial_proposals'/(identity+'.json'),proposal)
            event('trial_start',identity,phase=phase,task_id=task,method=method,candidate_id=candidate,scenario_id=condition,proposal=link(pp))
            td=panel/'trials'/identity
            write(td/'proposal.json',proposal)
            (td/'source_proposal.json').write_bytes(Path(seed['path']).read_bytes())
            manifest=dict(candidate_id=identity,status='success',strict_g1_admission_passed=True,strict_g1_preflight_count=2,
                actual_control_steps=200,actual_physics_steps=4000,preexecution_input_hashes=newpins)
            mp=write(td/'manifest.json',manifest)
            event('trial_complete',identity,phase=phase,manifest=link(mp),status='success',control_steps=200,physics_steps=4000)
            fake_audits[str(td)]=dict(manifest=manifest,fresh_anchor_audit={'control_rows':200},mean_joint_rmse_rad=.1,
                sub=np.array([[0,0,.8]]),arrays={},names=[])
            result=evaluation();fake_results[str(td)]=result
            if phase!='gate':write(panel/'evaluations'/(identity+'.json'),result)
            return dict(trial_id=identity,evaluation=link(panel/'evaluations'/(identity+'.json'))) if phase!='gate' else link(mp)
        cases=[]
        for role in roles:
            task='wave_a' if 'wave' in role else 'high5_b';old=base/'trials'/('nominal__'+role)
            manifest=trial('gate__'+role,'gate',task,'surrogate_search','gate_'+role,'nominal','global_clock_v1',[1.,1.,1.,1.,0.],link(old/'proposal.json'))
            cases.append(dict(role=role,trial_id='gate__'+role,old_trial=str(old),manifest=manifest,parity={'same':True}))
        gate=write(panel/'gate.json',dict(passed=True,cases=cases,protocol_sha256=phash))
        event('gate_passed','all4',gate=link(gate))
        candidates={};observations={}
        def propose(task,method,index,measured,proposed):
            identity=f'{task}__{method}__{index}'
            obs=dict(shared_prior=priors[task],prior_proposals=copy.deepcopy(proposed),prior_trial_results=copy.deepcopy([] if method=='llm_feedback_off' else measured))
            observations[identity]=obs;op=write(panel/'observations'/(identity+'.json'),obs)
            response=dict(control_tool='global_clock_v1',parameters=[1.,1.,1.,1.,0.],only_observation_path=str(op),brief_reason='test')
            requestpath=panel/'model_requests'/(identity+'.json')
            dispatch='2026-10-27T10:00:00+00:00'
            if method.startswith('llm'):
                write(requestpath,dict(candidate_id=identity,model='gpt-6.1-sol',fork_turns='none',
                    observation=link(op),expected_response_path=str(panel/'responses'/(identity+'.json')),
                    prompt='Read only '+str(op),isolation='fresh context, no OS sandbox',
                    dispatch_requested_at_utc=dispatch,provider_tokens=None,provider_billed_cost=None))
            rp=write(panel/'responses'/(identity+'.json'),response)
            proposal=dict(candidate_id=identity,task_id=task,method=method,trial_index=index,control_tool=response['control_tool'],parameters=response['parameters'],observation=link(op),response=link(rp))
            if method.startswith('llm'):write(panel/'model_calls'/(identity+'.json'),dict(candidate_id=identity,
                model='gpt-6.1-sol',fork_turns='none',observation=link(op),response=link(rp),request=link(requestpath),
                dispatch_requested_at_utc=dispatch,completion_observed_at_utc='2026-10-27T10:00:01+00:00',
                provider_tokens=None,provider_billed_cost=None,agent_task='synthetic',status='response_observed'))
            else:
                diagnostic={'index':index};proposal['search_diagnostics']=diagnostic
                write(panel/'search_decisions'/(identity+'.json'),dict(control_tool=response['control_tool'],parameters=response['parameters'],search_diagnostics=diagnostic))
            pp=write(panel/'proposals'/(identity+'.json'),proposal);event('proposal_committed',identity,proposal=link(pp))
            return proposal
        for task in TASKS:
            proposed=[]
            for i in range(8):
                proposal=propose(task,'llm_feedback_off',i,[],proposed)
                proposed.append({k:proposal[k] for k in ('candidate_id','control_tool','parameters')})
        for task in TASKS:
            for method in METHODS:
                measured=[];proposed=[]
                for i in range(8):
                    identity=f'{task}__{method}__{i}'
                    proposal=read_json(panel/'proposals'/(identity+'.json')) if method=='llm_feedback_off' else propose(task,method,i,measured,proposed)
                    refs={s:trial('candidate__'+identity+'__'+s,'candidate',task,method,identity,s,proposal['control_tool'],proposal['parameters'],seeds[task]) for s in DEVELOPMENT}
                    results={s:fake_results[str(panel/'trials'/refs[s]['trial_id'])] for s in DEVELOPMENT}
                    row=dict(candidate_id=identity,task_id=task,method=method,control_tool=proposal['control_tool'],parameters=proposal['parameters'],condition_records=refs,
                        aggregate=aggregate_conditions(results,DEVELOPMENT),search_diagnostics=proposal.get('search_diagnostics'),
                        provenance=dict(proposal=link(panel/'proposals'/(identity+'.json'))),
                        state_evaluations=[dict(scenario_id=s,integrity_passed=True,physical_gate_passed=True,features=results[s]['primary']['features']) for s in DEVELOPMENT])
                    write(panel/'candidates'/(identity+'.json'),row);candidates[identity]=row;measured.append(row)
                    proposed.append({k:proposal[k] for k in ('candidate_id','control_tool','parameters')})
        winners=[copy.deepcopy(candidates[f'{t}__{m}__0']) for t in TASKS for m in METHODS]
        sp=write(panel/'selection.json',dict(winners=winners,protocol_sha256=phash,confirmation_used_for_selection=False))
        event('selection_frozen','all6',selection=link(sp))
        confirmed=[]
        for winner in winners:
            task,method=winner['task_id'],winner['method'];refs=copy.deepcopy(winner['condition_records'])
            for s in CONFIRMATION:
                refs[s]=trial('confirmation__'+task+'__'+method+'__'+s,'confirmation',task,method,winner['candidate_id'],s,winner['control_tool'],winner['parameters'],seeds[task]);refs[s]['fresh_confirmation']=True
            results={s:read_json(r['evaluation']['path']) for s,r in refs.items()}
            confirmed.append(dict(task_id=task,method=method,selected_candidate_id=winner['candidate_id'],control_tool=winner['control_tool'],parameters=winner['parameters'],development_aggregate=winner['aggregate'],full_aggregate=aggregate_conditions(results,[s['scenario_id'] for s in scenarios]),primary_passed=13,condition_records=refs))
        write(panel/'summary.json',dict(completed=True,errors=[],training_admission=False,confirmation_is_known_development=True,
            protocol_sha256=phash,physical_trials=250,control_steps=50000,physics_steps=1000000,model_proposal_turns=32,winners=confirmed))
        return panel,priors,observations,fake_audits,fake_results

    def test_complete_receipt_panel_and_diagnostic_tamper(self):
        import numpy as np
        from unittest.mock import patch
        from contextlib import ExitStack
        from audit_repair import audit
        def fake_raw(path):
            key=str(path)
            return raw.get(key,dict(manifest={},arrays={},names=[]))
        with tempfile.TemporaryDirectory() as temporary:
            panel,priors,observations,raw,results=self.make_panel(temporary)
            with ExitStack() as stack:
                stack.enter_context(patch('evaluate_generalization._audit',side_effect=fake_raw))
                stack.enter_context(patch('audit_robustness_v2.check_initial_state',return_value={}))
                stack.enter_context(patch('audit_robustness_v2.check_center_definitions',return_value={}))
                stack.enter_context(patch('audit_multifeature.common_parity',return_value={}))
                stack.enter_context(patch('run_tool_compatibility_smoke.trace_parity',return_value={'same':True}))
                stack.enter_context(patch('evaluate_repair.evaluate_repair',side_effect=lambda directory,*args:results[str(directory)]))
                stack.enter_context(patch('repair_plan.prior_data',side_effect=lambda task,*args:priors[task['task_id']]))
                stack.enter_context(patch('repair_driver.observation_data',side_effect=lambda out,task,method,index:observations[f'{task}__{method}__{index}']))
                stack.enter_context(patch('repair_search.next_candidate',side_effect=lambda measured,*args:dict(control_tool='global_clock_v1',parameters=[1.,1.,1.,1.,0.],search_diagnostics={'index':len(measured)})))
                result=audit(panel,Path(temporary)/'audit_good')
                self.assertTrue(result['audit_passed'],result['errors'])
                target=panel/'candidates/wave_a__surrogate_search__7.json'
                changed=read_json(target);changed['search_diagnostics']={'index':999};target.write_text(json.dumps(changed))
                result=audit(panel,Path(temporary)/'audit_bad')
                self.assertFalse(result['audit_passed'])
                self.assertIn('diagnostics lineage',str(result['errors']))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))
