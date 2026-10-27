"""Freeze a warm-start development repair comparison before new proposals."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
from musical_tasks import ROOT, read, sha256, write_new_json
from audit_robustness_v2 import inventory

ROBUST = ROOT/'artifacts/runs/agentic/robustness_initial_state_v2'
OLD = ROOT/'artifacts/runs/agentic/tool_selection_development_v2'
METHODS = ['surrogate_search','llm_feedback','llm_feedback_off']
TASKS = ['wave_a','high5_b']
DEV = ['nominal','joint_position_minus','combined_plus','combined_minus']
NEW_CODE = ['repair_plan.py','repair_driver.py','repair_model_io.py','repair_search.py','execute_repair.py','evaluate_repair.py','audit_repair.py']


def link(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha256(path))


def feature_row(result, scenario):
    score=result.get('primary',result)
    return dict(scenario_id=scenario,integrity_passed=result['integrity_passed'],
        physical_gate_passed=result['physical_gate_passed'],features=score['features'])


def prior_data(task, old_protocol):
    """Same task-specific observed labels for every method; no old reason text."""
    source=task['source_id']; names=old_protocol['parameter_names']; observations=[]
    identities=[f"{task['task_id']}__{method}__{i}" for method in METHODS for i in range(8)]
    identities += ['calibration_'+source,'calibration_local_'+source]
    for identity in identities:
        proposal=read(OLD/'proposals'/(identity+'.json'))
        path=OLD/'evaluations'/(identity+'.json'); result=read(path)
        observations.append(dict(candidate_id='prior:'+identity,control_tool=proposal['control_tool'],
            parameters=[proposal['parameters'][n] for n in names],state_evaluations=[feature_row(result,'nominal')],
            provenance=dict(proposal=link(OLD/'proposals'/(identity+'.json')),evaluation=link(path)),
            scope='historical nominal observation; repeated development task'))
    case=next(c for c in read(ROBUST/'protocol.json')['cases'] if c['role']=='selected_'+task['task_id'])
    seed=read(ROOT/case['original_proposal']['path']); states=[]; links=[]
    for scenario in read(ROBUST/'protocol.json')['scenarios']:
        path=ROBUST/'evaluations'/(scenario['scenario_id']+'__selected_'+task['task_id']+'.json')
        states.append(feature_row(read(path),scenario['scenario_id']));links.append(link(path))
    observations.append(dict(candidate_id='incumbent',control_tool=seed['control_tool'],
        parameters=[seed['parameters'][n] for n in names],state_evaluations=states,
        provenance=dict(proposal=case['original_proposal'],evaluations=links),
        scope='fixed incumbent13 prior states; nominal repeats existing trace, not independent evidence'))
    return dict(schema='pulsemorph.agentic_repair_shared_prior.v1',task_id=task['task_id'],
        observations=observations,seed_proposal=case['original_proposal'],
        interpretation='Warm-start known development evidence shared equally. OFF excludes only NEW candidate feedback.',
        known_diagnostics='high5_b failures are left-B amplitude. wave_a combined-minus retains peaks at2.4/2.9s but their rank flips. Frozen dominant-peak metric unchanged.')


def initialize(output):
    out=Path(output).resolve()
    if (ROOT/'artifacts/runs/agentic').resolve() not in out.parents or out.exists():
        raise ValueError('new separate agentic output required')
    robust=read(ROBUST/'protocol.json'); old=read(OLD/'protocol.json')
    audit_path=ROBUST.parent/'robustness_initial_state_v2_audit_v1/audit.json'
    audit=read(audit_path)
    if not audit['audit_passed'] or audit['input_protocol_sha256']!=sha256(ROBUST/'protocol.json'):
        raise ValueError('accepted previous robustness audit required')
    for name,digest in robust['execution_code_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('previous frozen source changed: '+name)
    tasks=[copy.deepcopy(next(t for t in old['tasks'] if t['task_id']==name)) for name in TASKS]
    priors={t['task_id']:prior_data(t,old) for t in tasks}
    keys=['sources','parameter_names','bounds','anchor','feature_names','feature_windows_s','feature_tolerances',
        'physical_gate','accent_window_s','accent_smoothing_samples','accent_minimum_prominence','section_windows_s','duration_s','source_start_s']
    protocol={key:copy.deepcopy(old[key]) for key in keys}
    pins=dict(robust['execution_code_sha256'])
    for filename in NEW_CODE:pins['experiments/agentic/'+filename]=sha256(Path(__file__).parent/filename)
    pins['experiments/agentic/tool_selection_driver_v2.py']=sha256(Path(__file__).with_name('tool_selection_driver_v2.py'))
    protected_paths=[ROBUST,OLD,audit_path.parent,ROBUST.parent/'robustness_initial_state_v2_report_v1']
    protected=[dict(path=str(p),files=inventory(p)) for p in protected_paths]
    protocol.update(schema='pulsemorph.agentic_robust_repair_protocol.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,methods=METHODS,control_tools=['global_clock_v1','local_clock_1p4_2p8_v1'],budget=8,
        development_condition_ids=DEV,scenarios=robust['scenarios'],
        confirmation_condition_ids=[s['scenario_id'] for s in robust['scenarios'] if s['scenario_id'] not in DEV],
        baseline_panel=str(ROBUST),accepted_prior_audit=link(audit_path),original_protocol=link(OLD/'protocol.json'),
        robustness_protocol=link(ROBUST/'protocol.json'),execution_code_sha256=pins,protected_inventories=protected,
        seed_proposals={task:prior['seed_proposal'] for task,prior in priors.items()},
        repair_wrapper_pin_switch=dict(old=dict(path='experiments/agentic/execute_robustness.py',sha256=sha256(Path(__file__).with_name('execute_robustness.py'))),
            new=dict(path='experiments/agentic/execute_repair.py',sha256=sha256(Path(__file__).with_name('execute_repair.py')))),
        planned_budget=dict(gate_trials=4,candidate_trials=192,confirmation_trials=54,physical_trials=250,
            control_steps=50000,physics_steps=1000000,model_proposal_turns=32),
        selection='Incumbent plus8 new candidates: all4 physical gates first, max primary normalized feature error, sum squared normalized errors, candidate ID.',
        confirmation='Freeze all6 method winners before any confirmation. Reuse their4 development rows; run9 other known conditions. Incumbent cached rows explicitly linked. Never select again.',
        stopping_policy='No early success stop. Technical/schema/integrity errors abort with preserved evidence; measured physical/target failures stay data; no retries.',
        off_policy='Commit all16 OFF responses before any new candidate physics; initial shared prior is available, own proposed vectors only thereafter.',
        prior_policy='All24 old task candidates plus2 nominal identity tools and incumbent13-state evidence; equal labels for allmethods. Repeated nominal trace is not independent evidence.',
        optimizer=dict(algorithm='repair_weighted_affine_minimax_v1',nominal_jacobian_transfer_is_assumption=True,adaptive_tool_allocation=True,
            nominal_ridge=1e-4,state_correction_ridge=1e-3,distance_scale=.5,initial_trust_radius=.2,minimum_trust_radius=.025,
            duplicate_exact_labels_removed_from_fit=True),
        model=dict(model='gpt-6.1-sol',fork_turns='none',turns_per_task_method=8,provider_tokens=None,provider_billed_cost=None),
        runtime_input_sha256=robust['runtime_input_sha256'],training_admission=False,
        limitations=['Allsources/tasks/13conditions are known development evidence; no clean held-out source or condition claim.',
            'Canonical numeric goals and measured music events are provided; no raw language/audio understanding score.',
            'Equal proposal turns/physics slots do not match tokens, time or cost. Engineering agent calls excluded.',
            'No motor learning, live recovery, aesthetics or population robustness acceptance.'])
    out.mkdir(parents=True)
    for name,digest in pins.items():
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        if sha256(target)!=digest:raise ValueError('source changed during freeze')
    for task,prior in priors.items():write_new_json(out/'priors'/(task+'.json'),prior)
    protocol['shared_priors']={task:link(out/'priors'/(task+'.json')) for task in priors}
    from audit_repair import audit_plan
    audit_plan(protocol)
    write_new_json(out/'protocol.json',protocol)
    return protocol
