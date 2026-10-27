"""Freeze a development-only two-tool comparison; never relabel reused tasks."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
from musical_tasks import ROOT, read, sha256, write_new_json


def initialize(output):
    out = Path(output).resolve()
    if (ROOT/'artifacts/runs/agentic').resolve() not in out.parents:
        raise ValueError('new agentic output directory required')
    old_path = ROOT/'artifacts/runs/agentic/generalization_music_v1/protocol.json'
    old = read(old_path)
    if sha256(old_path) != '80c2fac7bbc1a886e774e4d32aa569c069a3f4750e91a4a33628f8594df667ec':
        raise ValueError('original task protocol changed')
    if [t['task_id'] for t in old['tasks']] != ['wave_a','wave_b','high5_a','high5_b']:
        raise ValueError('all four original development tasks required')
    inventory = ROOT/'artifacts/runs/agentic/tool_source_inventory_v1/inventory.json'
    if not inventory.is_file():
        raise ValueError('source availability inventory required')
    # Fail before creating an incomplete freeze if implementation is not ready.
    names = ['tool_selection_tasks_v2.py','tool_selection_driver_v2.py','tool_selection_search.py','execute_tool_selection.py','tool_selection_batch_v2.py']
    pins = dict(old['execution_code_sha256'])
    additions = [Path('experiments/agentic')/n for n in names]
    additions += [Path('src/pulsemorph/agentic/local_multifeature_reference.py')]
    additions += [Path('experiments/agentic/execute_local_capability.py')]
    for name in additions:
        pins[name.as_posix()] = sha256(ROOT/name)
    for name, digest in pins.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('required frozen source changed: '+name)
    out.mkdir(parents=True, exist_ok=False)
    tasks = copy.deepcopy(old['tasks'])
    for task in tasks:
        observation = task['music_observation']
        origin = Path(observation['waveform_path'])
        if sha256(origin) != observation['waveform_sha256']:
            raise ValueError('original waveform changed')
        folder = out/'music'/task['task_id']
        folder.mkdir(parents=True)
        target = folder/'music.wav'
        shutil.copyfile(origin,target)
        observation['waveform_path'] = str(target)
        write_new_json(folder/'waveform_observation.json', observation)
        truth = origin.parent/'authored_truth.json'
        shutil.copyfile(truth,folder/'authored_truth.json')
    base_keys = ['sources','parameter_names','bounds','anchor','feature_names','feature_windows_s',
                 'feature_tolerances','physical_gate','accent_window_s','accent_smoothing_samples',
                 'accent_minimum_prominence','section_windows_s','duration_s','source_start_s','selection']
    protocol = {key:copy.deepcopy(old[key]) for key in base_keys}
    protocol.update(
        schema='pulsemorph.agentic_tool_selection_protocol.v2',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,methods=['surrogate_search','llm_feedback','llm_feedback_off'],
        control_tools=['global_clock_v1','local_clock_1p4_2p8_v1'],
        execution_code_sha256=pins,
        original_task_protocol={'path':str(old_path),'sha256':sha256(old_path)},
        source_availability_inventory={'path':str(inventory),'sha256':sha256(inventory)},
        phase='development_comparison',
        new_source_main_panel_status='blocked: only one eligible new source inside existing trusted allowlist; four-source goal retained',
        source_claim='All four tasks reuse earlier development sources, instructions and waveform bytes. No held-out-source claim.',
        imported_prior_trial_records=[],
        trial_budget_per_method_per_task=8,new_candidate_budget=8,
        calibration_ids=['calibration_wave','calibration_local_wave','calibration_high5','calibration_local_high5'],
        calibration_policy='Two fresh identity runs per source, one for each tool. Require full qpos/qvel/PD byte parity, common q/dq/torso array equality, identical runtime input hashes and policy input values before any planner observation or candidate. Use the global identity run as the sole shared measured denominator; local run is an unscored parity check. All four runs counted.',
        optimizer={'algorithm':'mixed_tool_search_development_v1','schedule':'global seed, local seed, global maximin, local maximin, model, maximin, model, maximin',
                   'per_tool_models':True,'calibration_evidence_count_per_source':1,
                   'predictions_are_measurements':False,'old_feature_labels_imported':False},
        feedback_off_definition='Eight individual model turns for each of four tasks, conditioned on equal common context and own previous proposals only. All32 OFF responses must be committed before any candidate physics.',
        model={'requested_model':'gpt-6.1-sol','access':'Codex collaboration','calls_per_method_task':8,
               'fork_turns':'none','reasoning_setting':'same inherited setting for both conditions',
               'provider_tokens':None,'provider_billed_cost':None},
        semantic_stage='None: canonical numeric targets supplied equally; Turkish instructions are context. No new language or raw-audio understanding score.',
        stopping_policy='All8 candidate slots per method/task, no success early stop. No confirmation in this comparison. Schema/model/runtime/integrity error aborts this experiment with preserved raw evidence; no free repair, retry or silent replacement. Physical-gate or target failure remains a measured candidate.',
        planned_budget={'candidate_trials':96,'calibration_trials':4,'physical_trials':100,
                        'control_steps':20000,'physics_steps':400000,'model_proposal_responses':64},
        quality_diagnostics=copy.deepcopy(old['quality_diagnostics']),
        training_admission=False,
        calibration_payload_policy='Only observed task-independent calibration fields; no feature_errors, per_dimension_pass, intent_passed, utility, utility_l2 or wall_seconds in planner baseline.',
        supersedes_quarantined={'path':str(ROOT/'artifacts/runs/agentic/tool_selection_development_v1/abort.json'),'sha256':sha256(ROOT/'artifacts/runs/agentic/tool_selection_development_v1/abort.json')},
        limitations=[
            'Tools and search design were developed using these same sources/tasks; results are development evidence only.',
            'Four related tasks over two source files, same body and pretrained motor. Not four independent sources or people.',
            'Tool choice and parameter optimization are jointly measured; no oracle correct-tool labels are assumed.',
            'Equal proposal turns and physics slots do not match tokens, context size, compute or billed costs.',
            'LLM isolation relies on separate context and explicit file-access instructions, not OS sandbox enforcement.',
            'All audio is offline through4s; feedback comes only from previous completed own trials.',
            'No old candidate labels or fitted models imported; engineering knowledge of prior failures influenced tool design.',
            'Reference curves, PD target increments, actual acceleration, physical gates and perceived quality remain distinct.',
            'No new weights, learning release, live recovery, general body transfer or product-quality acceptance.'])
    for name,digest in pins.items():
        target=out/'protocol_source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        if sha256(target)!=digest: raise ValueError('source changed during freeze')
    write_new_json(out/'protocol.json',protocol)
    frozen=read(ROOT/'artifacts/models/learned_rhythm/v2/freeze.json')['sha256']
    write_new_json(out/'protected_hashes_before.json',{name:sha256(ROOT/name) for name in frozen})
    for source in ('wave','high5'):
        task=next(t for t in tasks if t['source_id']==source)
        for local in (False,True):
            candidate=('calibration_local_' if local else 'calibration_')+source
            proposal=dict(candidate_id=candidate,method='calibration',task_id=task['task_id'],source_id=source,
                source_motion=task['source_motion'],trial_index=0,observation_id=candidate,
                control_tool='local_clock_1p4_2p8_v1' if local else 'global_clock_v1',
                parameters=dict(zip(protocol['parameter_names'],protocol['anchor'])),
                duration_s=4.,source_start_s=0.,section_windows_s=protocol['section_windows_s'],
                audio={'duration_s':4.,'observed_through_sample':88200},
                protocol_sha256=sha256(out/'protocol.json'),created_at_utc=datetime.now(timezone.utc).isoformat(),
                brief_reason='Fresh identity calibration; cross-tool parity required before any candidate.')
            write_new_json(out/'proposals'/f'{candidate}.json',proposal)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(initialize(parser.parse_args().output))
