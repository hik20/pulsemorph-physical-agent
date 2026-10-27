"""V2 JSON-native evaluation; frozen, once-only initial-state panel. No optimizer or model requests.

The six nominal replays must match their original traces before perturbations.
Technical failures halt and preserve evidence; measured target failures do not.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from musical_tasks import ROOT, read, sha256, write_new_json
from execute_robustness import execute_trial
from evaluate_generalization import _audit
from evaluate_robustness_v2 import evaluate_robustness
from audit_multifeature import inventory
from audit_robustness_v2 import check_input_pins, calibration_measurements
from run_tool_compatibility_smoke import trace_parity, array_archive_parity


def require(value, message):
    if not value:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def journal(panel, event, **fields):
    with (panel/'execution_journal.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(dict(event=event, time_utc=stamp(), **fields), allow_nan=False)+'\n')
        stream.flush()


def check_frozen(panel, protocol, protocol_hash):
    require(sha256(panel/'protocol.json') == protocol_hash, 'frozen protocol changed')
    for name, digest in protocol['execution_code_sha256'].items():
        require(sha256(ROOT/name) == digest, 'frozen source changed: '+name)


def require_nominal_gate(panel, case_count=6, protocol_hash=None):
    gate = read(panel/'nominal_parity.json')
    require(gate['passed'] is True and len(gate['cases']) == case_count,
            'all six nominal byte parity checks must pass before perturbations')
    require(len({row['role'] for row in gate['cases']}) == case_count, 'duplicate nominal role')
    require({row['role'] for row in gate['cases']} == {'calibration_wave','calibration_high5',
        'selected_wave_a','selected_wave_b','selected_high5_a','selected_high5_b'}, 'incorrect nominal roles')
    require(all(row['passed'] is True for row in gate['cases']), 'failed nominal role')
    if protocol_hash is not None:
        require(gate['protocol_sha256'] == protocol_hash, 'nominal gate protocol mismatch')
    return gate


def summarize_rows(rows):
    result = []
    tasks = ['wave_a', 'wave_b', 'high5_a', 'high5_b']
    for task in tasks:
        relevant = [r for r in rows if r['task_id'] == task]
        perturbed = [r for r in relevant if r['scenario_id'] != 'nominal']
        nominal = [r for r in relevant if r['scenario_id'] == 'nominal']
        result.append(dict(task_id=task, nominal_primary_passed=(nominal[0]['primary']['joint_success'] if nominal else None),
            nonzero_cases=len(perturbed), primary_passed=sum(r['primary']['joint_success'] for r in perturbed),
            secondary_passed=sum(r['secondary']['joint_success'] for r in perturbed),
            physical_passed=sum(r['physical_gate_passed'] for r in perturbed),
            differing_success_cases=[r['scenario_id'] for r in perturbed if r['primary']['joint_success'] != r['secondary']['joint_success']],
            primary_failures=[r['scenario_id'] for r in perturbed if not r['primary']['joint_success']]))
    return result


def run(panel):
    from robustness_plan_v2 import freeze
    panel = Path(panel).resolve()
    protocol = freeze(panel)
    protocol_hash = sha256(panel/'protocol.json')
    summary = dict(schema='pulsemorph.agentic_robustness_summary.v1', panel_completed=False,
        errors=[], rows=[], tasks=[], new_llm_calls=0, training_admission=False,
        panel_scope='Deterministic initial-state sensitivity of four fixed development outputs over two sources; no probability, recovery or perceived quality claim.',
        protocol_sha256=protocol_hash)
    nominal_rows = []
    goals = {t['task_id']: t['goal'] for t in protocol['tasks']}
    try:
        journal(panel, 'panel_start', protocol_sha256=protocol_hash)
        for scenario in protocol['scenarios']:
            scenario_id = scenario['scenario_id']
            if scenario_id != 'nominal':
                require_nominal_gate(panel, protocol_hash=protocol_hash)
            for case in protocol['cases']:
                check_frozen(panel, protocol, protocol_hash)
                identity = scenario_id+'__'+case['role']
                proposal_path = panel/'proposals'/(identity+'.json')
                proposal = read(proposal_path)
                require(proposal['initial_perturbation'] == scenario['initial_perturbation'], 'proposal perturbation changed')
                require(proposal['parameter_source'] == case['original_proposal'], 'proposal source changed')
                require(proposal['protocol_sha256'] == protocol_hash, 'proposal protocol changed')
                trial = panel/'trials'/identity
                journal(panel, 'trial_start', trial_id=identity, proposal_sha256=sha256(proposal_path))
                manifest = execute_trial(proposal_path, trial)
                journal(panel, 'trial_complete', trial_id=identity, status=manifest['status'], manifest_sha256=sha256(trial/'manifest.json'),
                    control_steps=manifest.get('actual_control_steps',0), physics_steps=manifest.get('actual_physics_steps',0))
                require(manifest['status'] == 'success', identity+': '+str(manifest.get('error')))
                require(manifest.get('strict_g1_admission_passed') is True and manifest.get('strict_g1_preflight_count') == 2,
                        'strict G1 admission missing: '+identity)
                require((manifest['actual_control_steps'],manifest['actual_physics_steps']) == (200,4000), 'execution count mismatch')
                original = Path(case['original_trial'])
                if not original.is_absolute(): original = ROOT/original
                pin_parity = check_input_pins(read(original/'manifest.json'), manifest)
                array_archive_parity(original/'common_reference/arrays.npz', trial/'common_reference/arrays.npz')
                if case['role'].startswith('calibration_'):
                    measured = calibration_measurements(_audit(trial))
                    write_new_json(panel/'calibrations'/(identity+'.json'), measured)
                    if scenario_id == 'nominal':
                        require(measured['min_root_height_m'] >= protocol['physical_gate']['min_root_height_m']
                            and measured['mean_joint_rmse_rad'] <= protocol['physical_gate']['max_mean_joint_rmse_rad'],
                            'nominal calibration physical gate failed: '+identity)
                else:
                    nominal_baseline = panel/'trials'/('nominal__calibration_'+case['source_id'])
                    condition_baseline = panel/'trials'/(scenario_id+'__calibration_'+case['source_id'])
                    evaluation = evaluate_robustness(trial, nominal_baseline, condition_baseline, protocol, goals[case['task_id']])
                    write_new_json(panel/'evaluations'/(identity+'.json'), evaluation)
                    require(evaluation['integrity_passed'] is True, identity+': '+str(evaluation['integrity_errors']))
                    summary['rows'].append(dict(trial_id=identity, task_id=case['task_id'], source_id=case['source_id'], scenario_id=scenario_id,
                        primary=evaluation['primary'], secondary=evaluation['secondary'], physical_gate_passed=evaluation['physical_gate_passed'],
                        min_root_height_m=evaluation['min_root_height_m'], mean_joint_rmse_rad=evaluation['mean_joint_rmse_rad']))
                if scenario_id == 'nominal':
                    parity = trace_parity(original, trial)
                    nominal_rows.append(dict(role=case['role'], trial_id=identity, passed=True, trace_parity=parity, input_pin_parity=pin_parity))
                print(json.dumps(dict(trial_id=identity, status='audited', completed=len(list((panel/'trials').glob('*/manifest.json'))))), flush=True)
            if scenario_id == 'nominal':
                require(len(nominal_rows) == 6, 'six nominal comparisons required')
                write_new_json(panel/'nominal_parity.json', dict(passed=True, cases=nominal_rows, protocol_sha256=protocol_hash))
                journal(panel, 'nominal_gate', passed=True, nominal_parity_sha256=sha256(panel/'nominal_parity.json'))
        require(len(summary['rows']) == 52, 'exactly52 selected results required')
        summary['panel_completed'] = True
    except Exception as exc:
        summary['errors'].append(str(exc))
        journal(panel, 'panel_abort', error=str(exc))
        (panel/'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
    finally:
        manifests = [read(p) for p in (panel/'trials').glob('*/manifest.json')]
        summary.update(physical_trials=len(manifests), control_steps=sum(m.get('actual_control_steps',0) for m in manifests),
            physics_steps=sum(m.get('actual_physics_steps',0) for m in manifests), tasks=summarize_rows(summary['rows']))
        summary['source_pins_unchanged'] = all(sha256(ROOT/name) == digest for name,digest in protocol['execution_code_sha256'].items())
        # Original evidence protection is finalized by the independent full audit.
        if not summary['source_pins_unchanged']:
            summary['panel_completed'] = False
            summary['errors'].append('frozen source pins changed')
        journal(panel, 'panel_complete', completed=summary['panel_completed'], physical_trials=summary['physical_trials'])
        write_new_json(panel/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({key:result[key] for key in ('panel_completed','errors','physical_trials','control_steps','physics_steps','new_llm_calls','tasks')}))
    raise SystemExit(0 if result['panel_completed'] else 1)
