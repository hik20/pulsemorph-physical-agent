"""Versioned replay plan after the preserved v1 JSON serialization abort.

The 13-by-six experiment is unchanged. This module does not execute physics.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import robustness_plan as base

ROOT = base.ROOT
FAILED = Path('artifacts/runs/agentic/robustness_initial_state_v1')
QUARANTINE = Path('artifacts/runs/agentic/robustness_initial_state_v1_quarantine_v1')
NEW = ('robustness_plan_v2.py', 'robustness_driver_v2.py',
       'evaluate_robustness_v2.py', 'audit_robustness_v2.py')


def quarantine():
    """Record actual abort cost outside the unchanged failed panel, once."""
    target = ROOT/QUARANTINE
    if target.exists():
        raise FileExistsError(target)
    summary = base.read(ROOT/FAILED/'summary.json')
    if summary.get('panel_completed') is not False:
        raise ValueError('only an aborted v1 panel may be quarantined')
    manifests = sorted((ROOT/FAILED/'trials').glob('*/manifest.json'))
    actual = [base.read(path) for path in manifests]
    counts = dict(physical_trials=len(actual), control_steps=sum(m.get('actual_control_steps',0) for m in actual),
                  physics_steps=sum(m.get('actual_physics_steps',0) for m in actual), new_llm_calls=0,
                  nonzero_physical_trials=sum(not path.parent.name.startswith('nominal__') for path in manifests))
    if counts != dict(physical_trials=3,control_steps=600,physics_steps=12000,new_llm_calls=0,nonzero_physical_trials=0):
        raise ValueError('unexpected preserved v1 actual counts')
    if any(m.get('new_llm_calls') != 0 for m in actual):
        raise ValueError('v1 manifest model-call accounting differs')
    protocol = base.read(ROOT/FAILED/'protocol.json')
    inventory = base.protected_inventory(FAILED)
    source_pins = {path:dict(expected=digest,actual=base.sha(ROOT/path),unchanged=base.sha(ROOT/path)==digest)
                   for path,digest in protocol['execution_code_sha256'].items()}
    result = dict(schema='pulsemorph.agentic_initial_state_robustness_quarantine.v1',
        failed_panel=str(FAILED.as_posix()), failed_protocol=base.link(FAILED/'protocol.json'),
        failed_summary=base.link(FAILED/'summary.json'), failure_traceback=base.link(FAILED/'failure_traceback.txt'),
        reason='v1 evaluator returns numpy.bool_ values rejected by JSON serializer after three nominal physics runs',
        actual_counts=counts, completed_panel=False, nominal_gate_completed=False,
        evaluation_json_present=False, serialization_failed_before_evaluation_publication=True,
        measured_target_results_admitted=False, training_admission=False,
        source_pins=source_pins, all_frozen_sources_unchanged=all(row['unchanged'] for row in source_pins.values()),
        manifest_links=[base.link(path.relative_to(ROOT)) for path in manifests],
        inventory_file='inventory.json', new_physics_during_quarantine=0, new_llm_calls_during_quarantine=0)
    target.mkdir(parents=True,exist_ok=False)
    base.write_new(target/'inventory.json', inventory['files'])
    base.write_new(target/'summary.json', result)
    # Read-only audit must not itself mutate the failed panel.
    if base.protected_inventory(FAILED) != inventory:
        raise ValueError('failed panel changed during quarantine creation')
    return result


def build_protocol():
    quarantine_path=ROOT/QUARANTINE/'summary.json'
    record=base.read(quarantine_path)
    if record['actual_counts']['physical_trials'] != 3 or not record['all_frozen_sources_unchanged']:
        raise ValueError('accepted preserved v1 quarantine required')
    result=base.build_protocol()
    result['schema']='pulsemorph.agentic_initial_state_robustness_protocol.v2'
    result['supersedes_failure']=base.link(FAILED/'summary.json')
    result['quarantine']=base.link(QUARANTINE/'summary.json')
    result['quarantine_inventory']=base.link(QUARANTINE/'inventory.json')
    result['protected_inventories'].append(base.protected_inventory(FAILED))
    result['protected_evidence']=[dict(path=item['path'],inventory_file='protected_inventories/'+str(i)+'.json')
                                  for i,item in enumerate(result['protected_inventories'])]
    for name in NEW:
        path='experiments/agentic/'+name
        result['execution_code_sha256'][path]=base.sha(ROOT/path)
    result['serialization_fix']='Versioned JSON-native evaluator wrapper only; fixed candidates, panel, scoring and thresholds unchanged.'
    result['quarantined_cost_excluded_from_this_panel']=record['actual_counts']
    return result


def audit_quarantined_traces():
    """Separate raw trace confirmation; does not publish target evaluations."""
    from evaluate_generalization import _audit
    from audit_robustness import check_input_pins
    from run_tool_compatibility_smoke import trace_parity
    panel=ROOT/FAILED
    before=base.protected_inventory(FAILED)
    protocol=base.read(panel/'protocol.json')
    rows=[]
    for manifest_path in sorted((panel/'trials').glob('*/manifest.json')):
        trial=manifest_path.parent
        role=trial.name.removeprefix('nominal__')
        case=next(c for c in protocol['cases'] if c['role']==role)
        original=ROOT/case['original_trial']
        audited=_audit(trial)
        rows.append(dict(trial_id=trial.name, raw_integrity_audit_passed=True,
            full_trace_parity=trace_parity(original,trial),
            input_pin_parity=check_input_pins(base.read(original/'manifest.json'),audited['manifest'])))
    unchanged=base.protected_inventory(FAILED)==before
    result=dict(schema='pulsemorph.agentic_quarantined_nominal_trace_audit.v1',
        scope='Three existing raw nominal traces only; not a completed six-case gate or target evaluation.',
        traces=rows, audit_passed=len(rows)==3 and unchanged, failed_panel_unchanged=unchanged,
        target_evaluations_published=False,new_physical_trials=0,new_llm_calls=0,
        source=base.link(Path(__file__).relative_to(ROOT)))
    base.write_new(ROOT/QUARANTINE/'raw_trace_audit.json',result)
    return result


def freeze(output):
    """Exclusive v2 publication; no overwrite, retry, physics or model call."""
    output=Path(output).resolve()
    if (ROOT/'artifacts/runs/agentic').resolve() not in output.parents:
        raise ValueError('panel output must be strictly under agentic runs')
    if output.exists():
        raise FileExistsError(output)
    protocol=build_protocol()
    output.mkdir(parents=True,exist_ok=False)
    (output/'proposals').mkdir()
    for relative,digest in protocol['execution_code_sha256'].items():
        source=ROOT/relative; content=source.read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:
            raise ValueError('code changed before v2 snapshot: '+relative)
        target=output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        if base.sha(target)!=digest or base.sha(source)!=digest:
            raise ValueError('code changed during v2 snapshot: '+relative)
    (output/'protected_inventories').mkdir()
    for index,item in enumerate(protocol['protected_inventories']):
        base.write_new(output/'protected_inventories'/(str(index)+'.json'),item['files'])
    base.write_new(output/'protocol.json',protocol)
    digest=base.sha(output/'protocol.json')
    for scenario in protocol['scenarios']:
        for case in protocol['cases']:
            source=ROOT/case['original_proposal']['path']
            if base.sha(source)!=case['original_proposal']['sha256']:
                raise ValueError('original proposal changed during v2 freeze')
            value=base.replay_proposal(base.read(source),case,scenario,digest)
            base.write_new(output/'proposals'/(value['candidate_id']+'.json'),value)
    return protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quarantine',action='store_true',required=True)
    args=parser.parse_args()
    quarantine()
