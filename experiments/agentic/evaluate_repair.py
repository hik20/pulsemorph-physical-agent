"""Repair evaluation with an exact, declared wrapper-only baseline pin bridge."""
from __future__ import annotations
import copy
import types
import math
import json
import hashlib
from pathlib import Path
import evaluate_robustness_v2 as native


def bridge_pins(trial, baseline, switch):
    """Return trial pins in baseline namespace only after exact bridge checks."""
    def normalized_keys(pins):
        result = {}
        for key, digest in pins.items():
            if not isinstance(key, str):
                raise ValueError('pin path must be a string')
            normalized = key.replace('\\', '/')
            if normalized in result:
                raise ValueError('normalized pin path collision')
            result[normalized] = digest
        return result
    trial, baseline = normalized_keys(trial), normalized_keys(baseline)
    if not isinstance(switch, dict) or set(switch) != {'old', 'new'}:
        raise ValueError('exact repair_wrapper_pin_switch declaration required')
    for label, expected in [('old', 'experiments/agentic/execute_robustness.py'),
                            ('new', 'experiments/agentic/execute_repair.py')]:
        record = switch[label]
        if (not isinstance(record, dict) or set(record) != {'path', 'sha256'}
                or record['path'] != expected or not isinstance(record['sha256'], str)
                or len(record['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in record['sha256'])):
            raise ValueError('invalid wrapper pin bridge '+label)
    old, new = switch['old'], switch['new']
    if (trial.get(new['path']) != new['sha256'] or old['path'] in trial
            or baseline.get(old['path']) != old['sha256'] or new['path'] in baseline):
        raise ValueError('declared wrapper pins do not match runtime evidence')
    normalized = dict(trial)
    del normalized[new['path']]
    normalized[old['path']] = old['sha256']
    if normalized != baseline:
        raise ValueError('non-wrapper dependency pin mismatch')
    return normalized


def evaluate_repair(directory, nominal_baseline, condition_baseline, protocol, goal):
    """Run every legacy integrity/scoring rule after a narrowly audited bridge.

    Each raw artifact is audited before adapting an isolated in-memory view.
    No manifest, source, threshold, array or recorded observation is modified.
    """
    original = native._original
    base = original._base
    paths = list(map(str, (directory, nominal_baseline, condition_baseline)))
    audits = [base._audit(path) for path in (directory, nominal_baseline, condition_baseline)]
    if audits[0]['manifest'].get('execution_kind') != 'robust_initial_state_repair':
        raise ValueError('repair execution evidence required')
    proposal = json.loads((Path(directory)/'proposal.json').read_bytes())
    seed_bytes = (Path(directory)/'source_proposal.json').read_bytes()
    seed = json.loads(seed_bytes)
    origin = proposal.get('parameter_source')
    if (not isinstance(origin, dict) or set(origin) != {'path','sha256'}
            or hashlib.sha256(seed_bytes).hexdigest() != origin['sha256']
            or audits[0]['manifest'].get('parameter_source') != origin
            or proposal.get('execution_kind') != 'robust_initial_state_repair'
            or any(proposal.get(key) != seed.get(key) for key in ('task_id','source_motion'))):
        raise ValueError('repair seed provenance mismatch')
    switch = protocol.get('repair_wrapper_pin_switch')
    trial_pins = audits[0]['manifest']['preexecution_input_hashes']
    for baseline in audits[1:]:
        bridge_pins(trial_pins, baseline['manifest']['preexecution_input_hashes'], switch)
    adapted = copy.deepcopy(audits[0])
    adapted['manifest']['preexecution_input_hashes'] = audits[1]['manifest']['preexecution_input_hashes']
    lookup = dict(zip(paths, [adapted, *audits[1:]]))
    proxy = types.SimpleNamespace(**vars(base))
    proxy._audit = lambda path: lookup[str(path)]
    namespace = {**original.evaluate_robustness.__globals__, '_base': proxy}
    evaluator = types.FunctionType(original.evaluate_robustness.__code__, namespace)
    result = native.to_json_native(evaluator(directory, nominal_baseline, condition_baseline, protocol, goal))
    result.update(schema='pulsemorph.agentic_repair_evaluation.v1',
                  wrapper_pin_bridge=switch,
                  limitation='Observed joint proxies under declared startup conditions; no recovery, aesthetics or probability claim.')
    return result


def aggregate_conditions(evaluations, declared_condition_ids):
    """Strict complete-condition aggregate; missing accent cannot receive a score."""
    declared = list(declared_condition_ids)
    if (not declared or len(set(declared)) != len(declared)
            or set(evaluations) != set(declared)):
        raise ValueError('exact complete declared condition set required')
    rows = [evaluations[key] for key in declared]
    valid = all(row.get('integrity_passed') is True for row in rows)
    utilities = [row.get('primary', {}).get('utility') if row.get('primary') else None for row in rows]
    available = valid and all(type(value) in (int, float) and math.isfinite(value) for value in utilities)
    squared = [row.get('primary', {}).get('utility_l2') if row.get('primary') else None for row in rows]
    squared_available = valid and all(type(value) in (int, float) and math.isfinite(value) for value in squared)
    return dict(schema='pulsemorph.agentic_repair_condition_aggregate.v1',
                condition_ids=declared, integrity_passed=valid,
                utility=max(utilities) if available else None,
                utility_l2=sum(squared) if squared_available else None,
                utility_definition='maximum legacy primary normalized feature error across all declared conditions',
                physical_gate_passed=valid and all(row.get('physical_gate_passed') is True for row in rows),
                joint_success=available and all((row.get('primary') or {}).get('joint_success') is True for row in rows),
                condition_utilities=dict(zip(declared, utilities)))


def selection_key(aggregate, candidate_id):
    """Physical gate first, worst error, total squared error, stable ID last."""
    if aggregate.get('integrity_passed') is not True:
        raise ValueError('integrity-passed aggregate required for selection')
    if not isinstance(candidate_id, str):
        raise ValueError('string candidate ID required')
    # Missing accent is a measured target failure. Infinity is ranking-only and
    # must never be included in serialized aggregate/evaluation artifacts.
    utility = aggregate.get('utility')
    squared = aggregate.get('utility_l2')
    return (not aggregate['physical_gate_passed'],
            float('inf') if utility is None else utility,
            float('inf') if squared is None else squared, candidate_id)
