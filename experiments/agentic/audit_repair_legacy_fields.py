"""Read-only disclosure audit; never imports or executes the physics runtime.

The only writes are new report/annotation files in a separate output directory.
This supplements, and does not replace, the frozen full physical artifact audit.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('trial_index', 'brief_reason', 'search_decision', 'created_at_utc')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def link(p):
    return {'path': str(Path(p).resolve()), 'sha256': digest(p)}

def recursive_keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from recursive_keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from recursive_keys(v)

def main(run, output):
    p = read(run/'protocol.json')
    selection = read(run/'selection.json')
    winners = {w['candidate_id']: w for w in selection['winners']}
    snapshots = run/'source_snapshot'
    runtime = snapshots/'experiments/agentic/execute_repair.py'
    common = snapshots/'src/pulsemorph/agentic/common_player.py'
    driver = snapshots/'experiments/agentic/repair_driver.py'
    source = runtime.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    # All named proposal-field accesses in the frozen runtime, including .get.
    accesses = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'proposal':
            key = node.slice.value if isinstance(node.slice, ast.Constant) else '<dynamic>'
            accesses.append({'line': node.lineno, 'field': key})
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'proposal'):
            key = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else '<dynamic>'
            accesses.append({'line': node.lineno, 'field': key, 'method': node.func.attr})
    materialize = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == 'materialize_reference']
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_generator')
    producer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'proposal_producer')
    errors, annotations, counts = [], [], Counter()
    inputs_before = {}
    paths = sorted((run/'trial_proposals').glob('*.json'))
    for path in paths:
        raw = read(path); trial = run/'trials'/path.stem
        copied = trial/'proposal.json'; seed_copy = trial/'source_proposal.json'
        seed = read(seed_copy); seed_path = ROOT/raw['parameter_source']['path']
        tracked = [path, copied, seed_copy, seed_path, trial/'common_reference/manifest.json']
        inputs_before.update({str(q): digest(q) for q in tracked})
        checks = {'raw_copy_byte_equal': digest(path) == digest(copied),
                  'seed_copy_matches_declared_hash': digest(seed_copy) == raw['parameter_source']['sha256'],
                  'seed_live_matches_copy': digest(seed_path) == digest(seed_copy),
                  'candidate_id_matches_trial': raw['candidate_id'] == path.stem,
                  'protocol_matches': raw['protocol_sha256'] == digest(run/'protocol.json')}
        gate = path.stem.startswith('gate__')
        condition = 'nominal' if gate else path.stem.rsplit('__', 1)[1]
        scenario = next(s for s in p['scenarios'] if s['scenario_id'] == condition)
        checks['perturbation_matches_protocol'] = raw['initial_perturbation'] == scenario['initial_perturbation']
        current_path = None; current = None
        if not gate:
            current_path = run/'proposals'/(raw['repair_candidate_id']+'.json')
            current = read(current_path); inputs_before[str(current_path)] = digest(current_path)
            checks['current_identity_task_method_tool_parameters'] = (
                raw['repair_candidate_id'] == current['candidate_id']
                and all(raw[k] == current[k] for k in ('task_id','method','control_tool'))
                and raw['parameters'] == dict(zip(p['parameter_names'], current['parameters'])))
            response_path = Path(current['response']['path'])
            response = read(response_path); inputs_before[str(response_path)] = digest(response_path)
            checks['current_response_hash_tool_parameters_reason'] = (
                digest(response_path) == current['response']['sha256']
                and response['control_tool'] == current['control_tool']
                and response['parameters'] == current['parameters']
                and response['brief_reason'] == current['brief_reason'])
            winner = winners.get(current['candidate_id'])
            checks['confirmation_is_frozen_winner'] = not path.stem.startswith('confirmation__') or (
                winner is not None and all(current[k] == winner[k] for k in
                    ('candidate_id','task_id','method','control_tool','parameters')))
        else:
            checks['gate_identity_parameters_from_seed'] = (
                all(raw[k] == seed[k] for k in ('task_id','control_tool','parameters','source_motion'))
                and raw['method'] == 'surrogate_search')
        checks['source_and_runtime_inputs_preserved'] = all(raw[k] == seed[k] for k in
            ('source_motion','source_id','duration_s','source_start_s','section_windows_s','audio'))
        cm = read(trial/'common_reference/manifest.json')
        checks['no_legacy_field_names_in_common_manifest'] = not set(FIELDS).intersection(recursive_keys(cm))
        fields = {}
        for field in FIELDS:
            present = field in raw
            inherited = present and field in seed and raw[field] == seed[field]
            mismatch = present and current is not None and field in current and raw[field] != current[field]
            counts[field+':present'] += present
            counts[field+':inherited_seed'] += inherited
            counts[field+':different_from_current'] += mismatch
            counts[field+':inherited_null'] += inherited and raw[field] is None
            fields[field] = {'present': present, 'inherited_seed': inherited,
                'raw_value': raw.get(field), 'current_value': current.get(field) if current else None,
                'current_field_present': current is not None and field in current,
                'different_from_current': mismatch,
                'interpretation': ('legacy_seed_disclosure_not_current_execution_time' if field == 'created_at_utc'
                    else 'legacy_seed_field_not_authoritative_for_repair_proposal') if inherited else 'absent_or_not_inherited'}
        inherited_other = [k for k in seed if k not in FIELDS and k in raw and seed[k] == raw[k]]
        for key, passed in checks.items():
            counts['checks:'+key] += bool(passed)
            if not passed: errors.append(path.stem+': '+key)
        counts['gate_trials' if gate else 'scored_trials'] += 1
        annotations.append({'trial_id': path.stem, 'phase': 'gate' if gate else path.stem.split('__')[0],
            'raw_proposal': link(path), 'copied_proposal': link(copied),
            'seed_source': raw['parameter_source'], 'preserved_seed_snapshot': link(seed_copy),
            'current_high_level_proposal': link(current_path) if current_path else None,
            'repair_candidate_id': raw['repair_candidate_id'], 'condition_id': condition,
            'checks': checks, 'legacy_fields': fields, 'other_equal_seed_fields': inherited_other})
    for field in FIELDS:
        if field in source or field in common.read_text(encoding='utf-8-sig'):
            errors.append('legacy field mentioned in runtime/common source: '+field)
    if len(paths) != 250: errors.append('expected exactly 250 execution proposals')
    # Dynamic access in validate_repair_proposal is explicitly a bounded loop over
    # ('task_id','source_motion'); keep its exact frozen source for inspection.
    validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_repair_proposal')
    source_evidence = {'runtime': link(runtime), 'driver': link(driver), 'common_player': link(common),
        'runtime_proposal_accesses': accesses, 'materialize_calls': materialize,
        'build_generator': ast.unparse(build), 'producer': ast.unparse(producer),
        'validate_repair_proposal': ast.unparse(validate),
        'legacy_field_literal_mentions_in_runtime_and_common': [],
        'manual_dataflow_read': 'proposal object is passed only to local validation/source/producer/build functions; generator receives current validated parameters and explicit source/time/section inputs. materialize_reference receives explicit identity/source/audio/protocol/producer arguments; it never receives proposal, trial_index, brief_reason, search_decision, or created_at_utc. Raw proposal bytes/hash remain in evidence, so provenance is affected even though numeric inputs/common semantic metadata are bounded.'}
    changed = [q for q, old in inputs_before.items() if digest(q) != old]
    if changed: errors.append('input changed during disclosure audit')
    report = {'schema': 'pulsemorph.repair_legacy_fields_disclosure.v1',
        'status': 'provenance_incomplete_disclosure', 'bounded_impact_checks_passed': not errors,
        'raw_metadata_semantically_correct': False, 'errors': errors, 'counts': dict(counts),
        'protocol': link(run/'protocol.json'), 'selection': link(run/'selection.json'),
        'inputs_unchanged_during_review': not changed, 'checked_input_file_count': len(inputs_before),
        'source_evidence': source_evidence,
        'scope': 'No physics/model invocation, rescore, rerank, raw-proposal repair, or common-package rewrite. This is a provenance exception disclosure, not full audit acceptance.',
        'bug': 'trial_proposal deep-copies seed and overwrites current numeric/identity fields but leaves legacy descriptive/index/time fields. parameter_source identifies seed initialization, not authoritative current repair decision. Follow repair_candidate_id to the separate current high-level proposal and its response/observation hashes.'}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('report.json', report), ('annotations.json', annotations)]:
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'output':str(output),'counts':dict(counts),'errors':errors}, indent=2))
    return bool(errors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/runs/agentic/repair_development_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/runs/agentic/repair_legacy_fields_review_v1')
    args = parser.parse_args()
    raise SystemExit(main(args.run, args.output))
