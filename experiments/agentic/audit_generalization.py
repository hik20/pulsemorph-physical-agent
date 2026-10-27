"""Post-completion read-only evidence audit; no simulation or model calls.

Local UTC/filesystem chronology is provenance evidence, not signed attestation.
Recorded model reasons and route declarations are not provider token telemetry.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from audit_multifeature import read, sha, require, inventory, linked, equal, compact, common_parity
from evaluate_generalization import evaluate_generalization
from generalization_search import next_candidate
from musical_tasks import analyze_waveform
from pulsemorph.motion.contracts import load_reference

METHODS = ['surrogate_search', 'llm_feedback', 'llm_feedback_off']
TASKS = {'wave_a', 'wave_b', 'high5_a', 'high5_b'}
METRICS = ['features', 'feature_errors', 'per_dimension_pass', 'intent_passed',
           'physical_gate_passed', 'utility', 'utility_l2', 'mean_joint_rmse_rad',
           'min_root_height_m', 'quality_diagnostics']


def cid(task, method, index):
    return f'{task}__{method}__{index}'


def source_equal(left, right):
    return (set(left) == set(right) == {'path', 'sha256'} and left['sha256'] == right['sha256']
            and left['path'].replace('\\', '/') == right['path'].replace('\\', '/'))


def check_waveform(pilot, task):
    """Recompute PCM observations; authored onset remains separate provenance."""
    folder = pilot / 'music' / task['task_id']
    path = folder / 'music.wav'
    declared = task['music_observation']
    require(Path(declared['waveform_path']).resolve() == path.resolve(), 'waveform path identity')
    require(declared['waveform_sha256'] == sha(path), 'waveform digest mismatch')
    measured = analyze_waveform(path)
    require(measured == declared == read(folder / 'waveform_observation.json'), 'recomputed PCM observation mismatch')
    require(measured['observed_through_sample'] == 88200 and measured['duration_s'] == 4., 'four-second waveform access')
    event = next(e for e in measured['measured_bass_events'] if e['event_id'] == measured['strongest_bass_event_id'])
    require(task['goal']['feature_targets'][4] == event['time_s'] and task['goal']['cue_event_id'] == event['event_id'], 'goal must use measured strongest event')
    truth = read(folder / 'authored_truth.json')
    error = event['time_s'] - truth['authored_primary_cue_s']
    require(task['authored_cue_observation_error_s'] == error, 'separate authored onset error mismatch')
    return dict(waveform_sha256=sha(path), measured_strongest_event=event,
                authored_primary_cue_s=truth['authored_primary_cue_s'], measured_minus_authored_onset_s=error,
                canonical_timing_target_source='recomputed measured waveform event; authored onset reported separately')


def check_model_receipt(pilot, proposal):
    identity = proposal['candidate_id']
    receipt = read(pilot / 'model_calls' / (identity + '.json'))
    for key in ('candidate_id', 'task_id', 'method', 'trial_index'):
        require(receipt[key] == proposal[key], 'model receipt stream identity ' + key)
    require(receipt['requested_model'] == 'gpt-6.1-sol' and receipt['fork_turns'] == 'none', 'orchestrated model route')
    require(isinstance(receipt['agent_name'], str) and bool(receipt['agent_name']), 'model agent identity')
    require(receipt['observation_sha256'] == proposal['observation_artifact']['sha256'] and
            receipt['response_sha256'] == proposal['response_artifact']['sha256'], 'model receipt artifact linkage')
    require(receipt['receipt_scope'] == 'orchestrator-observed collaboration turn; not provider billing or token receipt', 'receipt scope disclosure')
    require(datetime.fromisoformat(receipt['completed_at_utc']).tzinfo is not None, 'receipt timezone required')
    return receipt['agent_name']


def check_observation(pilot, proposal, protocol, task, results):
    """Check exact own-history contents and forbid undeclared observation fields."""
    method, index = proposal['method'], proposal['trial_index']
    path = linked(pilot, proposal['observation_artifact'])
    obs = read(path)
    identity = cid(task['task_id'], method, index)
    require(path == (pilot / 'observations' / (identity + '.json')).resolve(), 'individual observation path')
    require(obs['observation_id'] == proposal['observation_id'] == identity, 'observation identity')
    require(obs['task_id'] == task['task_id'] and obs['method'] == method, 'cross-stream observation')
    require(obs['protocol_sha256'] == sha(pilot / 'protocol.json'), 'observation protocol hash')
    expected_history = [] if method == 'llm_feedback_off' else [compact(results[cid(task['task_id'], method, j)]) for j in range(index)]
    require(obs['prior_trial_results'] == expected_history, 'outcome history exposure mismatch')
    if method == 'surrogate_search':
        require(set(obs) == {'observation_id', 'task_id', 'method', 'prior_trial_results', 'protocol_sha256', 'proposal_compute_seconds'}, 'surrogate observation fields')
        rows = [results[cid(task['task_id'], method, j)] for j in range(index)]
        equal([proposal['parameters'][n] for n in protocol['parameter_names']], next_candidate(rows, protocol, task['goal']), 'deterministic search replay')
        return None
    fields = {'schema', 'observation_id', 'task_id', 'method', 'trial_index', 'protocol_sha256', 'source_id', 'source_motion',
              'user_instruction', 'canonical_goal', 'baseline', 'parameter_names', 'bounds', 'feature_names', 'feature_tolerances',
              'physical_gate', 'selection', 'music_observation', 'prior_proposals', 'prior_trial_results', 'trials_remaining',
              'forced_first_parameters', 'tool', 'access', 'response_schema', 'instructions'}
    require(set(obs) == fields, 'undeclared model observation fields')
    require(obs['schema'] == 'pulsemorph.agentic_generalization_observation.v1' and obs['trial_index'] == index, 'observation schema/index')
    for key in ('source_id', 'source_motion', 'music_observation'):
        require(obs[key] == task[key], 'observation task content ' + key)
    require(obs['user_instruction'] == task['instruction'] and obs['canonical_goal'] == task['goal'], 'observation instruction/goal')
    for key in ('parameter_names', 'bounds', 'feature_names', 'feature_tolerances', 'physical_gate', 'selection'):
        require(obs[key] == protocol[key], 'observation protocol content ' + key)
    require(obs['baseline'] == compact(results['calibration_' + task['source_id']]), 'baseline observation mismatch')
    previous = []
    for j in range(index):
        previous_id = cid(task['task_id'], method, j)
        p = read(pilot / 'proposals' / (previous_id + '.json'))
        previous.append(dict(candidate_id=previous_id, parameters=[p['parameters'][n] for n in protocol['parameter_names']],
                             proposal_sha256=sha(pilot / 'proposals' / (previous_id + '.json'))))
    require(obs['prior_proposals'] == previous, 'prior proposal history mismatch')
    require(obs['trials_remaining'] == 8 - index and obs['forced_first_parameters'] == (protocol['anchor'] if index == 0 else None), 'budget/anchor disclosure')
    require(obs['tool']['section_windows_s'] == protocol['section_windows_s'] and obs['tool']['measurement_windows_s'] == protocol['feature_windows_s'], 'tool windows')
    expected_tool = dict(section_windows_s=protocol['section_windows_s'], measurement_windows_s=protocol['feature_windows_s'],
        gains='Independent per-side initial-posture displacement gains with sin-squared section envelopes. Achieved RMS ratios differ from gains.',
        clock='source_time=t+timing_shift_s*sin(pi*t/4)^2 inside 0<t<4; identity outside; analytic chain-rule velocity.',
        output='Validated MotionReference -> fixed pretrained G1 ONNX -> PD -> MuJoCo. One offline take; no learning.')
    require(obs['tool'] == expected_tool, 'unexpected nested tool observation content')
    require(obs['access'] == 'Use only this observation and your own previous supplied observations. No sibling files or code. Feedback-off has no candidate outcomes.' and
            obs['instructions'] == 'Return exactly one next vector. First vector must be forced anchor. No extra candidates.', 'unexpected model access instructions')
    response_path = linked(pilot, proposal['response_artifact'])
    require(response_path == (pilot / 'responses' / (identity + '.json')).resolve(), 'individual response path')
    raw = read(response_path)
    require(Path(raw['only_observation_path']).resolve() == path, 'raw response observation linkage')
    require(obs['response_schema'] == dict(parameters='one flat list of five numbers', only_observation_path=str(path), brief_reason='short decision summary'), 'requested response schema/linkage')
    values = raw['parameters']
    require(isinstance(values, list) and len(values) == 5 and all(type(v) in (int, float) for v in values), 'one flat response required')
    equal(values, [proposal['parameters'][n] for n in protocol['parameter_names']], 'unchanged raw model values')
    return response_path


def check_record(pilot, identity, proposal, manifest):
    record = read(pilot / 'trial_records' / (identity + '.json'))
    require(record['candidate_id'] == identity and record['task_id'] == proposal['task_id'], 'record identity')
    require(record['source_motion'] == proposal['source_motion'] and record['source_group'] == proposal['source_motion']['sha256'], 'record source group')
    require(record['training_admission'] is False and record['shared_corpus_imported'] is False, 'unexpected training admission')
    required = {'proposal': pilot / 'proposals' / (identity + '.json'), 'evaluation': pilot / 'evaluations' / (identity + '.json'),
                'reference': pilot / 'trials' / identity / 'common_reference/manifest.json',
                'execution_manifest': pilot / 'trials' / identity / 'manifest.json',
                'actual_qpos': pilot / 'trials' / identity / 'qpos.npy', 'actual_qvel': pilot / 'trials' / identity / 'qvel.npy',
                'actual_policy_inputs': pilot / 'trials' / identity / 'policy_inputs.npz',
                'applied_pd_targets': pilot / 'trials' / identity / 'applied_joint_targets.npy'}
    for key, expected in required.items():
        require(linked(pilot, record[key]) == expected.resolve(), 'record artifact linkage ' + key)
    require(record['observation'] == proposal.get('observation_artifact') and record['model_response'] == proposal.get('response_artifact'), 'record observation/response linkage')
    require(record['provider_tokens'] == (None if proposal['method'].startswith('llm') else 0), 'unknown provider token disclosure')
    require(manifest['proposal_sha256'] == sha(required['proposal']), 'manifest proposal digest')


def audit(pilot, output):
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    require(output != pilot and pilot not in output.parents, 'audit output must be outside pilot')
    # Require completion before creating even an audit output directory.
    protocol, comparison = read(pilot / 'protocol.json'), read(pilot / 'comparison.json')
    output.mkdir(parents=True, exist_ok=False)
    before = inventory(pilot)
    dependencies = [Path(__file__)] + [Path(__file__).with_name(n) for n in
        ('audit_multifeature.py', 'evaluate_generalization.py', 'evaluate_multifeature.py', 'evaluate_trial.py',
         'execute_generalization.py', 'generalization_driver.py', 'generalization_search.py')]
    dependencies += [Path(__file__).with_name('musical_tasks.py')]
    dependencies += sorted((ROOT / 'src/pulsemorph/motion').glob('*.py'))
    pins = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    report = dict(schema='pulsemorph.agentic_generalization_final_audit.v1', audit_passed=False, errors=[], trials={}, selections=[],
        source_hashes_before=pins, scope='Posthoc read-only recomputation; no physics steps, model calls, threshold changes or corpus import.',
        limitations=['Local timestamps and route declarations are not external attestations.',
                     'Access restriction consists of supplied observations, agent instructions and fork_turns=none; it is not an OS sandbox or independent provider trace.',
                     'Audit acceptance is separate from task success, expression, human preference and hardware safety.',
                     'Provider token counts and billing are unknown for model proposals.'])
    try:
        for p in dependencies:
            dest = output / 'source_snapshot' / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
        require({t['task_id'] for t in protocol['tasks']} == TASKS and len(protocol['tasks']) == 4, 'expected four exact tasks')
        require(list(protocol['methods']) == METHODS and protocol['anchor'] == [1, 1, 1, 1, 0], 'method/anchor contract')
        require(comparison['protocol_sha256'] == sha(pilot / 'protocol.json'), 'comparison protocol digest')
        sources = {t['source_id']: t['source_motion'] for t in protocol['tasks']}
        require(set(sources) == {'wave', 'high5'} and len({s['sha256'] for s in sources.values()}) == 2, 'two distinct source groups required')
        for task in protocol['tasks']:
            require(task['source_motion'] == sources[task['source_id']], 'inconsistent source mapping')
        report['waveform_verification'] = {t['task_id']: check_waveform(pilot, t) for t in protocol['tasks']}
        calibrations = ['calibration_' + s for s in sorted(sources)]
        candidates = [cid(t['task_id'], m, i) for t in protocol['tasks'] for m in METHODS for i in range(8)]
        identities = calibrations + candidates
        require({p.name for p in (pilot / 'trials').iterdir() if p.is_dir()} == set(identities), 'exact 98 trial inventory')
        for directory in ('proposals', 'evaluations', 'trial_records'):
            require({p.stem for p in (pilot / directory).glob('*.json')} == set(identities), 'exact artifact inventory ' + directory)
        model_ids = {cid(t['task_id'], m, i) for t in protocol['tasks'] for m in METHODS[1:] for i in range(8)}
        require({p.stem for p in (pilot / 'responses').glob('*.json')} == model_ids, '64 individual model responses required')
        require({p.stem for p in (pilot / 'observations').glob('*.json')} == set(candidates), '96 individual observations required')
        start = min((pilot / 'trials' / identity / 'proposal.json').stat().st_ctime_ns for identity in candidates)
        start_utc = datetime.fromtimestamp(start / 1e9).astimezone()
        for task in protocol['tasks']:
            for i in range(8):
                identity = cid(task['task_id'], 'llm_feedback_off', i)
                path = pilot / 'proposals' / (identity + '.json')
                require(path.stat().st_ctime_ns < start and datetime.fromisoformat(read(path)['created_at_utc']) < start_utc, 'OFF commitment after candidate execution')
        report['feedback_off_precommitment'] = dict(proposals=32, earliest_candidate_creation_ns=start)
        for name, digest in protocol['execution_code_sha256'].items():
            require(sha(ROOT / name) == digest, 'execution code pin changed ' + name)
            require(sha(pilot / 'protocol_source_snapshot' / name) == digest, 'protocol source snapshot changed ' + name)
        require({p.stem for p in (pilot / 'model_calls').glob('*.json')} == model_ids, '64 individual orchestrator receipts required')
        results, baseline_manifests, stream_agents = {}, {}, {}
        for number, identity in enumerate(identities, 1):
            trial = pilot / 'trials' / identity
            proposal = read(pilot / 'proposals' / (identity + '.json'))
            manifest = read(trial / 'manifest.json')
            task = next(t for t in protocol['tasks'] if t['task_id'] == proposal['task_id'])
            source_id = task['source_id']; source = sources[source_id]
            require(proposal['candidate_id'] == identity and proposal['protocol_sha256'] == sha(pilot / 'protocol.json'), 'proposal identity/protocol')
            require(proposal['source_id'] == source_id and proposal['source_motion'] == source and source_equal(manifest['source_motion'], source), 'trial source identity')
            require(sha(trial / 'proposal.json') == sha(pilot / 'proposals' / (identity + '.json')), 'executed proposal copy')
            parameters = [proposal['parameters'][n] for n in protocol['parameter_names']]
            bounds = np.asarray(protocol['bounds'])
            require(np.isfinite(parameters).all() and np.all(parameters >= bounds[:, 0]) and np.all(parameters <= bounds[:, 1]), 'parameter bounds')
            calibration = identity in calibrations
            if calibration:
                require(proposal['method'] == 'calibration' and parameters == protocol['anchor'], 'calibration anchor')
                baseline_manifests[source_id] = manifest
            else:
                require(identity == cid(task['task_id'], proposal['method'], proposal['trial_index']), 'candidate indexing')
                if proposal['trial_index'] == 0: require(parameters == protocol['anchor'], 'first anchor altered')
            baseline = pilot / 'trials' / ('calibration_' + source_id)
            result = evaluate_generalization(trial, baseline, protocol, task['goal'])
            require(result['integrity_passed'], identity + ': ' + str(result['integrity_errors']))
            require(result.get('fresh_anchor_audit') is not None, 'fresh FK audit absent')
            for key in ('preexecution_input_hashes', 'versions', 'onnx_sha256', 'motion_sha256', 'physics_contract'):
                require(manifest[key] == baseline_manifests[source_id][key], 'baseline identity ' + key)
            for name, digest in manifest['preexecution_input_hashes'].items():
                protocol_digest = protocol['execution_code_sha256'].get(name.replace('\\', '/'))
                if protocol_digest is not None:
                    require(digest == protocol_digest, 'trial implementation differs from protocol pin ' + name)
            require(manifest['motion_sha256'] == source['sha256'] and manifest['common_contract_validated'] is True, 'source/common manifest')
            result.update(candidate_id=identity, parameters=parameters, wall_seconds=manifest['wall_seconds'])
            if calibration: result.update(intent_passed=None, utility=None, utility_l2=None)
            else:
                check_observation(pilot, proposal, protocol, task, results)
                if proposal['method'].startswith('llm'):
                    stream = (task['task_id'], proposal['method'])
                    agent = check_model_receipt(pilot, proposal)
                    require(stream_agents.setdefault(stream, agent) == agent, 'agent changed within model stream')
            result['common_parity'] = common_parity(trial, proposal, output / 'roundtrip_packages' / identity)
            ref = load_reference(trial / 'common_reference')
            provenance = ref.metadata['provenance']['sources']
            require(len(provenance) == 1 and source_equal(provenance[0], source), 'common source provenance')
            expected_kind = 'agentic' if proposal['method'].startswith('llm') else 'baseline'
            require(ref.metadata['producer'] == manifest['producer'] and manifest['producer']['kind'] == expected_kind, 'producer classification')
            if expected_kind == 'agentic': require(manifest['producer']['model'] == 'gpt-6.1-sol', 'declared model route')
            recorded = read(pilot / 'evaluations' / (identity + '.json'))
            for key in METRICS: require(recorded.get(key) == result.get(key), 'stored evaluation ' + identity + ':' + key)
            require(recorded['execution_manifest_sha256'] == sha(trial / 'manifest.json'), 'evaluation manifest linkage')
            check_record(pilot, identity, proposal, manifest)
            results[identity] = result; report['trials'][identity] = result
            if number % 10 == 0: print(f'audited {number}/98 trials', flush=True)
        for source_id in sources:
            anchors = [cid(t['task_id'], m, 0) for t in protocol['tasks'] if t['source_id'] == source_id for m in METHODS]
            for identity in anchors:
                for filename in ('qpos.npy', 'qvel.npy', 'applied_joint_targets.npy'):
                    require((pilot / 'trials' / identity / filename).read_bytes() == (pilot / 'trials' / ('calibration_' + source_id) / filename).read_bytes(), 'paired complete anchor inequality ' + identity + ':' + filename)
        report['paired_anchors'] = dict(source_groups=2, streams_per_source=6, whole_trajectory_qpos_qvel_pd_bitwise_equal=True)
        for task in protocol['tasks']:
            stored_task = next(t for t in comparison['tasks'] if t['task_id'] == task['task_id'])
            for method in METHODS:
                rows = [results[cid(task['task_id'], method, i)] for i in range(8)]
                eligible = [r for r in rows if r['integrity_passed'] and r['physical_gate_passed'] and r['utility'] is not None]
                selected = min(eligible, key=lambda r: (r['utility'], r['utility_l2'], r['candidate_id'])) if eligible else None
                stored_method = next(m for m in stored_task['methods'] if m['method'] == method)
                require([compact(r) for r in stored_method['trials']] == [compact(r) for r in rows], 'comparison trial metrics mismatch')
                stored = stored_method['selected']
                require((stored['candidate_id'] if stored else None) == (selected['candidate_id'] if selected else None), 'independent selector mismatch')
                if stored is not None: require(compact(stored) == compact(selected), 'selected metric mismatch')
                require(stored_method['model_proposal_calls'] == (8 if method.startswith('llm') else 0), 'model proposal call count')
                require(stored_method['provider_tokens'] == (None if method.startswith('llm') else 0) and
                        stored_method['provider_billed_cost'] == (None if method.startswith('llm') else 0), 'provider telemetry disclosure')
                report['selections'].append(dict(task_id=task['task_id'], method=method, selected_candidate=selected['candidate_id'] if selected else None,
                    full_goal_success=bool(selected and selected['intent_passed']), per_dimension_pass=selected['per_dimension_pass'] if selected else None))
        require((comparison['physical_trials'], comparison['control_steps'], comparison['physics_steps']) == (98, 19600, 392000), 'aggregate count mismatch')
        protected = read(pilot / 'protected_hashes_before.json')
        require(bool(protected), 'protected inventory required')
        report['protected_hashes'] = {n: sha(ROOT / n) == d for n, d in protected.items()}
        require(all(report['protected_hashes'].values()), 'protected artifact changed')
        report['actual_counts'] = dict(physical_trials=98, control_steps=19600, physics_steps=392000, individual_model_responses=64)
        require(len(stream_agents) == len(set(stream_agents.values())) == 8, 'eight distinct isolated model stream agents required')
        report['orchestrated_model_route'] = dict(requested_model='gpt-6.1-sol', receipt_count=64,
            isolated_stream_agents={task + ':' + method: agent for (task, method), agent in stream_agents.items()},
            scope='Orchestrator-observed requested route; not independent provider attestation, billing or token telemetry.')
        report['all_selected_goals_successful'] = all(s['full_goal_success'] for s in report['selections'])
        report['audit_passed'] = True
    except Exception as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        report['original_evidence_unchanged'] = inventory(pilot) == before
        report['source_hashes_after'] = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
        report['audit_sources_unchanged'] = report['source_hashes_after'] == pins
        if not report['original_evidence_unchanged'] or not report['audit_sources_unchanged']:
            report['audit_passed'] = False; report['errors'].append('source or evidence changed during audit')
        (output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(audit_passed=report['audit_passed'], errors=report['errors'], output=str(output / 'audit.json'))), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(0 if audit(args.pilot, args.output)['audit_passed'] else 1)
