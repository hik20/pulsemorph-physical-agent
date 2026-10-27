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
from audit_multifeature import read, sha, require, inventory, linked, equal, common_parity
from evaluate_generalization import evaluate_generalization
from tool_selection_search import next_candidate, TOOLS
from tool_selection_driver_v2 import compact, tool_descriptions, validate_calibrations, select, artifact
from execute_tool_selection import proposal_producer, validate_parameters
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


def check_control_provenance(proposal, manifest, reference_metadata, draft, adapter):
    """Use the REAL frozen runtime producer contract, including tool identity."""
    expected = dict(proposal_producer(proposal), control_tool=proposal['control_tool'])
    require(manifest['schema'] == 'pulsemorph.agentic_generalization_trial.v1', 'runtime schema')
    require(manifest['producer'] == expected, 'exact runtime producer classification')
    require(reference_metadata['producer'] == expected, 'exact common producer classification')
    for value in (manifest, reference_metadata, draft, adapter):
        require(value['control_tool'] == proposal['control_tool'], 'control_tool provenance mismatch')
    require(draft['producer'] == expected, 'draft producer classification')
    require(manifest['experiment_kind'] == reference_metadata['experiment_kind'] == 'tool_selection_v1', 'experiment kind')


def check_quarantined_predecessor(protocol):
    link=protocol['supersedes_quarantined']
    expected=(ROOT/'artifacts/runs/agentic/tool_selection_development_v1/abort.json').resolve()
    require(set(link)=={'path','sha256'} and Path(link['path']).resolve()==expected, 'exact quarantined predecessor link')
    require(sha(expected)==link['sha256'], 'quarantined predecessor digest')
    previous=read(expected)
    require(previous['schema']=='pulsemorph.agentic_tool_selection_abort.v1'
        and previous['stage']=='shared_calibration_goal_context_error'
        and previous['retry_allowed'] is False and previous['slot_consumed'] is True, 'quarantined predecessor stage')
    require(protocol['imported_prior_trial_records']==[], 'no quarantined candidate labels imported')
    created=datetime.fromisoformat(previous['created_at_utc'])
    require(created.tzinfo is not None, 'predecessor quarantine timezone')
    agents={read(path)['agent_name'] for path in (expected.parent/'model_calls').glob('*.json')}
    return dict(path=str(expected),sha256=link['sha256'],stage=previous['stage'],
        created_at_utc=previous['created_at_utc'],previous_model_agents=sorted(agents),old_trial_labels_imported=False)


def check_development_scope(protocol, comparison, old):
    require(protocol['schema'] == 'pulsemorph.agentic_tool_selection_protocol.v2', 'protocol schema')
    require(comparison['schema'] == 'pulsemorph.agentic_tool_selection_comparison.v1', 'comparison schema')
    require(protocol['phase'] == 'development_comparison' and comparison['development_only'] is True, 'development scope required')
    require(protocol['training_admission'] is False and protocol['imported_prior_trial_records'] == [], 'no training/old labels')
    require(protocol['optimizer']['old_feature_labels_imported'] is False, 'old optimizer labels')
    require(protocol['control_tools'] == list(TOOLS), 'exact equal tool availability')
    require(protocol['trial_budget_per_method_per_task'] == protocol['new_candidate_budget'] == 8, 'eight-slot budget')
    require('All8 candidate slots' in protocol['stopping_policy'] and 'no success early stop' in protocol['stopping_policy'], 'stopping rule')
    require('No confirmation' in protocol['stopping_policy'] and 'no free repair, retry or silent replacement' in protocol['stopping_policy'], 'retry/confirmation rule')
    require(protocol['planned_budget'] == dict(candidate_trials=96, calibration_trials=4, physical_trials=100,
        control_steps=20000, physics_steps=400000, model_proposal_responses=64), 'planned counts')
    for key in ('sources', 'parameter_names', 'bounds', 'anchor', 'feature_names', 'feature_windows_s', 'feature_tolerances',
                'physical_gate', 'accent_window_s', 'accent_smoothing_samples', 'accent_minimum_prominence',
                'section_windows_s', 'duration_s', 'source_start_s', 'selection', 'quality_diagnostics'):
        require(protocol[key] == old[key], 'unchanged original protocol '+key)
    require(len(protocol['tasks']) == len(old['tasks']) == 4, 'original task count')
    for task, original in zip(protocol['tasks'], old['tasks']):
        normalized = json.loads(json.dumps(task))
        normalized['music_observation']['waveform_path'] = original['music_observation']['waveform_path']
        require(normalized == original, 'original task/goal/audio changed beyond waveform path')
    return True


def calibration_measurements(result):
    """Independent exact goal-free view; never import driver's baseline builder."""
    keys=('candidate_id','parameters','control_tool','features','integrity_passed',
          'physical_gate_passed','mean_joint_rmse_rad','min_root_height_m')
    view={name:result.get(name) for name in keys}
    if isinstance(result.get('accent'),dict):
        measured=('available','peak_time_s','peak_rms_speed_rad_s','event_times_s',
                  'event_count','window_event_count')
        view['accent']={name:result['accent'][name] for name in measured if name in result['accent']}
    return view


def check_calibration_view(observed, result):
    require(observed == calibration_measurements(result), 'exact target-independent calibration measurements')


def check_observation(pilot, proposal, protocol, task, results):
    """Rebuild exact own-history observations from independently recomputed rows."""
    method, index = proposal['method'], proposal['trial_index']
    path = linked(pilot, proposal['observation_artifact'])
    obs = read(path)
    if index:
        prior_file=(pilot/('proposals' if method == 'llm_feedback_off' else 'evaluations')/
            (cid(task['task_id'],method,index-1)+'.json'))
        require(prior_file.stat().st_ctime_ns < path.stat().st_ctime_ns, 'observation own-history chronology')
    require(path.stat().st_ctime_ns < (pilot/'proposals'/(proposal['candidate_id']+'.json')).stat().st_ctime_ns, 'observation before proposal')
    identity = cid(task['task_id'], method, index)
    require(path == (pilot/'observations'/(identity+'.json')).resolve(), 'individual observation path')
    history = [] if method == 'llm_feedback_off' else [compact(results[cid(task['task_id'], method, j)]) for j in range(index)]
    baseline_id = 'calibration_'+task['source_id']
    baseline = results[baseline_id]
    cal_path = pilot/'evaluations'/(baseline_id+'.json')
    cal_provenance = artifact(pilot, cal_path)
    check_calibration_view(obs['baseline'],baseline)
    if method == 'surrogate_search':
        calibration = dict(parameters=protocol['anchor'], features=baseline['features'], integrity_passed=True,
            physical_gate_passed=True, provenance=cal_provenance, control_tool=TOOLS[0])
        choice = next_candidate(history, protocol, task['goal'], calibration)
        require(proposal['control_tool'] == choice['control_tool'], 'deterministic search tool replay')
        equal([proposal['parameters'][n] for n in protocol['parameter_names']], choice['parameters'], 'deterministic search replay')
        require(proposal['search_decision'] == choice['decision'], 'full search decision replay')
        expected = dict(schema='pulsemorph.agentic_tool_selection_search_observation.v1', observation_id=identity,
            task_id=task['task_id'], method=method, prior_trial_results=history, canonical_goal=task['goal'],
            music_observation=task['music_observation'], baseline=calibration_measurements(baseline), calibration_provenance=cal_provenance,
            tools=tool_descriptions(protocol), search_decision=choice['decision'], control_tool=choice['control_tool'],
            protocol_sha256=sha(pilot/'protocol.json'), proposal_compute_seconds=obs['proposal_compute_seconds'])
        require(type(obs['proposal_compute_seconds']) in (int,float) and np.isfinite(obs['proposal_compute_seconds']) and obs['proposal_compute_seconds'] >= 0, 'proposal duration')
        require(obs == expected, 'exact optimizer observation')
        require(proposal['response_artifact'] is None, 'optimizer cannot have model response')
        return None
    previous = []
    for j in range(index):
        previous_id=cid(task['task_id'],method,j)
        p=read(pilot/'proposals'/(previous_id+'.json'))
        previous.append(dict(candidate_id=previous_id,control_tool=p['control_tool'],
            parameters=[p['parameters'][n] for n in protocol['parameter_names']],
            proposal_sha256=sha(pilot/'proposals'/(previous_id+'.json'))))
    expected = dict(schema='pulsemorph.agentic_tool_selection_observation.v2', observation_id=identity,
        task_id=task['task_id'], method=method, trial_index=index, protocol_sha256=sha(pilot/'protocol.json'),
        source_id=task['source_id'],source_motion=task['source_motion'],user_instruction=task['instruction'],
        canonical_goal=task['goal'],baseline=calibration_measurements(baseline),parameter_names=protocol['parameter_names'],
        bounds=protocol['bounds'],feature_names=protocol['feature_names'],feature_tolerances=protocol['feature_tolerances'],
        physical_gate=protocol['physical_gate'],selection=protocol['selection'],music_observation=task['music_observation'],
        prior_proposals=previous,prior_trial_results=history,trials_remaining=8-index,forced_first_parameters=None,
        tools=tool_descriptions(protocol),calibration_provenance=cal_provenance,
        access='Use only this observation and your own previous supplied observations. No sibling files or code. Feedback-off has no candidate outcomes.',
        response_schema=dict(control_tool=list(TOOLS),parameters='one flat list of five numbers',only_observation_path=str(path),brief_reason='short decision summary'),
        instructions='Return exactly one tool and one vector in the exact response schema. Every slot is freely chosen; identity calibration is already supplied. No extra candidates or keys.')
    require(obs == expected, 'exact isolated model observation/history')
    require(proposal['observation_id'] == identity and proposal['search_decision'] is None, 'model proposal context')
    response_path=linked(pilot,proposal['response_artifact'])
    require(response_path == (pilot/'responses'/(identity+'.json')).resolve(), 'individual response path')
    raw=read(response_path)
    require(set(raw) == {'control_tool','parameters','only_observation_path','brief_reason'}, 'exact raw response fields')
    require(raw['control_tool'] == proposal['control_tool'] and Path(raw['only_observation_path']).resolve() == path, 'raw tool/path linkage')
    require(type(raw['brief_reason']) is str and raw['brief_reason'] == proposal['brief_reason'], 'raw reason unchanged')
    require(isinstance(raw['parameters'],list) and len(raw['parameters']) == 5 and all(type(v) in (int,float) for v in raw['parameters']), 'flat numeric response')
    equal(raw['parameters'],[proposal['parameters'][n] for n in protocol['parameter_names']], 'unchanged raw model values')
    return response_path


def check_record(pilot, identity, proposal, manifest):
    record = read(pilot / 'trial_records' / (identity + '.json'))
    require(record['schema'] == 'pulsemorph.agentic_tool_selection_trial_record.v1' and record['control_tool'] == proposal['control_tool'], 'record schema/tool')
    require(record['split'] == 'unassigned' and record['quality'] == 'unassessed', 'record development admission')
    require(record['candidate_id'] == identity and record['task_id'] == proposal['task_id'], 'record identity')
    require(record['source_motion'] == proposal['source_motion'] and record['source_group'] == proposal['source_motion']['sha256'], 'record source group')
    require(linked(pilot,record['protocol']) == (pilot/'protocol.json').resolve(), 'record protocol linkage')
    require(record['model_reason_is_measurement'] is False, 'model reasons not measurements')
    require(record['provider_billed_cost'] == (None if proposal['method'].startswith('llm') else 0), 'unknown provider cost disclosure')
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


def check_summary_tools(stored_method, rows):
    selected=select(rows)
    require(len(rows) == 8, 'eight complete candidates required')
    require(stored_method['selected_tool'] == (selected['control_tool'] if selected else None), 'selected tool mismatch')
    require(stored_method['tool_usage'] == {tool:sum(r['control_tool']==tool for r in rows) for tool in TOOLS}, 'tool usage counts')
    require(stored_method['physical_trials'] == 8, 'no success early stopping')
    require(len(stored_method['prefix_best']) == 8, 'eight prefix selections')
    for i,prefix in enumerate(stored_method['prefix_best'],1):
        wanted=select(rows[:i]); recorded_prefix=prefix['selected']
        require(prefix['budget'] == i, 'prefix budget')
        require((compact(recorded_prefix) if recorded_prefix else None) == (compact(wanted) if wanted else None), 'prefix independent selection')


def audit(pilot, output):
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    require(output != pilot and pilot not in output.parents, 'audit output must be outside pilot')
    # Require completion before creating even an audit output directory.
    protocol, comparison = read(pilot / 'protocol.json'), read(pilot / 'comparison.json')
    output.mkdir(parents=True, exist_ok=False)
    before = inventory(pilot)
    dependencies = [Path(__file__)] + [Path(__file__).with_name(n) for n in
        ('audit_multifeature.py', 'evaluate_generalization.py', 'evaluate_multifeature.py', 'evaluate_trial.py',
         'execute_tool_selection.py', 'tool_selection_driver_v2.py', 'tool_selection_search.py',
         'execute_generalization.py', 'execute_local_capability.py', 'tool_selection_tasks_v2.py', 'tool_selection_batch_v2.py', 'audit_generalization.py',
         'generalization_search.py', 'multifeature_driver.py')]
    dependencies += [Path(__file__).with_name('musical_tasks.py')]
    dependencies += sorted((ROOT / 'src/pulsemorph/motion').glob('*.py'))
    dependencies += [ROOT/'src/pulsemorph/agentic'/n for n in ('reference.py','common_player.py',
        'multifeature_reference.py','local_multifeature_reference.py')]
    dependencies += [ROOT/'experiments/causal_music/reference.py']
    pins = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    report = dict(schema='pulsemorph.agentic_tool_selection_final_audit.v2', audit_passed=False, errors=[], trials={}, selections=[],
        source_hashes_before=pins, evidence_inventory_before=before, scope='Posthoc read-only recomputation; no physics steps, model calls, threshold changes or corpus import.',
        limitations=['Local timestamps and route declarations are not external attestations.',
                     'Access restriction consists of supplied observations, agent instructions and fork_turns=none; it is not an OS sandbox or independent provider trace.',
                     'Audit acceptance is separate from task success, expression, human preference and hardware safety.',
                     'Provider token counts and billing are unknown for model proposals.'])
    try:
        for p in dependencies:
            dest = output / 'source_snapshot' / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
        require(not (pilot/'abort.json').exists(), 'aborted experiment cannot pass')
        original=protocol['original_task_protocol']
        require(sha(Path(original['path'])) == original['sha256'] == '80c2fac7bbc1a886e774e4d32aa569c069a3f4750e91a4a33628f8594df667ec', 'original protocol pin')
        check_development_scope(protocol,comparison,read(Path(original['path'])))
        predecessor=check_quarantined_predecessor(protocol)
        report['quarantined_predecessor']=predecessor
        report['calibration_parity']=validate_calibrations(pilot)
        require(comparison['calibration_parity'] == report['calibration_parity'], 'comparison calibration parity proof')
        availability=protocol['source_availability_inventory']
        require(sha(Path(availability['path'])) == availability['sha256'], 'source availability inventory pin')
        require({t['task_id'] for t in protocol['tasks']} == TASKS and len(protocol['tasks']) == 4, 'expected four exact tasks')
        require(list(protocol['methods']) == METHODS and protocol['anchor'] == [1, 1, 1, 1, 0], 'method/anchor contract')
        require(comparison['protocol_sha256'] == sha(pilot / 'protocol.json'), 'comparison protocol digest')
        sources = {t['source_id']: t['source_motion'] for t in protocol['tasks']}
        require(set(sources) == {'wave', 'high5'} and len({s['sha256'] for s in sources.values()}) == 2, 'two distinct source groups required')
        for task in protocol['tasks']:
            require(task['source_motion'] == sources[task['source_id']], 'inconsistent source mapping')
        report['waveform_verification'] = {t['task_id']: check_waveform(pilot, t) for t in protocol['tasks']}
        calibrations = ['calibration_'+s for s in sorted(sources)] + ['calibration_local_'+s for s in sorted(sources)]
        require(protocol['calibration_ids'] == ['calibration_wave','calibration_local_wave','calibration_high5','calibration_local_high5'], 'calibration ids')
        candidates = [cid(t['task_id'], m, i) for t in protocol['tasks'] for m in METHODS for i in range(8)]
        identities = calibrations + candidates
        require({p.name for p in (pilot / 'trials').iterdir() if p.is_dir()} == set(identities), 'exact 100 trial inventory')
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
        calibration_completion=max((pilot/'evaluations'/(identity+'.json')).stat().st_ctime_ns for identity in calibrations)
        require(all(path.stat().st_ctime_ns > calibration_completion for path in (pilot/'observations').glob('*.json')), 'observation predates full fresh calibration')
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
            require(datetime.fromisoformat(proposal['created_at_utc']) > datetime.fromisoformat(predecessor['created_at_utc']), 'fresh proposal must postdate quarantine')
            require(proposal['candidate_id'] == identity and proposal['protocol_sha256'] == sha(pilot / 'protocol.json'), 'proposal identity/protocol')
            require(proposal['source_id'] == source_id and proposal['source_motion'] == source and source_equal(manifest['source_motion'], source), 'trial source identity')
            require(sha(trial / 'proposal.json') == sha(pilot / 'proposals' / (identity + '.json')), 'executed proposal copy')
            validate_parameters(proposal['parameters'])
            parameters = [proposal['parameters'][n] for n in protocol['parameter_names']]
            bounds = np.asarray(protocol['bounds'])
            require(np.isfinite(parameters).all() and np.all(parameters >= bounds[:, 0]) and np.all(parameters <= bounds[:, 1]), 'parameter bounds')
            calibration = identity in calibrations
            if calibration:
                require(proposal['method'] == 'calibration' and parameters == protocol['anchor'], 'calibration anchor')
                expected_tool=TOOLS[1] if identity.startswith('calibration_local_') else TOOLS[0]
                require(proposal['control_tool'] == expected_tool, 'calibration tool identity')
                if expected_tool == TOOLS[0]: baseline_manifests[source_id] = manifest
            else:
                require(identity == cid(task['task_id'], proposal['method'], proposal['trial_index']), 'candidate indexing')
                require(proposal['schema'] == 'pulsemorph.agentic_tool_selection_proposal.v1', 'proposal schema')
            baseline = pilot / 'trials' / ('calibration_' + source_id)
            result = evaluate_generalization(trial, trial if calibration else baseline, protocol, task['goal'])
            require(result['integrity_passed'], identity + ': ' + str(result['integrity_errors']))
            require(result.get('fresh_anchor_audit') is not None, 'fresh FK audit absent')
            for key in ('preexecution_input_hashes', 'versions', 'onnx_sha256', 'motion_sha256', 'physics_contract'):
                require(manifest[key] == baseline_manifests[source_id][key], 'baseline identity ' + key)
            for name, digest in manifest['preexecution_input_hashes'].items():
                protocol_digest = protocol['execution_code_sha256'].get(name.replace('\\', '/'))
                if protocol_digest is not None:
                    require(digest == protocol_digest, 'trial implementation differs from protocol pin ' + name)
            require(manifest['motion_sha256'] == source['sha256'] and manifest['common_contract_validated'] is True, 'source/common manifest')
            result.update(candidate_id=identity, parameters=parameters, wall_seconds=manifest['wall_seconds'], control_tool=proposal['control_tool'])
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
            check_control_provenance(proposal,manifest,ref.metadata,read(trial/'motion_reference_draft.json'),read(trial/'adapter_motion_reference_draft.json'))
            require(manifest['actual_control_steps'] == 200 and manifest['actual_physics_steps'] == 4000
                and manifest['control_boundary_forward_count'] == 200 and manifest['status'] == 'success', 'real execution counts')
            recorded = read(pilot / 'evaluations' / (identity + '.json'))
            require(recorded['protocol_sha256'] == sha(pilot/'protocol.json') and recorded['evaluator_sha256'] == sha(Path(__file__).with_name('evaluate_generalization.py')), 'evaluation protocol/evaluator pin')
            require((recorded['control_steps'],recorded['physics_steps'],recorded['executor_status']) == (200,4000,'success'), 'evaluation execution counts')
            require(recorded['control_tool'] == proposal['control_tool'], 'stored evaluation tool')
            for key in METRICS: require(recorded.get(key) == result.get(key), 'stored evaluation ' + identity + ':' + key)
            require(recorded['execution_manifest_sha256'] == sha(trial / 'manifest.json'), 'evaluation manifest linkage')
            check_record(pilot, identity, proposal, manifest)
            results[identity] = result; report['trials'][identity] = result
            if number % 10 == 0: print(f'audited {number}/100 trials', flush=True)
        report['paired_anchors'] = dict(source_groups=2, tools_per_source=2, whole_trajectory_qpos_qvel_pd_bitwise_equal=True)
        require(len(comparison['tasks']) == 4 and {t['task_id'] for t in comparison['tasks']} == TASKS, 'exact comparison tasks')
        require(len(comparison['calibrations']) == 4 and {r['candidate_id'] for r in comparison['calibrations']} == set(calibrations), 'comparison calibration inventory')
        for recorded_calibration in comparison['calibrations']:
            require(compact(recorded_calibration) == compact(results[recorded_calibration['candidate_id']]), 'comparison calibration metrics')
        for task in protocol['tasks']:
            stored_task = next(t for t in comparison['tasks'] if t['task_id'] == task['task_id'])
            require(stored_task['goal'] == task['goal'] and stored_task['source_id'] == task['source_id'], 'comparison task goal/source')
            require(len(stored_task['methods']) == 3 and [m['method'] for m in stored_task['methods']] == METHODS, 'comparison method inventory')
            for method in METHODS:
                rows = [results[cid(task['task_id'], method, i)] for i in range(8)]
                eligible = [r for r in rows if r['integrity_passed'] and r['physical_gate_passed'] and r['utility'] is not None]
                selected = min(eligible, key=lambda r: (r['utility'], r['utility_l2'], r['candidate_id'])) if eligible else None
                stored_method = next(m for m in stored_task['methods'] if m['method'] == method)
                require([compact(r) for r in stored_method['trials']] == [compact(r) for r in rows], 'comparison trial metrics mismatch')
                stored = stored_method['selected']
                require((stored['candidate_id'] if stored else None) == (selected['candidate_id'] if selected else None), 'independent selector mismatch')
                if stored is not None: require(compact(stored) == compact(selected), 'selected metric mismatch')
                check_summary_tools(stored_method,rows)
                require(stored_method['model_proposal_calls'] == (8 if method.startswith('llm') else 0), 'model proposal call count')
                require(stored_method['provider_tokens'] == (None if method.startswith('llm') else 0) and
                        stored_method['provider_billed_cost'] == (None if method.startswith('llm') else 0), 'provider telemetry disclosure')
                report['selections'].append(dict(task_id=task['task_id'], method=method, selected_candidate=selected['candidate_id'] if selected else None,
                    selected_tool=selected['control_tool'] if selected else None,
                    prefix_selected_tools=[select(rows[:i])['control_tool'] if select(rows[:i]) else None for i in range(1,9)],
                    full_goal_success=bool(selected and selected['intent_passed']), per_dimension_pass=selected['per_dimension_pass'] if selected else None))
        require((comparison['physical_trials'], comparison['control_steps'], comparison['physics_steps'], comparison['model_proposal_calls']) == (100, 20000, 400000, 64), 'aggregate count mismatch')
        require(comparison['comparison_valid'] is True, 'comparison validity')
        protected = read(pilot / 'protected_hashes_before.json')
        expected_protected=read(ROOT/'artifacts/models/learned_rhythm/v2/freeze.json')['sha256']
        require(set(protected) == set(expected_protected), 'complete protected inventory required')
        report['protected_hashes'] = {n: sha(ROOT / n) == d for n, d in protected.items()}
        require(all(report['protected_hashes'].values()), 'protected artifact changed')
        report['actual_counts'] = dict(physical_trials=100, control_steps=20000, physics_steps=400000, individual_model_responses=64)
        require(len(stream_agents) == len(set(stream_agents.values())) == 8, 'eight distinct isolated model stream agents required')
        require(not (set(stream_agents.values()) & set(predecessor['previous_model_agents'])), 'quarantined model contexts reused')
        report['orchestrated_model_route'] = dict(requested_model='gpt-6.1-sol', receipt_count=64,
            isolated_stream_agents={task + ':' + method: agent for (task, method), agent in stream_agents.items()},
            scope='Orchestrator-observed requested route; not independent provider attestation, billing or token telemetry.')
        report['all_selected_goals_successful'] = all(s['full_goal_success'] for s in report['selections'])
        report['audit_passed'] = True
    except Exception as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        report['evidence_inventory_after'] = inventory(pilot)
        report['original_evidence_unchanged'] = report['evidence_inventory_after'] == before
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
