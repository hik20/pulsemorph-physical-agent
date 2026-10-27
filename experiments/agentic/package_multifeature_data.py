"""Supplemental provenance manifest; not the shared corpus API or admission.

Read a completed pilot without modifying it. No import, training, or physics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def link(path, required=True):
    path = Path(path).resolve()
    if required and not path.is_file():
        raise ValueError('required evidence missing: '+str(path))
    return dict(path=str(path), available=path.is_file(), sha256=digest(path) if path.is_file() else None)


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def package(pilot, output):
    pilot, output = Path(pilot).resolve(), Path(output).resolve()
    if output == pilot or pilot in output.parents:
        raise ValueError('supplemental output must be outside the immutable pilot')
    comparison = read(pilot/'comparison.json')
    protocol = read(pilot/'protocol.json')
    tasks = {t['task_id']:t for t in protocol['tasks']}
    candidates = ['calibration_0']+[r['candidate_id'] for task in comparison['tasks']
                                   for method in task['methods'] for r in method['trials']]
    if len(candidates) != 49 or len(set(candidates)) != 49:
        raise ValueError('completed unique49-trial comparison required')
    source_groups, audio_groups, records = {}, {}, []
    prepared = []
    for candidate in candidates:
        proposal = read(pilot/f'proposals/{candidate}.json')
        trial = pilot/f'trials/{candidate}'
        manifest = read(trial/'manifest.json')
        evaluation = read(pilot/f'evaluations/{candidate}.json')
        common = read(trial/'common_reference/manifest.json')['metadata']
        source_hash = manifest['motion_sha256']
        source_groups.setdefault(source_hash, []).append(candidate)
        task = tasks.get(proposal['task_id'])
        observation_artifact = proposal.get('observation_artifact')
        observed = read(pilot/observation_artifact['path']) if observation_artifact else None
        history = observed.get('prior_trial_results', []) if observed else []
        feedback_ids = [r['candidate_id'] for r in history]
        baseline_supplied = bool(observed and 'baseline' in observed)
        if baseline_supplied:
            feedback_ids = ['calibration_0']+feedback_ids
        feedback = []
        for previous in feedback_ids:
            previous_manifest = read(pilot/f'trials/{previous}/manifest.json')
            previous_common = read(pilot/f'trials/{previous}/common_reference/manifest.json')['metadata']
            feedback.append(dict(candidate_id=previous,
                runtime_session_id=previous_common['context']['runtime_session_id'],
                observed_through_execution_time_s=previous_manifest['actual_control_steps']*previous_manifest['control_dt'],
                evaluation=link(pilot/f'evaluations/{previous}.json'),
                scope='Completed separate offline take summary; never future body observations from the proposed take'))
        audio = task['music_observation'] if task else None
        if audio:
            waveform = link(audio['waveform_path'])
            if waveform['sha256'] != audio['waveform_sha256']:
                raise ValueError('waveform changed')
            audio_groups.setdefault(audio['waveform_sha256'], []).append(candidate)
        else:
            waveform = None
        artifacts = {name:link(trial/path) for name,path in dict(
            common_reference='common_reference/manifest.json', common_numeric_payload='common_reference/arrays.npz',
            actual_policy_inputs='policy_inputs.npz', actual_policy_outputs='policy_outputs.npz',
            applied_pd_targets='applied_joint_targets.npy', actual_qpos='qpos.npy', actual_qvel='qvel.npy',
            actual_state_times='state_times.npy', reference_cache='reference_cache.npz',
            substep_telemetry='substep_telemetry.npz', execution_manifest='manifest.json',
            preexecution_inputs='preexecution_inputs.json').items()}
        artifacts.update(proposal=link(pilot/f'proposals/{candidate}.json'),
                         evaluation=link(pilot/f'evaluations/{candidate}.json'),
                         existing_draft_record=link(pilot/f'trial_records/{candidate}.json'))
        for name,key in [('allowed_observation','observation_artifact'), ('model_response','response_artifact')]:
            item = proposal.get(key)
            artifacts[name] = link(pilot/item['path']) if item else None
            if item and artifacts[name]['sha256'] != item['sha256']:
                raise ValueError('linked proposal artifact changed')
        record = dict(schema='pulsemorph.agentic_supplemental_trial_manifest.v1', candidate_id=candidate,
            shared_corpus_api=False, shared_corpus_imported=False,
            labels=dict(split='unassigned', quality='unassessed', training_admission=False, human_preference=None),
            source_group=dict(id='motion_sha256:'+source_hash, motion_sha256=source_hash,
                scope='All49 trials share one existing motion; tasks/audio are not independent-motion tests'),
            audio_group='waveform_sha256:'+audio['waveform_sha256'] if audio else None,
            intent=dict(task_id=proposal['task_id'], natural_language=task['instruction'] if task else 'Shared unchanged calibration',
                canonical_target=task['goal'] if task else None,
                canonical_target_origin='Authored joint-ratio target and section timeline; timing target selects measured low-band waveform peak' if task else None,
                protocol=link(pilot/'protocol.json')),
            allowed_input=dict(method=proposal['method'], observation_id=proposal['observation_id'],
                model_input_audio_representation='Measured waveform-feature JSON; raw PCM not supplied to model' if proposal['method'].startswith('llm') else 'No LLM proposal call',
                music_observation=audio, waveform_artifact=waveform,
                waveform_observed_through_sample=audio['observed_through_sample'] if audio else None,
                waveform_observed_through_time_s=audio['duration_s'] if audio else None,
                music_analysis_causality='Offline zero-phase analysis, full4s file access' if audio else None,
                authored_sections_s=[[0,2],[2,4]] if task else None,
                authored_sections_are_detected_music_phrases=False,
                prior_completed_take_feedback=feedback,
                feedback_is_current_take_future_observation=False,
                feedback_off_commitment=proposal['method']=='llm_feedback_off'),
            chosen_reference=dict(parameters=proposal['parameters'], metadata=common,
                numeric_target_meaning='Desired q/dq/torso; distinct from applied PD target and realized physical state'),
            execution=dict(runtime_session_id=common['context']['runtime_session_id'],
                fresh_reset=True, snapshot_restore=False, initialization_call_count=manifest['instrumented_set_initial_pose_call_count'],
                initial_reference_joint_posture_sha256=manifest['reference_initial_posture_sha256'],
                actual_initial_qpos_sha256=manifest['actual_initial_qpos_sha256'],
                actual_initial_qvel_sha256=manifest['actual_initial_qvel_sha256'],
                fingerprints_are_same_semantics=False,
                local_controller_observation='Actual current state at50Hz; separate from offline planner summaries',
                actual_observed_through_execution_time_s=manifest['actual_control_steps']*manifest['control_dt'],
                source_body_carriers='Source interpolation; not edited-limb FK',
                control_boundary_forward_count=manifest['control_boundary_forward_count'],
                observation_alignment=manifest['observation_alignment'],
                same_time_fk_implementation_sha256=next(v for p,v in manifest['preexecution_input_hashes'].items() if p.replace('\\','/').endswith('experiments/agentic/execute_multifeature.py')),
                preexecution_input_hashes_unchanged=manifest['preexecution_hashes_unchanged'],
                source_snapshot_directory=str(trial/'source_snapshot'),
                body_profile_sha256=manifest['body_profile_sha256'], controller_profile_sha256=manifest['controller_profile_sha256']),
            realized_outcome=dict(executor_status=manifest['status'], control_steps=manifest['actual_control_steps'],
                physics_steps=manifest['actual_physics_steps'], evaluation=evaluation,
                physical_gate_is_not_training_or_quality_admission=True),
            timestamp_meanings=dict(proposal_created_at_utc=proposal.get('created_at_utc'),
                proposal_time='UTC commitment wall clock; null when not recorded; not simulation event time',
                observed_through='Execution-relative seconds of separately named runtime; offline audio sample boundary is4s',
                target_offsets='Execution-relative desired reference times; lookahead through4.14s is not observed future body or audio',
                actual_state_times='MuJoCo simulation seconds in actual_state_times artifact; each new take resets'),
            provider_accounting=dict(tokens=None if proposal['method'].startswith('llm') else 0,
                billed_cost=None if proposal['method'].startswith('llm') else 0,
                isolated_model_latency_seconds=None, unknown_is_not_zero=True),
            artifacts=artifacts)
        prepared.append((candidate,record))
    if len(source_groups) != 1 or len(audio_groups) != 2:
        raise ValueError('expected one motion group and two audio groups')
    output.mkdir(parents=True, exist_ok=False)
    for candidate,record in prepared:
        path=output/(candidate+'.json'); write(path,record); records.append(link(path))
    audits=[link(p) for p in pilot.glob('*audit*.json')]
    index=dict(schema='pulsemorph.agentic_supplemental_index.v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Supplemental integration manifests only; not shared corpus API, corpus import, or training admission',
        pilot=str(pilot), comparison=link(pilot/'comparison.json'), protocol=link(pilot/'protocol.json'),
        packager=link(Path(__file__)), record_count=len(records), source_groups=source_groups,
        audio_groups=audio_groups, shared_calibration_without_audio_group='calibration_0',
        independent_motion_test=False, records=records, optional_independent_audits=audits,
        split='unassigned', quality='unassessed', training_admission=False)
    write(output/'index.json',index)
    return index


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=package(args.pilot,args.output)
    print(json.dumps(dict(record_count=result['record_count'], source_groups=len(result['source_groups']), audio_groups=len(result['audio_groups']))))
