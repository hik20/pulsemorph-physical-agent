"""Independent auditor rejects leakage and link tampering without physics."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_generalization', ROOT / 'experiments/agentic/audit_generalization.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def fixture(tmp_path):
    source = dict(path='vendor/source.motion', sha256='a' * 64)
    protocol = dict(anchor=[1, 1, 1, 1, 0], parameter_names=['a', 'b', 'c', 'd', 'e'], bounds=[[0, 2]] * 5,
                    feature_names=['f'] * 5, feature_tolerances=[.1] * 5, physical_gate={}, selection='worst then L2',
                    section_windows_s=[[.4, 2], [2, 3.6]], feature_windows_s=[[.4, 2], [2, 3.6]])
    task = dict(task_id='wave_a', source_id='wave', source_motion=source, music_observation={}, instruction='test', goal={})
    write(tmp_path / 'protocol.json', protocol)
    identity = 'wave_a__llm_feedback_off__0'
    (tmp_path / 'observations').mkdir()
    observation = tmp_path / 'observations' / (identity + '.json')
    (tmp_path / 'responses').mkdir()
    response = tmp_path / 'responses' / (identity + '.json')
    results = {'calibration_wave': {}}
    obs = dict(schema='pulsemorph.agentic_generalization_observation.v1', observation_id=identity, task_id='wave_a',
        method='llm_feedback_off', trial_index=0, protocol_sha256=audit.sha(tmp_path / 'protocol.json'),
        source_id='wave', source_motion=source, user_instruction='test', canonical_goal={}, baseline=audit.compact({}),
        parameter_names=protocol['parameter_names'], bounds=protocol['bounds'], feature_names=protocol['feature_names'],
        feature_tolerances=protocol['feature_tolerances'], physical_gate={}, selection=protocol['selection'], music_observation={},
        prior_proposals=[], prior_trial_results=[], trials_remaining=8, forced_first_parameters=protocol['anchor'],
        tool=dict(section_windows_s=protocol['section_windows_s'], measurement_windows_s=protocol['feature_windows_s'],
            gains='Independent per-side initial-posture displacement gains with sin-squared section envelopes. Achieved RMS ratios differ from gains.',
            clock='source_time=t+timing_shift_s*sin(pi*t/4)^2 inside 0<t<4; identity outside; analytic chain-rule velocity.',
            output='Validated MotionReference -> fixed pretrained G1 ONNX -> PD -> MuJoCo. One offline take; no learning.'),
        access='Use only this observation and your own previous supplied observations. No sibling files or code. Feedback-off has no candidate outcomes.',
        response_schema=dict(parameters='one flat list of five numbers', only_observation_path=str(observation), brief_reason='short decision summary'),
        instructions='Return exactly one next vector. First vector must be forced anchor. No extra candidates.')
    write(observation, obs)
    write(response, dict(parameters=protocol['anchor'], only_observation_path=str(observation)))
    proposal = dict(method='llm_feedback_off', trial_index=0, observation_id=identity, parameters=dict(zip(protocol['parameter_names'], protocol['anchor'])),
        observation_artifact=dict(path='observations/' + observation.name, sha256=audit.sha(observation)),
        response_artifact=dict(path='responses/' + response.name, sha256=audit.sha(response)))
    return protocol, task, results, obs, observation, proposal


def test_off_observation_validates_without_candidate_outcomes(tmp_path):
    protocol, task, results, obs, path, proposal = fixture(tmp_path)
    assert audit.check_observation(tmp_path, proposal, protocol, task, results).name.endswith('__0.json')


@pytest.mark.parametrize('leak', ['extra_field', 'prior_outcome', 'cross_stream'])
def test_rejects_off_outcomes_and_cross_stream_observation(tmp_path, leak):
    protocol, task, results, obs, path, proposal = fixture(tmp_path)
    if leak == 'extra_field': obs['sibling_results'] = []
    elif leak == 'prior_outcome': obs['prior_trial_results'] = [{'features': [1] * 5}]
    else: obs['task_id'] = 'wave_b'
    write(path, obs)
    proposal['observation_artifact']['sha256'] = audit.sha(path)
    with pytest.raises(ValueError):
        audit.check_observation(tmp_path, proposal, protocol, task, results)


def test_linked_artifact_mutation_rejected(tmp_path):
    protocol, task, results, obs, path, proposal = fixture(tmp_path)
    obs['prior_trial_results'] = [1]
    write(path, obs)
    with pytest.raises(ValueError, match='linked artifact'):
        audit.check_observation(tmp_path, proposal, protocol, task, results)


def test_path_separator_equivalence_requires_same_digest():
    assert audit.source_equal(dict(path='a\\b', sha256='x'), dict(path='a/b', sha256='x'))
    assert not audit.source_equal(dict(path='a/b', sha256='y'), dict(path='a/b', sha256='x'))


def test_incomplete_pilot_does_not_create_output(tmp_path):
    pilot = tmp_path / 'pilot'
    pilot.mkdir()
    output = tmp_path / 'new-audit'
    with pytest.raises(FileNotFoundError):
        audit.audit(pilot, output)
    assert not output.exists()


@pytest.mark.parametrize('tamper', [None, 'pcm', 'target', 'authored_error'])
def test_pcm_recomputation_and_authored_onset_are_distinct(tmp_path, tamper):
    import wave
    from generalization_tasks import music
    pcm, truth = music(2.98, 3101)
    folder = tmp_path / 'music' / 'wave_a'
    folder.mkdir(parents=True)
    path = folder / 'music.wav'
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050); f.writeframes(pcm.tobytes())
    measured = audit.analyze_waveform(path)
    event = next(e for e in measured['measured_bass_events'] if e['event_id'] == measured['strongest_bass_event_id'])
    write(folder / 'waveform_observation.json', measured)
    write(folder / 'authored_truth.json', truth)
    task = dict(task_id='wave_a', music_observation=measured,
                goal=dict(feature_targets=[1] * 4 + [event['time_s']], cue_event_id=event['event_id']),
                authored_cue_observation_error_s=event['time_s'] - truth['authored_primary_cue_s'])
    if tamper == 'pcm':
        altered = bytearray(path.read_bytes()); altered[-1] ^= 1; path.write_bytes(altered)
    elif tamper == 'target': task['goal']['feature_targets'][4] = truth['authored_primary_cue_s']
    elif tamper == 'authored_error': task['authored_cue_observation_error_s'] += .01
    if tamper:
        with pytest.raises(ValueError): audit.check_waveform(tmp_path, task)
    else:
        result = audit.check_waveform(tmp_path, task)
        assert result['measured_minus_authored_onset_s'] != 0
        assert result['measured_strongest_event']['time_s'] == task['goal']['feature_targets'][4]
