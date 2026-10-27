"""Freeze serialization checks only; tests publish synthetic temporary plans."""
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
plan = importlib.import_module('robustness_plan')


def test_actual_preserved_selected_cases_are_exact():
    original = plan.read(plan.ROOT/plan.PILOT/'protocol.json')
    comparison = plan.read(plan.ROOT/plan.PILOT/'comparison.json')
    cases = plan.original_cases(original, comparison)
    assert [c['role'] for c in cases] == ['calibration_wave', 'calibration_high5',
        'selected_wave_a', 'selected_wave_b', 'selected_high5_a', 'selected_high5_b']
    assert [Path(c['original_trial']).name for c in cases[2:]] == [
        'wave_a__llm_feedback__7', 'wave_b__llm_feedback__7',
        'high5_a__llm_feedback__6', 'high5_b__llm_feedback__6']
    assert all(c['goal'] is None for c in cases[:2])


def test_replay_only_changes_authorized_identity_and_execution_fields():
    original = {'candidate_id':'old', 'method':'llm_feedback', 'parameters':{'gain':1},
                'protocol_sha256':'oldhash', 'observation_id':'old_observation'}
    case = {'role':'selected_wave_a', 'original_proposal':{'path':'old.json', 'sha256':'a'*64}}
    scenario = plan.build_scenarios(['hip'])[1]
    result = plan.replay_proposal(original, case, scenario, 'b'*64)
    assert result['parameters'] == original['parameters']
    assert result['method'] == original['method']
    assert result['observation_id'] == result['candidate_id'] + ':replay-observation'
    assert result['observation_artifact'] is None and result['response_artifact'] is None
    assert result['runtime_session_id'] == 'b'*64 + ':joint_position_plus:selected_wave_a'
    assert result['execution_kind'] == 'initial_state_sensitivity_replay'
    assert result['initial_perturbation'] == scenario['initial_perturbation']
    result['parameters']['gain'] = 2
    assert original['parameters']['gain'] == 1


def test_synthetic_freeze_78_exclusive_proposals_and_protocol_hash(tmp_path, monkeypatch):
    root = tmp_path/'source'; root.mkdir()
    source = root/'origin.json'; source.write_text('{"candidate_id":"old","parameters":{}}', encoding='utf8')
    code = root/'executor.py'; code.write_bytes(b'"""Synthetic source."""\n')
    cases = [{'role':role, 'original_proposal':{'path':'origin.json', 'sha256':plan.sha(source)}}
             for role in ('calibration_wave', 'calibration_high5', 'selected_wave_a',
                          'selected_wave_b', 'selected_high5_a', 'selected_high5_b')]
    protocol = {'cases':cases, 'scenarios':plan.build_scenarios(['hip']), 'expected_counts':plan.COUNTS,
                'execution_code_sha256':{'executor.py':plan.sha(code)}}
    monkeypatch.setattr(plan, 'ROOT', root)
    monkeypatch.setattr(plan, 'build_protocol', lambda:protocol)
    output = root/'artifacts/runs/agentic/synthetic_panel'
    assert plan.freeze(output) == protocol
    paths = list((output/'proposals').glob('*.json'))
    assert len(paths) == 78
    assert (output/'source_snapshot/executor.py').read_bytes() == code.read_bytes()
    digest = plan.sha(output/'protocol.json')
    for path in paths:
        proposal = json.loads(path.read_text(encoding='utf8'))
        assert proposal['protocol_sha256'] == digest
        assert proposal['candidate_id'] == path.stem
        assert proposal['runtime_session_id'].startswith(digest+':')
    with pytest.raises(FileExistsError):
        plan.freeze(output)
    with pytest.raises(ValueError, match='strictly under'):
        plan.freeze(tmp_path/'outside_panel')


def test_protected_inventory_detects_changed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, 'ROOT', tmp_path)
    directory=tmp_path/'old'; directory.mkdir()
    source=directory/'evidence'; source.write_bytes(b'original')
    first=plan.protected_inventory(Path('old'))
    source.write_bytes(b'modified')
    assert first != plan.protected_inventory(Path('old'))
    assert first['path'] == 'old'
