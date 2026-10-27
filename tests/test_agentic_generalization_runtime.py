"""Generalization source trust and descriptive runtime diagnostics; no physics run."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments/agentic' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

@pytest.fixture
def runtime(tmp_path, monkeypatch):
    value = module('execute_generalization')
    monkeypatch.setattr(value, 'ROOT', tmp_path)
    monkeypatch.setattr(value, 'UPSTREAM', tmp_path / 'research/vendor/ProtoMotions')
    source = value.UPSTREAM / 'data/g1-kimodo-generated/proto/example.motion'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'trusted fixture; never deserialize')
    return value, source

def proposal(value, source):
    return {'source_motion': {'path': str(source.relative_to(value.ROOT)), 'sha256': value.sha256(source)}}

def test_exact_trusted_path_and_digest(runtime):
    value, source = runtime
    assert value.resolve_source_motion(proposal(value, source)) == (source, value.sha256(source))

@pytest.mark.parametrize('path', ['other.motion', '../example.motion', 'research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/nested/a.motion'])
def test_reject_outside_direct_allowlist(runtime, path):
    value, source = runtime
    p = proposal(value, source)
    p['source_motion']['path'] = path
    with pytest.raises(ValueError):
        value.resolve_source_motion(p)

def test_hash_mismatch_before_any_deserialization(runtime, monkeypatch):
    value, source = runtime
    p = proposal(value, source)
    p['source_motion']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='before deserialization'):
        value.resolve_source_motion(p)

def test_bad_source_proposal_rejects_before_motor_import(runtime, tmp_path):
    import json
    value, source = runtime
    p = proposal(value, source)
    p.update(candidate_id='fixture', method='calibration')
    p['source_motion']['sha256'] = '0' * 64
    input_path = tmp_path / 'proposal.json'
    input_path.write_text(json.dumps(p))
    result = value.execute_trial(input_path, tmp_path / 'artifacts/runs/agentic/rejected')
    assert result['status'] == 'failed'
    assert 'before deserialization' in result['error']
    assert result['actual_physics_steps'] == 0

def test_diagnostics_use_actual_targets_and_velocity(tmp_path):
    value = module('evaluate_generalization')
    times = np.arange(201) * .02
    qpos = np.zeros((201, 36)); qpos[:, 0] = times
    qvel = np.zeros((201, 35)); qvel[:, 6:] = 2 * times[:, None]
    targets = np.repeat((np.arange(200) * .1)[:, None], 29, axis=1)
    source = np.zeros((201, 3)); source[:, 0] = .5 * times
    np.save(tmp_path / 'unwarped_source_root_positions.npy', source)
    result = value.quality_diagnostics({'arrays': dict(qpos=qpos, qvel=qvel, state_times=times, applied_joint_targets=targets)}, tmp_path)
    assert result['diagnostic_only'] is True
    assert result['applied_pd_target_step_rms_rad'] == pytest.approx(.1)
    assert result['joint_acceleration_finite_difference_rms_rad_s2'] == pytest.approx(2)
    assert result['root_displacement_difference_final_m'] == pytest.approx(2)
    assert not any('passed' in key for key in result)

def test_generalization_audit_explicitly_requires_fresh_fk(tmp_path, monkeypatch):
    import json
    value = module('evaluate_generalization')
    manifest = {'schema': 'pulsemorph.agentic_generalization_trial.v1', 'artifact_hashes': {}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    # Isolate the legacy numerical audit while exercising new inventory/schema routing.
    def old_audit(directory):
        return {'manifest': {}, 'arrays': {}}
    monkeypatch.setattr(value._legacy, '_audit', old_audit)
    called = []
    def fresh(directory, audit):
        called.append(directory)
        return {'fresh': True}
    monkeypatch.setattr(value, '_audit_fresh_anchor', fresh)
    result = value._audit(tmp_path)
    assert called == [tmp_path.resolve()]
    assert result['fresh_anchor_audit'] == {'fresh': True}


@pytest.mark.parametrize('method,kind', [
    ('calibration', 'baseline'), ('surrogate_search', 'baseline'),
    ('llm_feedback', 'agentic'), ('llm_feedback_off', 'agentic')])
def test_producer_classification(method, kind):
    value = module('execute_generalization')
    producer = value.proposal_producer({'method': method})
    assert producer['kind'] == kind
    assert producer['id'] == 'generalization-' + method
    if kind == 'agentic':
        assert producer['model'] == 'gpt-6.1-sol'
    else:
        assert 'model' not in producer

@pytest.mark.parametrize('method', [None, 'jacobian_search', 'unknown', '', 123])
def test_producer_rejects_unsupported_methods(method):
    value = module('execute_generalization')
    with pytest.raises(ValueError, match='unsupported generalization method'):
        value.proposal_producer({'method': method})


def test_snapshot_copies_origin_before_linking(runtime):
    value, source = runtime
    expected = source.read_bytes()
    store = value.ROOT / 'artifacts/runs/agentic/experiment/source_snapshot_objects'
    destination = value.ROOT / 'artifacts/runs/agentic/experiment/trials/a/source_snapshot/a.py'
    digest = value.sha256(source)
    value.snapshot_source(source, destination, store, digest)
    source.write_bytes(b'changed live source')
    assert destination.read_bytes() == expected
    assert (store / digest).read_bytes() == expected
    assert destination.stat().st_ino == (store / digest).stat().st_ino
    assert destination.stat().st_ino != source.stat().st_ino


def test_snapshot_rejects_existing_object_hash_collision(runtime):
    value, source = runtime
    store = value.ROOT / 'artifacts/runs/agentic/experiment/source_snapshot_objects'
    store.mkdir(parents=True)
    digest = value.sha256(source)
    (store / digest).write_bytes(b'wrong existing object')
    destination = value.ROOT / 'artifacts/runs/agentic/experiment/trials/a/source_snapshot/a.py'
    with pytest.raises(ValueError, match='collision/corruption'):
        value.snapshot_source(source, destination, store, digest)
    assert not destination.exists()
    assert (store / digest).read_bytes() == b'wrong existing object'


def test_snapshot_hardlink_failure_falls_back_to_isolated_copy(runtime, monkeypatch):
    value, source = runtime
    store = value.ROOT / 'artifacts/runs/agentic/experiment/source_snapshot_objects'
    destination = value.ROOT / 'artifacts/runs/agentic/experiment/trials/a/source_snapshot/a.py'
    digest = value.sha256(source)
    expected = source.read_bytes()
    def unavailable(*args):
        raise OSError('hardlinks unavailable')
    monkeypatch.setattr(value.os, 'link', unavailable)
    value.snapshot_source(source, destination, store, digest)
    source.write_bytes(b'changed')
    assert destination.read_bytes() == expected
    assert not list(store.iterdir())
