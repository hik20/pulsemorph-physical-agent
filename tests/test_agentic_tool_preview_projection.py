"""Explicit projection audit gates; no rendering or new physics."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('projection_preview', Path(__file__).parents[1] / 'experiments/agentic/tool_selection_preview_projection.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def accepted_fixture(tmp_path):
    pilot, projection = tmp_path / 'pilot', tmp_path / 'projection'
    pilot.mkdir(); projection.mkdir()
    (pilot / 'evidence.json').write_text('{}')
    current = p.inventory(pilot)
    failed = tmp_path / 'failed.json'
    write(failed, dict(schema='pulsemorph.agentic_tool_selection_final_audit.v2', audit_passed=False,
                       errors=['ValueError: unsupported target metadata fields']))
    failed_link = dict(path=str(failed), sha256=p.sha256(failed))
    index = dict(schema='pulsemorph.agentic_tool_reference_projection.v1', pilot=str(pilot),
        original_strict_g1_admission=False, projection_strict_g1_admission=True,
        numerical_only_no_resimulation=True, reference_count=100, references=[{} for _ in range(100)],
        original_inventory_before=current, original_inventory_after=current, failed_audit=failed_link)
    write(projection / 'index.json', index)
    pins = {name: p.sha256(p.ROOT / name) for name in p.REQUIRED_AUDIT_SOURCES}
    report = dict(schema='pulsemorph.agentic_tool_selection_projection_audit.v1',
        audit_passed=True, errors=[], original_strict_g1_admission=False, compatible_projection_required=True,
        projection_raw_arrays_unchanged=True, original_evidence_unchanged=True, audit_sources_unchanged=True,
        compatible_export_unchanged=True, failed_raw_audit_unchanged=True, compatible_projection_root=str(projection),
        evidence_inventory_before=current, evidence_inventory_after=current,
        projection_inventory_before=p.inventory(projection), projection_inventory_after=p.inventory(projection),
        projection_index_sha256=p.sha256(projection / 'index.json'), failed_raw_audit=failed_link,
        source_hashes_before=pins, source_hashes_after=pins,
        trials={str(i): dict(common_parity=dict(original_strict_g1_admission=False,
            compatible_projection_admitted=True, raw_array_bytes_unchanged=True, save_load_passed=True,
            decisions=200, future_samples=800)) for i in range(100)})
    audit = tmp_path / 'audit.json'
    write(audit, report)
    return pilot, projection, audit, report


def test_accepted_schema_and_numeric_evidence_fixture(tmp_path):
    pilot, projection, audit, report = accepted_fixture(tmp_path)
    assert p.verify_audit(pilot, projection, audit)[0] == report


@pytest.mark.parametrize('key,value', [('original_strict_g1_admission', True),
    ('compatible_projection_required', None), ('projection_raw_arrays_unchanged', False),
    ('compatible_export_unchanged', False), ('failed_raw_audit_unchanged', False)])
def test_unknown_or_wrong_projection_flags_rejected(tmp_path, key, value):
    pilot, projection, audit, report = accepted_fixture(tmp_path)
    report[key] = value
    write(audit, report)
    with pytest.raises(ValueError, match='explicit projection'):
        p.verify_audit(pilot, projection, audit)


def test_failed_raw_audit_cannot_be_substituted(tmp_path):
    pilot, projection, audit, report = accepted_fixture(tmp_path)
    write(audit, dict(schema='pulsemorph.agentic_tool_selection_final_audit.v2', audit_passed=False))
    with pytest.raises(ValueError, match='explicit projection'):
        p.verify_audit(pilot, projection, audit)


@pytest.mark.parametrize('tree', ['pilot', 'projection'])
def test_evidence_cannot_be_swapped_after_audit(tmp_path, tree):
    pilot, projection, audit, _ = accepted_fixture(tmp_path)
    path = pilot if tree == 'pilot' else projection
    (path / 'new.json').write_text('{}')
    with pytest.raises(ValueError, match='actual .* evidence'):
        p.verify_audit(pilot, projection, audit)


def test_partial_numeric_parity_rejected(tmp_path):
    pilot, projection, audit, report = accepted_fixture(tmp_path)
    report['trials']['0']['common_parity']['decisions'] = 1
    write(audit, report)
    with pytest.raises(ValueError, match='numeric projection parity'):
        p.verify_audit(pilot, projection, audit)


def test_failed_audit_hash_changed_rejected(tmp_path):
    pilot, projection, audit, report = accepted_fixture(tmp_path)
    Path(report['failed_raw_audit']['path']).write_text('{}')
    with pytest.raises(ValueError, match='failed raw audit hash'):
        p.verify_audit(pilot, projection, audit)
