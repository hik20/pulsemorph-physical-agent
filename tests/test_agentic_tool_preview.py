"""Audit gates without renderer, simulation, or model invocation."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('tool_preview', Path(__file__).parents[1] / 'experiments/agentic/tool_selection_preview.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def fixture(tmp_path):
    pilot = tmp_path / 'pilot'
    pilot.mkdir()
    (pilot / 'evidence.json').write_text('{}')
    inv = p.inventory(pilot)
    dep = Path(p.__file__).resolve()
    pins = {str(dep.relative_to(p.ROOT)): p.sha256(dep)}
    report = dict(schema='pulsemorph.agentic_tool_selection_final_audit.v1', audit_passed=True,
                  original_evidence_unchanged=True, audit_sources_unchanged=True, errors=[],
                  evidence_inventory_before=inv, evidence_inventory_after=inv,
                  source_hashes_before=pins, source_hashes_after=pins)
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(report))
    return pilot, path, report


def test_audit_binds_exact_inventory_and_current_source(tmp_path):
    pilot, path, report = fixture(tmp_path)
    assert p.verify_audit(pilot, path)[0] == report
    (pilot / 'evidence.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='actual pilot'):
        p.verify_audit(pilot, path)


@pytest.mark.parametrize('field', ['audit_passed', 'original_evidence_unchanged', 'audit_sources_unchanged'])
def test_failed_or_incomplete_audit_rejected(tmp_path, field):
    pilot, path, report = fixture(tmp_path)
    report[field] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='accepted'):
        p.verify_audit(pilot, path)


def test_stale_dependency_rejected_even_with_matching_before_after(tmp_path):
    pilot, path, report = fixture(tmp_path)
    report['source_hashes_before'] = report['source_hashes_after'] = {'experiments/agentic/tool_selection_preview.py': '0' * 64}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='source changed'):
        p.verify_audit(pilot, path)


def test_existing_output_rejected_without_render(tmp_path):
    pilot, audit, _ = fixture(tmp_path)
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        p.render(pilot, 'wave_a', output, audit)
