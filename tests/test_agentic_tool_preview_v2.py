"""Audit gates without renderer, simulation, or model invocation."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('tool_preview_v2', Path(__file__).parents[1] / 'experiments/agentic/tool_selection_preview_v2.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def fixture(tmp_path):
    pilot = tmp_path / 'pilot'
    pilot.mkdir()
    (pilot / 'evidence.json').write_text('{}')
    inv = p.inventory(pilot)
    dep = Path(p.__file__).resolve()
    pins = {str(dep.relative_to(p.ROOT)): p.sha256(dep)}
    pins.update({name: p.sha256(p.ROOT / name) for name in p.REQUIRED_AUDIT_SOURCES})
    report = dict(schema='pulsemorph.agentic_tool_selection_final_audit.v2', audit_passed=True,
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
    report['source_hashes_before']['experiments/agentic/tool_selection_preview_v2.py'] = '0' * 64
    report['source_hashes_after'] = dict(report['source_hashes_before'])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='source changed'):
        p.verify_audit(pilot, path)


def test_existing_output_rejected_without_render(tmp_path):
    pilot, audit, _ = fixture(tmp_path)
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        p.render(pilot, 'wave_a', output, audit)


def test_v1_audit_rejected(tmp_path):
    pilot, path, report = fixture(tmp_path)
    report['schema'] = 'pulsemorph.agentic_tool_selection_final_audit.v1'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='accepted'):
        p.verify_audit(pilot, path)


def test_v2_source_inventory_required(tmp_path):
    pilot, path, report = fixture(tmp_path)
    missing = next(iter(p.REQUIRED_AUDIT_SOURCES))
    del report['source_hashes_before'][missing]
    report['source_hashes_after'] = dict(report['source_hashes_before'])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not stable'):
        p.verify_audit(pilot, path)


def test_compact_titles_fit_shared_renderer_panel():
    from PIL import ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    labels = ['DEV BASE | GLOBAL | 1P/0M']
    labels += [f'DEV {method} | {tool} | 8P/{calls}M'
               for method, calls in [('NUMERIC', 0), ('LLM+FB', 8), ('LLM-OFF', 8)]
               for tool in ['GLOBAL', 'LOCAL']]
    assert all(font.getlength(label) <= 456 for label in labels)
