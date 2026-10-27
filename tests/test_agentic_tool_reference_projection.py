"""Actual recorded raw package admission and metadata-only projection invariants."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('tool_projection', ROOT / 'experiments/agentic/project_tool_reference.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
TRIAL = ROOT / 'artifacts/runs/agentic/tool_selection_development_v2/trials/calibration_wave'


def recorded():
    return p.load_reference(TRIAL / 'common_reference'), p.read(TRIAL / 'manifest.json')


def test_actual_raw_rejection_projected_admission_and_roundtrip_bytes(tmp_path):
    raw, manifest = recorded()
    before = p.inventory(TRIAL / 'common_reference')
    with pytest.raises(ValueError, match='unsupported target metadata fields'):
        p.assemble_reference_inputs(raw, 0.)
    projected, disclosure = p.project_reference(raw, manifest)
    assert not disclosure['original_strict_g1_admission']
    assert disclosure['projection_strict_g1_admission']
    expected = p._plain(raw.metadata)
    for key in ('control_tool', 'experiment_kind'):
        del expected[key]
    assert p._plain(projected.metadata) == expected
    assert projected.metadata['producer']['control_tool'] == raw.metadata['producer']['control_tool']
    saved = p.save_reference(projected, tmp_path / 'projected')
    loaded = p.load_reference(saved)
    p.array_proof(raw, loaded)
    with np.load(TRIAL / 'common_reference/arrays.npz', allow_pickle=False) as a, np.load(saved.with_name('arrays.npz'), allow_pickle=False) as b:
        for name in p.ARRAYS:
            assert a[name].dtype == b[name].dtype
            assert a[name].shape == b[name].shape
            assert a[name].tobytes() == b[name].tobytes()
    for offset in [0., 3.98]:
        assembled = p.assemble_reference_inputs(loaded, offset)
        sampled = p.sample_reference(raw, offset + p.FUTURE_OFFSETS_S)
        for out, name in [('mimic_future_dof_pos', 'dof_pos'), ('mimic_future_dof_vel', 'dof_vel'), ('mimic_future_anchor_rot', 'anchor_rot')]:
            assert assembled[out].tobytes() == sampled[name][None].astype('float32').tobytes()
    assert p.inventory(TRIAL / 'common_reference') == before


@pytest.mark.parametrize('field,value', [('unknown', 'reject'), ('control_tool', 'local_clock_1p4_2p8_v1'), ('experiment_kind', 'different')])
def test_unknown_or_conflicting_metadata_rejected(field, value):
    raw, manifest = recorded()
    metadata = p._plain(raw.metadata)
    metadata[field] = value
    altered = replace(raw, metadata=metadata)
    with pytest.raises(ValueError):
        p.project_reference(altered, manifest)


def test_manifest_producer_conflict_rejected():
    raw, manifest = recorded()
    manifest['producer']['id'] = 'wrong-producer'
    with pytest.raises(ValueError, match='conflict'):
        p.project_reference(raw, manifest)


def test_other_supported_metadata_retained():
    raw, manifest = recorded()
    metadata = p._plain(raw.metadata)
    metadata['context']['review_extra'] = {'keep': [1, 2]}
    projected, _ = p.project_reference(replace(raw, metadata=metadata), manifest)
    assert p._plain(projected.metadata)['context']['review_extra'] == {'keep': [1, 2]}


def test_existing_output_rejected(tmp_path):
    with pytest.raises(ValueError, match='fresh output'):
        p.export(TRIAL, tmp_path, tmp_path / 'not-read.json')
