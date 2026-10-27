"""Two-tool dispatch/domain/provenance checks; intentionally no physics execution."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from test_agentic_reference import fixture
from pulsemorph.agentic.multifeature_reference import MultiFeatureReferencePlayer
from pulsemorph.agentic.local_multifeature_reference import LocalMultiFeatureReferencePlayer

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/agentic/execute_tool_selection.py'
spec = importlib.util.spec_from_file_location('tool_selection_runtime_test', PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
TOOLS = ('global_clock_v1', 'local_clock_1p4_2p8_v1')


def proposal(tool=TOOLS[0], **changes):
    parameters = dict(left_gain_a=1., right_gain_a=1., left_gain_b=1., right_gain_b=1., timing_shift_s=0.)
    parameters.update(changes)
    return dict(control_tool=tool, method='calibration', parameters=parameters)


def generator(p):
    raw, old = fixture()
    return runtime.build_generator(p, raw, joint_names=old.joint_names,
        body_names=old.body_names, source_start_s=1., control_steps=200,
        source_provenance={'sha256': 'fixture'}, section_windows_s=[[.4, 2.], [2., 3.6]])


@pytest.mark.parametrize('tool', [None, True, {}, [], 'global_clock', 'local_clock', 1])
def test_malformed_tool(tool):
    with pytest.raises(ValueError):
        runtime.proposal_producer(proposal(tool))


@pytest.mark.parametrize('tool', TOOLS)
@pytest.mark.parametrize('changes', [dict(left_gain_a=True), dict(right_gain_b=False),
    dict(left_gain_b=.349999), dict(right_gain_a=1.600001), dict(timing_shift_s=.300001),
    dict(timing_shift_s=-.300001), dict(timing_shift_s=True), dict(left_gain_a=float('nan')),
    dict(left_gain_a='1'), dict(timing_shift_s=None)])
def test_same_strict_bounds(tool, changes):
    with pytest.raises(ValueError):
        generator(proposal(tool, **changes))


@pytest.mark.parametrize('tool', TOOLS)
@pytest.mark.parametrize('method', ['calibration', 'surrogate_search', 'llm_feedback', 'llm_feedback_off'])
def test_actual_producer(tool, method):
    p = proposal(tool); p['method'] = method
    value = runtime.proposal_producer(p)
    assert value['id'] == 'tool-selection-' + method
    assert value['version'] == '1'
    assert value['kind'] == ('agentic' if method.startswith('llm_') else 'baseline')
    assert value.get('model') == ('gpt-6.1-sol' if method.startswith('llm_') else None)


@pytest.mark.parametrize('tool', TOOLS)
def test_extreme_valid_domain(tool):
    generator(proposal(tool, left_gain_a=.35, right_gain_a=1.6, timing_shift_s=.3))


def test_dispatch_anchor_equal_all_reference_carriers():
    a, b = (generator(proposal(tool)) for tool in TOOLS)
    assert type(a) is MultiFeatureReferencePlayer
    assert type(b) is LocalMultiFeatureReferencePlayer
    for index in [0, 1, 70, 101, 140, 199, 200]:
        for key, value in a.get_state_at_frame(index).items():
            np.testing.assert_array_equal(value, b.get_state_at_frame(index)[key])
    for index in [0, 69, 100, 199]:
        for key, value in a.get_future_references(index, [1, 2, 4, 8]).items():
            np.testing.assert_array_equal(value, b.get_future_references(index, [1, 2, 4, 8])[key])


def test_nonzero_shift_uses_correct_clocks_and_distinct_references():
    a, b = (generator(proposal(tool, timing_shift_s=.3)) for tool in TOOLS)
    for t in [.8, 1.7, 2.1, 3.2]:
        assert a.source_clock(t)[0] == pytest.approx(1+t+.3*np.sin(np.pi*t/4)**2)
        expected = 1+t+.3*np.sin(np.pi*(t-1.4)/1.4)**2 if 1.4<t<2.8 else 1+t
        assert b.source_clock(t)[0] == pytest.approx(expected)
    assert not np.array_equal(a.state_at_time(.8)['dof_pos'], b.state_at_time(.8)['dof_pos'])


def test_no_old_runtime_calls_and_exact_source_inventory():
    tree = ast.parse(PATH.read_text(encoding='utf8'))
    imported = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all('execute_generalization' not in ast.unparse(node) and
               'execute_local_capability' not in ast.unparse(node) for node in imported)
    text = PATH.read_text(encoding='utf8')
    for source in ['local_multifeature_reference.py', 'multifeature_reference.py',
                   'execute_generalization.py', 'execute_local_capability.py']:
        assert source in text
    assert 'sources = [Path(__file__)' in text
    assert 'mujoco.mj_forward(model, data)' in text
    assert "'pulsemorph.agentic_generalization_trial.v1'" in text


def test_snapshot_isolated_from_live_input(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    source = tmp_path / 'live.py'; source.write_bytes(b'original')
    dest = tmp_path / 'artifacts/runs/agentic/test/source_snapshot/live.py'
    store = tmp_path / 'artifacts/runs/agentic/objects'
    runtime.snapshot_source(source, dest, store, runtime.sha256(source))
    source.write_bytes(b'changed')
    assert dest.read_bytes() == b'original'
    assert source.stat().st_ino != dest.stat().st_ino
