import numpy as np
import pytest
import importlib.util
from pathlib import Path

from pulsemorph.agentic.local_multifeature_reference import LocalMultiFeatureReferencePlayer
from test_agentic_multifeature_reference import player as global_player
from test_agentic_reference import fixture


def player(**changes):
    raw, old = fixture()
    args = dict(joint_names=old.joint_names, body_names=old.body_names,
                source_start_s=1., control_steps=200, source_provenance={'sha256': 'fixture'})
    args.update(changes)
    return LocalMultiFeatureReferencePlayer(raw, **args)


def test_zero_shift_exact_global_parity_with_gains():
    gains = dict(left_gain_a=.35, right_gain_a=1.6, left_gain_b=.7, right_gain_b=1.2)
    local, reference = player(**gains), global_player(**gains)[0]
    for t in (0., .321, 1.4, 1.9, 2.8, 3.99, 4., 4.14):
        assert local.source_clock(t) == reference.source_clock(t)
        for key, value in reference.state_at_time(t).items():
            np.testing.assert_array_equal(local.state_at_time(t)[key], value)


@pytest.mark.parametrize('shift', [-.3, .3])
def test_fixed_support_c1_minimum_and_clock_derivative(shift):
    local = player(timing_shift_s=shift)
    times = np.linspace(0, 4.14, 2001)
    clock = np.asarray([local.source_clock(t) for t in times])
    assert np.isfinite(clock).all() and np.all(np.diff(clock[:, 0]) > 0)
    assert local.minimum_source_rate == pytest.approx(1-.3*np.pi/1.4)
    assert clock[:, 1].min() >= local.minimum_source_rate > .25
    for t in (0., .9, 1.4, 2.8, 3.5, 4., 4.14):
        assert local.source_clock(t) == (1+t, 1.)
    for boundary in (1.4, 2.8):
        for t in (boundary-1e-8, boundary+1e-8):
            tau, rate = local.source_clock(t)
            assert tau == pytest.approx(1+boundary, abs=2e-8)
            assert rate == pytest.approx(1., abs=1e-7)
    for t in (1.5, 1.75, 2.1, 2.45, 2.7):
        epsilon = 1e-6
        derivative = (local.source_clock(t+epsilon)[0]-local.source_clock(t-epsilon)[0])/(2*epsilon)
        assert derivative == pytest.approx(local.source_clock(t)[1], abs=1e-8)


@pytest.mark.parametrize('shift', [-.3, .3])
def test_actual_edited_joint_derivative_and_finite_lookahead(shift):
    local = player(timing_shift_s=shift, left_gain_a=.35, right_gain_a=1.6,
                   left_gain_b=1.6, right_gain_b=.35)
    for t in (.75, 1.51, 1.79, 2.13, 2.49, 2.73, 3.2):
        epsilon = .001
        before, after, state = (local.state_at_time(s) for s in (t-epsilon, t+epsilon, t))
        np.testing.assert_allclose((after['dof_pos']-before['dof_pos'])/(2*epsilon),
                                   state['dof_vel'], atol=.003, rtol=.001)
    for value in local.get_future_references(199, [1, 2, 4, 8]).values():
        assert np.isfinite(value).all()
    _, metadata = local.draft_motion_reference(199)
    assert metadata['control_tool'] == 'local_clock_1p4_2p8_v1'
    assert metadata['clock_window_s'] == [1.4, 2.8]
    assert metadata['physical_acceptance'] == 'not_evaluated'


@pytest.mark.parametrize('changes', [dict(clock_window_s=(1.3, 2.8)),
    dict(clock_window_s=(1.4, np.nan)), dict(clock_window_s=(1.4,)),
    dict(timing_shift_s=.30001), dict(timing_shift_s=np.nan), dict(timing_shift_s=True),
    dict(left_gain_a=.349), dict(right_gain_b=1.601), dict(control_steps=199)])
def test_invalid_parameters(changes):
    with pytest.raises(ValueError):
        player(**changes)


@pytest.mark.parametrize('time', [-.001, 4.141, np.inf, np.nan])
def test_finite_execution_timestamps(time):
    with pytest.raises(ValueError):
        player().source_clock(time)


def test_runtime_requires_explicit_local_tool_and_discloses_producer():
    path = Path(__file__).resolve().parents[1] / 'experiments/agentic/execute_local_capability.py'
    spec = importlib.util.spec_from_file_location('local_capability_runtime_test', path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    for tool in (None, 'global', True):
        with pytest.raises(ValueError):
            runtime.proposal_producer(dict(method='calibration', control_tool=tool))
    for method in ('calibration', 'surrogate_search', 'llm_feedback', 'llm_feedback_off'):
        producer = runtime.proposal_producer(dict(method=method, control_tool='local_clock_1p4_2p8_v1'))
        assert producer['id'] == 'local-capability-'+method
    assert runtime.proposal_producer(dict(method='llm_feedback',
        control_tool='local_clock_1p4_2p8_v1'))['model'] == 'gpt-6.1-sol'
