import numpy as np
import pytest

from pulsemorph.agentic.multifeature_reference import MultiFeatureReferencePlayer
from test_agentic_reference import fixture


def player(**changes):
    raw, old = fixture()
    args = dict(joint_names=old.joint_names, body_names=old.body_names,
                source_start_s=1., control_steps=200, source_provenance={'sha256': 'fixture'})
    args.update(changes)
    return MultiFeatureReferencePlayer(raw, **args), old


def test_identity_exact_old_player_parity_and_forecasts():
    new, old = player()
    for t in (0., .321, 1., 2., 3.99, 4., 4.14):
        for key, value in old.state_at_time(t).items():
            np.testing.assert_array_equal(new.state_at_time(t)[key], value)
    for i in (0, 75, 199):
        before, after = old.get_future_references(i, [1, 2, 4, 8]), new.get_future_references(i, [1, 2, 4, 8])
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


def test_independent_sides_sections_and_initial_boundary_parity():
    new, old = player(left_gain_a=.2, right_gain_b=1.8)
    for t in (0., 2., 4., 4.14):
        for key, value in old.state_at_time(t).items():
            np.testing.assert_array_equal(new.state_at_time(t)[key], value)
    for t, changed, same in [(1., new.side_indices[0], new.side_indices[1]),
                             (3., new.side_indices[1], new.side_indices[0])]:
        before, after = old.state_at_time(t), new.state_at_time(t)
        assert not np.array_equal(before['dof_pos'][changed], after['dof_pos'][changed])
        np.testing.assert_array_equal(before['dof_pos'][same], after['dof_pos'][same])
        np.testing.assert_array_equal(before['dof_pos'][:15], after['dof_pos'][:15])
        for key in ('body_pos', 'body_rot', 'body_vel', 'body_ang_vel'):
            np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize('shift', [-.3, .3])
def test_clock_monotone_smooth_endpoint_and_analytic_derivative(shift):
    new, old = player(timing_shift_s=shift, left_gain_a=.3, right_gain_a=1.7,
                      left_gain_b=1.6, right_gain_b=.4)
    clocks = [new.source_clock(t) for t in np.linspace(0, 4.14, 1001)]
    assert np.all(np.diff([c[0] for c in clocks]) > 0)
    assert min(c[1] for c in clocks) >= .75
    for t in (0., 4., 4.14):
        assert new.source_clock(t) == (1+t, 1.)
        for key in old.state_at_time(t):
            np.testing.assert_array_equal(new.state_at_time(t)[key], old.state_at_time(t)[key])
    for t in (.45, 1.12, 1.78, 2.13, 3.15, 3.7):
        eps = .001
        before, after, at = new.state_at_time(t-eps), new.state_at_time(t+eps), new.state_at_time(t)
        for pos, vel in [('dof_pos', 'dof_vel'), ('body_pos', 'body_vel')]:
            np.testing.assert_allclose((after[pos]-before[pos])/(2*eps), at[vel], atol=.003, rtol=.001)
        tau, rate = new.source_clock(t)
        source = new.source.state(tau, rate)
        for key in ('body_pos', 'body_rot', 'body_vel', 'body_ang_vel'):
            np.testing.assert_array_equal(at[key], source[key])


def test_archive_export_shapes_torso_and_copied_metadata():
    new, _ = player(timing_shift_s=.2)
    archive = new.archive()
    assert archive['committed__dof_pos'].shape == (201, 29)
    assert archive['forecast__dof_vel'].shape == (200, 4, 29)
    arrays, metadata = new.draft_motion_reference(199)
    np.testing.assert_array_equal(arrays['anchor_orientation_xyzw'], archive['forecast__body_rot'][199, :, 0])
    np.testing.assert_array_equal(arrays['joint_position_rad'], archive['forecast__dof_pos'][199])
    metadata['source_provenance']['sha256'] = 'mutated'
    assert new.source_provenance['sha256'] == 'fixture'
    assert metadata['shared_contract_implemented'] is False


@pytest.mark.parametrize('changes', [dict(left_gain_a=-.01), dict(right_gain_b=2.01),
    dict(left_gain_b=np.nan), dict(timing_shift_s=.36), dict(timing_shift_s=np.inf),
    dict(section_windows_s=[(0, 2.1), (2., 4.)]), dict(section_windows_s=[(0, 5), (5, 6)]),
    dict(section_windows_s=[(0, 2)]), dict(control_steps=True), dict(source_start_s=8.),
    dict(control_steps=0)])
def test_invalid_edit_and_source_bounds(changes):
    with pytest.raises(ValueError):
        player(**changes)


def test_exhaustion_and_integer_indices():
    new, _ = player()
    for t in (-.001, 4.141, np.nan):
        with pytest.raises(ValueError): new.state_at_time(t)
    for index in (-1, 201, True, .5):
        with pytest.raises(IndexError): new.get_state_at_frame(index)
    for index, offsets in [(200, [1]), (True, [1]), (0, [9]), (0, [True]), (0, [])]:
        with pytest.raises(IndexError): new.get_future_references(index, offsets)


def test_reject_source_shape_quaternion_and_order_mismatch():
    raw, old = fixture()
    args = dict(joint_names=old.joint_names, body_names=old.body_names, source_start_s=0.,
                control_steps=20, source_provenance={'id': 'x'})
    for key, value in [('dof_pos', raw['dof_pos'][:, :28]),
                       ('rigid_body_rot', np.zeros_like(raw['rigid_body_rot'])),
                       ('joint_names', list(reversed(old.joint_names)))]:
        altered = dict(raw, **{key: value})
        with pytest.raises(ValueError): MultiFeatureReferencePlayer(altered, **args)
