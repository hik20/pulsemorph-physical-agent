import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pulsemorph.agentic.reference import EditedReferencePlayer, _causal


def fixture(**updates):
    t = np.arange(501)/50
    joints = [f'leg_{i}' for i in range(15)] + [f'{side}_{part}_joint' for side in ('left', 'right')
              for part in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
    raw = dict(fps=50, dof_pos=np.tile((t*t)[:, None], (1, 29)),
               rigid_body_pos=np.stack([t, t*0, t*0], axis=-1)[:, None],
               rigid_body_rot=Rotation.from_euler('z', t*.3).as_quat()[:, None])
    options = dict(joint_names=joints, body_names=['torso_link'], source_start_s=1.,
                   control_steps=200, source_rate=1., source_provenance={'sha256': 'fixture'},
                   edit_window_s=(1., 3.))
    options.update(updates)
    return raw, EditedReferencePlayer(raw, **options)


def test_identity_exact_interpolator_parity_and_initial_pose():
    raw, player = fixture()
    source = _causal.SourceInterpolator(raw)
    for i in (0, 17, 199, 200):
        for key, value in player.get_state_at_frame(i).items():
            np.testing.assert_array_equal(value, source.state(1+i*.02, 1)[key])
    _, edited = fixture(arm_amplitude=.3)
    for key in source.state(1., 1.):
        np.testing.assert_array_equal(edited.get_state_at_frame(0)[key], player.get_state_at_frame(0)[key])


def test_identity_current_decision_player_parity():
    raw, player = fixture()
    decisions = [dict(control_index=i, t_s=i*.02, forecast=[
        dict(offset=j, source_time_s=1+(i+j)*.02, source_rate=1.)
        for j in range(10)]) for i in range(200)]
    plan = dict(schema='pulsemorph.causal_decision_plan.v1', control_dt_s=.02,
                control_steps=200, decisions=decisions,
                realized_terminal=dict(source_time_s=5., source_rate=1.))
    original = _causal.DecisionPlayer(raw, plan)
    for i in (0, 40, 199):
        before = original.get_future_references(i, [1, 2, 4, 8])
        after = player.get_future_references(i, [1, 2, 4, 8])
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize('rate,amplitude', [(1.3, 1.), (.7, .3), (1.2, 1.3)])
def test_retimed_edited_velocity_is_actual_execution_derivative(rate, amplitude):
    _, player = fixture(source_rate=rate, arm_amplitude=amplitude)
    for t in (.5, 1.4, 2., 2.6, 3.4):
        epsilon = .001
        derivative = (player.state_at_time(t+epsilon)['dof_pos']-player.state_at_time(t-epsilon)['dof_pos'])/(2*epsilon)
        np.testing.assert_allclose(player.state_at_time(t)['dof_vel'], derivative, rtol=4e-4, atol=.001)


def test_export_torso_order_no_heading_alignment_and_forecast_parity():
    _, player = fixture(arm_amplitude=.5)
    arrays, metadata = player.draft_motion_reference(80)
    refs = player.get_future_references(80, [1, 2, 4, 8])
    np.testing.assert_array_equal(arrays['joint_position_rad'], refs['dof_pos'])
    np.testing.assert_array_equal(arrays['anchor_orientation_xyzw'], refs['body_rot'][:, 0])
    np.testing.assert_allclose(np.linalg.norm(arrays['anchor_orientation_xyzw'], axis=-1), 1, atol=1e-6)
    assert metadata['heading_alignment'] == 'deferred_to_tracker_once'
    assert metadata['shared_contract_implemented'] is False
    assert 'body_pos' not in arrays
    metadata['source_provenance']['sha256'] = 'changed'
    assert player.source_provenance['sha256'] == 'fixture'
    archive = player.archive()
    assert archive['committed__dof_pos'].shape == (201, 29)
    assert archive['forecast__dof_pos'].shape == (200, 4, 29)


def test_bounded_window_and_no_silent_end_clamping():
    for changes in ({'source_rate': 0}, {'source_rate': float('nan')}, {'arm_amplitude': 2.1},
                    {'source_start_s': 8}, {'control_steps': 1.5}, {'edit_window_s': (3., 1.)},
                    {'source_provenance': {}}):
        with pytest.raises(ValueError):
            fixture(**changes)
    _, player = fixture()
    with pytest.raises(IndexError): player.get_future_references(0, [9])
    with pytest.raises(IndexError): player.get_future_references(200, [1])
    with pytest.raises(ValueError): player.state_at_time(-.1)


def test_source_shape_and_quaternion_rejected():
    raw, player = fixture()
    raw['rigid_body_rot'][0] = 0
    with pytest.raises(ValueError, match='quaternion'):
        EditedReferencePlayer(raw, joint_names=player.joint_names, body_names=player.body_names,
                              source_start_s=0, control_steps=20, source_provenance={'id': 'x'})
