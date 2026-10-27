import numpy as np
import pytest

from pulsemorph.agentic.common_player import CommonReferencePlayer, materialize_reference
from pulsemorph.agentic.multifeature_reference import MultiFeatureReferencePlayer
from pulsemorph.motion.contracts import save_reference, load_reference
from pulsemorph.motion.g1 import sample_reference
from test_agentic_reference import fixture


def make_common(tmp_path, producer_kind='agentic', **parameters):
    raw, old = fixture()
    generator = MultiFeatureReferencePlayer(raw, joint_names=old.joint_names,
        body_names=old.body_names, source_start_s=1., control_steps=200,
        source_provenance={'sha256': 'fixture'}, **parameters)
    reference = materialize_reference(generator, profile={'body_sha256':'a'*64, 'yaml_sha256':'b'*64},
        candidate_id='fixture', observation_id='fixture-observation', task_id='fixture-task',
        runtime_session_id='fixture-runtime', source_path='fixture.motion', source_sha256='c'*64,
        audio_observed_through_sample=88200, audio_duration_s=4., producer_kind=producer_kind)
    save_reference(reference, tmp_path/'common')
    loaded = load_reference(tmp_path/'common')
    return generator, CommonReferencePlayer(generator, loaded)


def test_loaded_common_controls_all_actual_reference_inputs(tmp_path):
    generator, player = make_common(tmp_path, left_gain_a=.4, right_gain_b=1.5, timing_shift_s=.3)
    assert len(player.reference.time_offsets_s) == 208
    for index in (0, 19, 67, 130, 199):
        future = player.get_future_references(index, [1, 2, 4, 8])
        for offset, frame in enumerate([1, 2, 4, 8]):
            expected = generator.state_at_time((index+frame)*.02)
            for key in ('dof_pos', 'dof_vel', 'body_rot'):
                np.testing.assert_allclose(future[key][offset], expected[key], rtol=0, atol=1e-7)
    for t in (.315, 1.711, 2.825, 4.133):
        common = sample_reference(player.reference, [t])
        actual, carrier = player.state_at_time(t), generator.state_at_time(t)
        np.testing.assert_array_equal(actual['dof_pos'], common['dof_pos'][0].astype('float32'))
        np.testing.assert_array_equal(actual['dof_vel'], common['dof_vel'][0].astype('float32'))
        np.testing.assert_array_equal(actual['body_rot'][0], common['anchor_rot'][0].astype('float32'))
        for key in ('body_pos', 'body_vel', 'body_ang_vel'):
            np.testing.assert_array_equal(actual[key], carrier[key])
    assert player.reference.metadata['observation']['observed_through_time_s'] == 4.
    assert player.reference.metadata['validity']['end_time_s'] == 4.14


def test_common_hermite_derivative_and_exact_knot_velocity(tmp_path):
    generator, player = make_common(tmp_path, left_gain_a=.4, timing_shift_s=-.2)
    for t in (.713, 1.289, 2.677, 3.371):
        epsilon = .0001
        minus, plus = sample_reference(player.reference, [t-epsilon]), sample_reference(player.reference, [t+epsilon])
        value = sample_reference(player.reference, [t])
        np.testing.assert_allclose((plus['dof_pos']-minus['dof_pos'])/(2*epsilon), value['dof_vel'], atol=2e-6, rtol=1e-6)
    for t in player.reference.time_offsets_s:
        np.testing.assert_allclose(player.state_at_time(t)['dof_vel'], generator.state_at_time(t)['dof_vel'], rtol=0, atol=0)


def test_export_initial_parity_and_exhaustion(tmp_path):
    generator, player = make_common(tmp_path)
    for key, value in generator.get_state_at_frame(0).items():
        np.testing.assert_array_equal(player.get_state_at_frame(0)[key], value)
    arrays, metadata = player.draft_motion_reference(50)
    assert metadata['shared_contract_implemented'] is True
    np.testing.assert_array_equal(arrays['joint_position_rad'], player.get_future_references(50, [1, 2, 4, 8])['dof_pos'])
    with pytest.raises(ValueError): player.state_at_time(4.15)
    with pytest.raises(IndexError): player.get_state_at_frame(201)
    with pytest.raises(IndexError): player.get_future_references(200, [1])


def test_baseline_producer_survives_save_load_and_unknown_kind_rejected(tmp_path):
    _, player = make_common(tmp_path, producer_kind='baseline')
    assert player.reference.metadata['producer']['kind'] == 'baseline'
    with pytest.raises(ValueError, match='producer kind'):
        make_common(tmp_path/'invalid', producer_kind='unrecognized')
