import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('robustness_runtime', ROOT / 'experiments/agentic/execute_robustness.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def zero_spec():
    return {'joint_position_offset_rad': [0.] * 29,
            'joint_velocity_offset_rad_s': [0.] * 29,
            'root_linear_velocity_offset_world_m_s': [0.] * 3,
            'root_yaw_offset_rad': 0.}


def model_data():
    # Independent root orientation makes local/world translation distinguishable.
    children = ''.join(f'<body pos="0 0 .03"><joint name="j{i}" type="hinge" range="-1 1"/><geom type="sphere" size=".01" mass=".01"/></body>' for i in range(29))
    model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".03" mass="1"/>' + children + '</body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(.6), 0., np.sin(.6), 0.]
    mujoco.mj_forward(model, data)
    return model, data


class RobustnessRuntimeTests(unittest.TestCase):
    def test_zero_spec_and_noop_bit_parity(self):
        model, data = model_data()
        qpos, qvel = data.qpos.tobytes(), data.qvel.tobytes()
        proxy = Mock(wraps=mujoco)
        proxy.mjtJoint = mujoco.mjtJoint
        record = runtime.apply_initial_perturbation(model, data, zero_spec(), proxy)
        self.assertEqual(qpos, data.qpos.tobytes())
        self.assertEqual(qvel, data.qvel.tobytes())
        proxy.mj_forward.assert_not_called()
        self.assertFalse(record['nonzero'])

    def test_offsets_yaw_world_and_forward(self):
        model, data = model_data()
        before = data.qpos.copy()
        perturb = zero_spec()
        perturb['joint_position_offset_rad'][4] = .02
        perturb['joint_velocity_offset_rad_s'][7] = -.1
        perturb['root_linear_velocity_offset_world_m_s'] = [.05, -.03, .01]
        perturb['root_yaw_offset_rad'] = .02
        proxy = Mock(wraps=mujoco)
        proxy.mjtJoint = mujoco.mjtJoint
        record = runtime.apply_initial_perturbation(model, data, perturb, proxy)
        expected = np.empty(4)
        mujoco.mju_mulQuat(expected, np.array([np.cos(.01), 0, 0, np.sin(.01)]), before[3:7])
        np.testing.assert_allclose(data.qpos[3:7], expected, rtol=0, atol=1e-15)
        self.assertEqual(data.qpos[11], .02)
        self.assertEqual(data.qvel[13], -.1)
        np.testing.assert_array_equal(data.qvel[:3], [.05, -.03, .01])
        proxy.mj_forward.assert_called_once()
        self.assertEqual(record['applied_once_at_time_s'], 0.)
        self.assertNotEqual(record['before_qpos_sha256'], record['after_qpos_sha256'])

    def test_freejoint_linear_velocity_is_world_frame(self):
        model, data = model_data()
        position = data.qpos[:3].copy()
        data.qvel[:3] = [.04, -.02, .01]
        # MuJoCo's actual position integrator, independent of injection code.
        mujoco.mj_integratePos(model, data.qpos, data.qvel, .1)
        np.testing.assert_allclose(data.qpos[:3] - position, [.004, -.002, .001], atol=1e-15)

    def test_joint_bound_failure_is_atomic(self):
        model, data = model_data()
        data.qpos[7] = .99
        before = data.qpos.tobytes()
        perturb = zero_spec()
        perturb['joint_position_offset_rad'][0] = .02
        with self.assertRaisesRegex(ValueError, 'joint limit'):
            runtime.apply_initial_perturbation(model, data, perturb)
        self.assertEqual(before, data.qpos.tobytes())

    def test_validation(self):
        for key, value in [('root_yaw_offset_rad', True), ('root_yaw_offset_rad', float('nan')),
                           ('root_yaw_offset_rad', .04), ('joint_position_offset_rad', [0.] * 28),
                           ('joint_velocity_offset_rad_s', [float('inf')] * 29),
                           ('root_linear_velocity_offset_world_m_s', [.051, 0, 0])]:
            perturb = zero_spec(); perturb[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    runtime.validate_initial_perturbation(perturb)
        perturb = zero_spec(); perturb['unknown'] = 0
        with self.assertRaises(ValueError): runtime.validate_initial_perturbation(perturb)

    def test_nonzero_time_rejected(self):
        model, data = model_data(); data.time = .001
        with self.assertRaisesRegex(ValueError, 'time zero'):
            runtime.apply_initial_perturbation(model, data, zero_spec())

    def test_real_reset_hook_calls_once_then_captures_perturbed_state(self):
        import textwrap
        source = (ROOT / 'experiments/agentic/execute_robustness.py').read_text()
        start = source.index('        def initial(model, data, motion_player):')
        end = source.index('        def step(model, data):', start)
        model, data = model_data()
        perturb = zero_spec(); perturb['joint_position_offset_rad'][0] = .01
        reset = Mock()
        capture = Mock()
        states, velocities, times = [], [], []
        counts = {'initialization': 0, 'substeps': 0}
        manifest = {}
        env = dict(real_initial=reset, apply_initial_perturbation=runtime.apply_initial_perturbation,
                   perturbation=perturb, mujoco=mujoco, counts=counts, states=states,
                   velocities=velocities, state_times=times, manifest=manifest,
                   sha256_array=runtime.sha256_array, capture=capture)
        factory = 'def factory():\n    warnings_initial = None\n' + textwrap.indent(textwrap.dedent(source[start:end]), '    ') + '\n    return initial\n'
        exec(factory, env)
        hook = env['factory']()
        hook(model, data, object())
        reset.assert_called_once()
        capture.assert_called_once()
        self.assertEqual(times, [0.])
        self.assertEqual(states[0][7], .01)
        self.assertEqual(manifest['initial_perturbation']['before_qpos'][7], 0.)
        with self.assertRaisesRegex(RuntimeError, 'unexpected simulation reset'):
            hook(model, data, object())
        reset.assert_called_once()
        self.assertEqual(len(states), 1)

    def test_hook_precedes_capture(self):
        source = (ROOT / 'experiments/agentic/execute_robustness.py').read_text()
        start = source.index('        def initial(model, data, motion_player):')
        end = source.index('        def step(model, data):', start)
        hook = source[start:end]
        self.assertEqual(hook.count('real_initial(model, data, motion_player)'), 1)
        self.assertEqual(hook.count('apply_initial_perturbation('), 1)
        self.assertLess(hook.index('real_initial('), hook.index('apply_initial_perturbation('))
        self.assertLess(hook.index('apply_initial_perturbation('), hook.index('states.append('))


if __name__ == '__main__': unittest.main()

def test_actual_strict_g1_metadata_roundtrip(tmp_path):
    from pulsemorph.agentic.common_player import materialize_reference
    from pulsemorph.motion.contracts import save_reference, load_reference
    from pulsemorph.motion.g1 import profile_from_yaml
    profile = profile_from_yaml()
    times = np.arange(151) / 30
    raw = dict(fps=30, dof_pos=np.zeros((151, 29)),
               rigid_body_pos=np.zeros((151, len(profile['body_names']), 3)),
               rigid_body_rot=np.broadcast_to([0., 0., 0., 1.], (151, len(profile['body_names']), 4)).copy())
    for tool in ['global_clock_v1', 'local_clock_1p4_2p8_v1']:
        proposal = dict(control_tool=tool, method='llm_feedback', parameters=dict(
            left_gain_a=.8, right_gain_a=1.2, left_gain_b=1.2, right_gain_b=.8, timing_shift_s=-.3))
        generator = runtime.build_generator(proposal, raw, joint_names=profile['joint_names'],
            body_names=profile['body_names'], source_start_s=0., control_steps=200,
            section_windows_s=[[.4, 2.], [2., 3.6]], source_provenance={'sha256': 'a'*64})
        reference = materialize_reference(generator, profile=profile, candidate_id='test',
            observation_id='test', task_id='test', runtime_session_id='test', source_path='fixture.motion',
            source_sha256='a'*64, audio_observed_through_sample=88200, audio_duration_s=4.,
            protocol_sha256='b'*64, producer_kind='agentic')
        producer = runtime.proposal_producer(proposal); producer['control_tool'] = tool
        reference = runtime.compatible_metadata(reference, producer)
        save_reference(reference, tmp_path/tool)
        loaded = load_reference(tmp_path/tool)
        assert runtime.strict_g1_preflight(loaded) == 2
        assert set(loaded.metadata) == {'schema_version', 'reference_id', 'producer', 'body', 'controller',
            'frame', 'context', 'validity', 'observation', 'provenance'}
        assert loaded.metadata['producer']['id'] == 'robustness-llm_feedback'
