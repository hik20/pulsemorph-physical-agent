"""Metric invariants using synthetic arrays; no physics or old-score rewrite."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('robustness_evaluation', ROOT/'experiments/agentic/evaluate_robustness.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class RobustnessEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(parameter_names=ev._base.PARAMETER_NAMES, bounds=[[.35,1.6]]*4+[[-.3,.3]],
            feature_tolerances=[.08]*4+[.1], feature_windows_s=[[.8,1.6],[2.4,3.2]],
            accent_window_s=[1.8,3.2], physical_gate=dict(min_root_height_m=.6,max_mean_joint_rmse_rad=.35))
        self.goal = dict(feature_targets=[1.]*4+[2.5])
        self.nominal = self.make_audit(0., 1.)
        self.condition = self.make_audit(.2, .5)
        self.candidate = self.make_audit(.2, .5)

    def make_audit(self, center, amplitude):
        times = np.arange(201)*.02
        q = np.zeros((201,36)); q[:,2] = .8; q[:,3] = 1
        q[:,7:] = center+amplitude; q[0,7:] = center
        vel = np.zeros((201,35)); vel[:,6:] = np.exp(-((times-2.5)/.08)**2)[:,None]
        vel[0] = 0
        before = q[0].copy(); before[7:] = 0
        record = dict(applied_once_at_time_s=0,reference_rotated=False,quaternion_order='wxyz',
            root_linear_velocity_frame='world',root_yaw_frame='world_left_multiply',
            before_qpos=before.tolist(),before_qvel=vel[0].tolist(),after_qpos=q[0].tolist(),after_qvel=vel[0].tolist(),
            spec={'synthetic_center':center})
        names = [side+'_shoulder_'+str(i) for side in ('left','right') for i in range(7)]+['other_'+str(i) for i in range(15)]
        return dict(manifest=dict(schema='pulsemorph.agentic_generalization_trial.v1',
            reference_initial_posture_is_physics_state=False,initial_perturbation=record,
            onnx_sha256='onnx',motion_sha256='source',physics_contract={},versions={},preexecution_input_hashes={}),
            arrays=dict(qpos=q,qvel=vel,state_times=times), names=names, sub=np.array([[0,0,.8]]),mean_joint_rmse_rad=.1)

    def evaluate(self):
        with patch.object(ev._base,'_audit',side_effect=[self.candidate,self.nominal,self.condition]), patch.object(ev._base,'quality_diagnostics',return_value={'diagnostic_only':True}):
            return ev.evaluate_robustness('candidate','nominal','condition',self.protocol,self.goal)

    def test_center_and_denominator_can_mask_nominal_drift(self):
        result = self.evaluate()
        self.assertTrue(result['integrity_passed'],result)
        self.assertAlmostEqual(result['primary']['features'][0], .7)
        self.assertAlmostEqual(result['secondary']['features'][0],1.)
        self.assertFalse(result['primary']['intent_passed'])
        self.assertTrue(result['secondary']['intent_passed'])
        dimension = result['dimensions'][0]
        self.assertAlmostEqual(dimension['candidate_rms_about_nominal_rad'],.7)
        self.assertAlmostEqual(dimension['candidate_rms_about_condition_rad'],.5)
        self.assertAlmostEqual(dimension['nominal_baseline_rms_rad'],1.)
        self.assertAlmostEqual(dimension['condition_baseline_rms_about_nominal_rad'],.7)
        self.assertAlmostEqual(dimension['condition_baseline_rms_rad'],.5)
        self.assertAlmostEqual(dimension['primary_absolute_rms_target_rad'],1.)
        self.assertAlmostEqual(dimension['primary_absolute_rms_tolerance_rad'],.08)

    def test_zero_condition_agrees_with_nominal_formula(self):
        self.condition = copy.deepcopy(self.nominal)
        self.candidate = copy.deepcopy(self.nominal)
        self.candidate['arrays']['qpos'][1:,7:] *= 1.04
        result = self.evaluate()
        self.assertTrue(result['integrity_passed'],result)
        self.assertEqual(result['primary'],result['secondary'])
        self.assertAlmostEqual(result['primary']['features'][0],1.04)

    def test_source_and_pin_mismatch_rejected(self):
        for key in ('motion_sha256','onnx_sha256','preexecution_input_hashes','versions','physics_contract'):
            with self.subTest(key=key):
                original = self.condition['manifest'][key]
                self.condition['manifest'][key] = 'different'
                result = self.evaluate()
                self.assertFalse(result['integrity_passed'])
                self.assertIsNone(result['primary'])
                self.condition['manifest'][key] = original

    def test_condition_initial_state_and_before_state_are_required(self):
        self.condition['arrays']['qvel'][0,0] = .01
        self.condition['manifest']['initial_perturbation']['after_qvel'][0] = .01
        self.assertFalse(self.evaluate()['integrity_passed'])
        self.condition = self.make_audit(.2,.5)
        self.candidate['manifest']['initial_perturbation']['before_qpos'][7] = .01
        self.assertFalse(self.evaluate()['integrity_passed'])

    def test_condition_initial_signed_zero_difference_is_rejected(self):
        self.condition['arrays']['qvel'][0,0] = -0.
        self.condition['manifest']['initial_perturbation']['after_qvel'][0] = -0.
        self.assertFalse(self.evaluate()['integrity_passed'])

    def test_physics_failure_keeps_primary_measurements(self):
        self.candidate['sub'][0,2] = .5
        self.condition['sub'][0,2] = .4
        self.nominal['sub'][0,2] = .3
        result = self.evaluate()
        self.assertTrue(result['integrity_passed'])
        self.assertFalse(result['physical_gate_passed'])
        self.assertFalse(result['nominal_baseline_physical']['physical_gate_passed'])
        self.assertFalse(result['condition_baseline_physical']['physical_gate_passed'])
        self.assertIsNotNone(result['primary']['features'])
        self.assertTrue(result['secondary']['intent_passed'])
        self.assertFalse(result['secondary']['joint_success'])

    def test_zero_condition_denominator_is_unavailable_secondary_only(self):
        self.condition['arrays']['qpos'][:,7:] = .2
        result = self.evaluate()
        self.assertTrue(result['integrity_passed'])
        self.assertIsNotNone(result['primary']['utility'])
        self.assertIsNone(result['secondary']['utility'])
        self.assertFalse(result['secondary']['intent_passed'])

    def test_missing_accent_stays_failed(self):
        self.candidate['arrays']['qvel'][:] = 0
        result = self.evaluate()
        self.assertTrue(result['integrity_passed'])
        self.assertIsNone(result['primary']['features'][4])
        self.assertFalse(result['primary']['per_dimension_pass'][4])

    def test_audit_failure_returns_no_fake_features(self):
        with patch.object(ev._base,'_audit',side_effect=ValueError('artifact hash mismatch')):
            result = ev.evaluate_robustness('c','n','p',self.protocol,self.goal)
        self.assertFalse(result['integrity_passed'])
        self.assertIsNone(result['primary'])
        self.assertIn('artifact hash mismatch',result['integrity_errors'])


if __name__ == '__main__':
    unittest.main()
