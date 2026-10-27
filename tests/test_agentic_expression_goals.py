"""Pure measured-component goal contracts; no controller or compiler calls."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pulsemorph.agentic.expression.goals_v1 import (
    parse_goals, evaluate_goals, select_final, GOALS_SCHEMA, DENOMINATOR_FLOOR)
from pulsemorph.agentic.phrase_expression_v1 import GAIN_FIELDS


GOALS = {'schema': 'pulsemorph.expression_goals.v1', 'mean_pose_ratio': .8, 'temporal_ratio': 1.}


def trial(index=-1, mean=1., temporal=1., gate=True):
    expression = {'schema': 'pulsemorph.phrase_expression.v1', 'order': ['wave', 'high5'],
                  'heading_policy': 'align_join_heading', 'rationale': 'Synthetic fixture',
                  **{key: 1. for key in GAIN_FIELDS}}
    rows = [{'phase': phase, 'side': side, 'window_s': [1.2, 3.2] if phase == 0 else [5.2, 7.2],
             'sample_count': 101, 'centered_temporal_rms_rad': temporal if (phase, side) == (1, 'left') else .4,
             'mean_pose_offset_rms_rad': mean if (phase, side) == (1, 'left') else .3,
             'per_joint_peak_to_peak_rad': [1.] * 7}
            for phase, side in ((0, 'left'), (0, 'right'), (1, 'left'), (1, 'right'))]
    return {'id': 'fixture__baseline' if index == -1 else 'fixture__trial_' + str(index),
            'index': index, 'expression': expression, 'diagnostics': rows, 'integrity_passed': True,
            'metrics': {'arm_rms_rad': [.2, .4, .6, .8], 'min_root_height_m': .8 if gate else .5,
                        'mean_joint_rmse_rad': .1, 'physical_gate_passed': gate,
                        'actual_physics_steps': 8000, 'online_onnx_call_count': 400},
            'source_identity': {'schema': 'pulsemorph.expression_source_identity.v1',
                                'implementation_sha256': 'a' * 64, 'environment_sha256': 'b' * 64,
                                'music_sha256': 'c' * 64, 'source_id': 'phrase_wave_high5_v1',
                                'duration_s': 8, 'sample_period_s': .02}}


class GoalTests(unittest.TestCase):
    def test_strict_goals_and_owned_output(self):
        parsed = parse_goals(GOALS)
        parsed['mean_pose_ratio'] = 1.
        self.assertEqual(GOALS['mean_pose_ratio'], .8)
        self.assertFalse(GOALS_SCHEMA['additionalProperties'])
        cases = [{}, {**GOALS, 'tolerance': 1}, {**GOALS, 'schema': 'old'}]
        for value in (True, None, float('nan'), float('inf'), 10**1000, .7499, 1.2501, '1'):
            cases.append({**GOALS, 'mean_pose_ratio': value})
        for value in cases:
            with self.subTest(value=str(value)[:50]), self.assertRaises(ValueError):
                parse_goals(value)

    def test_two_new_metrics_not_own_q0_score_and_no_alias(self):
        base = trial()
        candidate = trial(0, .8, 1.)
        candidate['metrics']['arm_rms_rad'] = [99.] * 4
        evaluated = evaluate_goals(candidate, base, GOALS)
        self.assertEqual(evaluated['features'], [.8, 1.])
        self.assertEqual(evaluated['normalized_errors'], [0., 0.])
        self.assertTrue(evaluated['intent_passed'])
        self.assertEqual(evaluated['assessment'], 'target_met')
        self.assertEqual(evaluated['goal_count'], 2)
        evaluated['goals']['mean_pose_ratio'] = 1.2
        evaluated['actual_components_rad'][0] = 90
        self.assertEqual(GOALS['mean_pose_ratio'], .8)
        self.assertEqual(candidate['diagnostics'][2]['mean_pose_offset_rms_rad'], .8)

    def test_nearzero_denominator_explicit_and_no_q0_fallback(self):
        for value in (0., DENOMINATOR_FLOOR / 2):
            base = trial(mean=value)
            result = evaluate_goals(trial(0, .8, 1.), base, GOALS)
            self.assertFalse(result['assessable'])
            self.assertEqual(result['assessment'], 'unassessable')
            self.assertIsNone(result['features'])
            self.assertIsNone(result['worst_error'])
            self.assertEqual(result['goals_passed'], 0)
            self.assertFalse(result['intent_passed'])
            json.dumps(result, allow_nan=False)
        result = evaluate_goals(trial(0, DENOMINATOR_FLOOR * .8), trial(mean=DENOMINATOR_FLOOR), GOALS)
        self.assertTrue(result['assessable'])
        self.assertAlmostEqual(result['features'][0], .8)

    def test_zero_candidate_observed_and_ratio_overflow_unassessable(self):
        result = evaluate_goals(trial(0, 0., 0.), trial(), GOALS)
        self.assertTrue(result['assessable'])
        self.assertEqual(result['features'], [0., 0.])
        self.assertFalse(result['intent_passed'])
        result = evaluate_goals(trial(0, 1e308), trial(mean=1e-8), GOALS)
        self.assertEqual(result['reason'], 'nonfinite_ratio_evaluation')
        json.dumps(result, allow_nan=False)

    def test_diagnostic_shape_window_pair_and_finite_tamper_rejected(self):
        changes = [lambda t: t['diagnostics'].pop(),
                   lambda t: t['diagnostics'][2].update(phase=True),
                   lambda t: t['diagnostics'][2].update(window_s=[1.2, 3.2]),
                   lambda t: t['diagnostics'][2].update(sample_count=True),
                   lambda t: t['diagnostics'][2].update(sample_count=100),
                   lambda t: t['diagnostics'][2].update(side='right'),
                   lambda t: t['diagnostics'][2].update(mean_pose_offset_rms_rad=float('nan')),
                   lambda t: t['diagnostics'][2].update(centered_temporal_rms_rad=10**1000),
                   lambda t: t['diagnostics'][2].update(mean_pose_offset_rms_rad=-1),
                   lambda t: t['diagnostics'][2].update(per_joint_peak_to_peak_rad=[1.] * 6),
                   lambda t: t['diagnostics'][2].update(extra=0)]
        for change in changes:
            candidate = trial(0)
            change(candidate)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_goals(candidate, trial(), GOALS)

    def test_source_heading_baseline_expression_and_identity_tamper_rejected(self):
        changes = [lambda t: t['source_identity'].update(music_sha256='d' * 64),
                   lambda t: t['source_identity'].update(implementation_sha256='d' * 64),
                   lambda t: t['source_identity'].update(duration_s=8.),
                   lambda t: t['source_identity'].update(extra=True),
                   lambda t: t['expression'].update(heading_policy='recorded_world'),
                   lambda t: t['expression'].update(order=['high5', 'wave']),
                   lambda t: t.update(id='fixture__baseline'),
                   lambda t: t.update(index=True),
                   lambda t: t.update(integrity_passed=False),
                   lambda t: t.update(schema='old_trial')]
        for change in changes:
            candidate = trial(0)
            change(candidate)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_goals(candidate, trial(), GOALS)
        base = trial()
        base['expression']['first_left_motion_gain'] = .9
        with self.assertRaises(ValueError):
            evaluate_goals(trial(0), base, GOALS)

    def test_raw_physical_wrapper_and_flag_consistency(self):
        candidate = trial(0, .8, 1.)
        candidate['physical_metrics'] = copy.deepcopy(candidate['metrics'])
        candidate['metrics'] = {'features': ['UI aliases are not physical proof']}
        self.assertTrue(evaluate_goals(candidate, trial(), GOALS)['intent_passed'])
        candidate['physical_metrics']['physical_gate_passed'] = False
        with self.assertRaises(ValueError):
            evaluate_goals(candidate, trial(), GOALS)
        candidate = trial(0, .8, 1., gate=False)
        evaluated = evaluate_goals(candidate, trial(), GOALS)
        self.assertEqual(evaluated['goals_passed'], 2)
        self.assertEqual(evaluated['assessment'], 'physical_gate_failed')
        self.assertFalse(evaluated['intent_passed'])

    def test_selection_physical_priority_baseline_eligible_and_earlier_index(self):
        baseline = trial()
        failed_gate = trial(0, .8, 1., gate=False)
        self.assertEqual(select_final([failed_gate], baseline, GOALS)['trial']['index'], -1)
        a, b = trial(0, .81, 1.), trial(1, .81, 1.)
        self.assertEqual(select_final([b, a], baseline, GOALS)['trial']['index'], 0)
        rejected = {'id': 'rejected__0', 'index': 2, 'status': 'rejected', 'metrics': None}
        selected = select_final([rejected, a], baseline, GOALS)
        self.assertEqual(selected['trial']['index'], 0)
        selected['trial']['diagnostics'][2]['mean_pose_offset_rms_rad'] = 0
        self.assertEqual(a['diagnostics'][2]['mean_pose_offset_rms_rad'], .81)
        with self.assertRaises(ValueError):
            select_final([{**rejected, 'metrics': {}}], baseline, GOALS)

    def test_selection_worst_error_then_l2_then_index_and_duplicate_fail(self):
        # Both max errors .5; only second dimension differs, so L2 decides.
        a, b = trial(0, .84, 1.032), trial(1, .84, 1.008)
        selected = select_final([a, b], trial(), GOALS)
        self.assertEqual(selected['trial']['index'], 1)
        with self.assertRaises(ValueError):
            select_final([a, copy.deepcopy(a)], trial(), GOALS)
        undefined = trial(mean=0.)
        self.assertEqual(select_final([trial(0)], undefined, GOALS)['trial']['index'], -1)


if __name__ == '__main__':
    unittest.main()
