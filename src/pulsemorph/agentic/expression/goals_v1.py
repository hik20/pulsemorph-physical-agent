"""Fixed physical component goals, distinct from eight reference controls.

Pure DTO evaluation only. The caller must first verify every saved trial with
ExpressionRuntime and attach source_identity derived from that verified runtime.
That identity is a comparison key, not a standalone integrity attestation.
"""
import copy
import json
import math
import re
import sys

from ..phrase_expression_v1 import parse_expression, GAIN_FIELDS

SCHEMA = 'pulsemorph.expression_goals.v1'
TOLERANCE = .08
DENOMINATOR_FLOOR = 1e-8
GOALS_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'schema': {'const': SCHEMA},
        'mean_pose_ratio': {'type': 'number', 'minimum': .75, 'maximum': 1.25},
        'temporal_ratio': {'type': 'number', 'minimum': .75, 'maximum': 1.25}},
    'required': ['schema', 'mean_pose_ratio', 'temporal_ratio']}
_IDENTITY_KEYS = {'schema', 'implementation_sha256', 'environment_sha256',
                  'music_sha256', 'source_id', 'duration_s', 'sample_period_s'}
_ROW_KEYS = {'phase', 'side', 'window_s', 'sample_count',
             'centered_temporal_rms_rad', 'mean_pose_offset_rms_rad',
             'per_joint_peak_to_peak_rad'}
_METRIC_KEYS = {'arm_rms_rad', 'min_root_height_m', 'mean_joint_rmse_rad',
                'physical_gate_passed', 'actual_physics_steps', 'online_onnx_call_count'}


def parse_goals(value):
    """Return owned canonical targets; no caller-defined metric/window/tolerance."""
    if type(value) is not dict or set(value) != set(GOALS_SCHEMA['required']):
        raise ValueError('Exact expression goal fields required')
    if type(value['schema']) is not str or value['schema'] != SCHEMA:
        raise ValueError('Expression goal schema required')
    ratios = {}
    for key in ('mean_pose_ratio', 'temporal_ratio'):
        v = value[key]
        # Range comparison precedes isfinite/conversion, including huge integers.
        if type(v) not in (int, float) or not .75 <= v <= 1.25 or not math.isfinite(v):
            raise ValueError('Goal ratios must be finite numbers in [.75,1.25]')
        ratios[key] = float(v)
    return {'schema': SCHEMA, **ratios}


def _number(value, *, nonnegative=True):
    if (type(value) not in (int, float) or
            not -sys.float_info.max <= value <= sys.float_info.max):
        raise ValueError('Finite physical diagnostic number required')
    v = float(value)
    if not math.isfinite(v) or (nonnegative and v < 0):
        raise ValueError('Finite nonnegative diagnostic required')
    return v


def _identity(value):
    if type(value) is not dict or set(value) != _IDENTITY_KEYS:
        raise ValueError('Verified expression source identity required')
    if (value['schema'] != 'pulsemorph.expression_source_identity.v1' or
            type(value['schema']) is not str or
            type(value['source_id']) is not str or value['source_id'] != 'phrase_wave_high5_v1' or
            type(value['duration_s']) is not int or value['duration_s'] != 8 or
            type(value['sample_period_s']) is not float or value['sample_period_s'] != .02):
        raise ValueError('Fixed source identity/clocks required')
    for key in ('implementation_sha256', 'environment_sha256', 'music_sha256'):
        if type(value[key]) is not str or re.fullmatch('[a-f0-9]{64}', value[key]) is None:
            raise ValueError('Lowercase source identity SHA256 required')
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _trial(trial):
    required = {'id', 'index', 'expression', 'metrics', 'diagnostics',
                'integrity_passed', 'source_identity'}
    if type(trial) is not dict or not required <= set(trial):
        raise ValueError('Verified measured expression trial required')
    if (type(trial['id']) is not str or re.fullmatch('[A-Za-z0-9_-]{1,200}', trial['id']) is None or
            type(trial['index']) is not int or trial['index'] < -1 or
            trial['integrity_passed'] is not True):
        raise ValueError('Verified trial ID/index/integrity required')
    if 'schema' in trial and trial['schema'] != 'pulsemorph.expression_physical_trial.v1':
        raise ValueError('Physical expression trial schema required')
    if trial.get('status', 'measured') not in ('measured', 'completed', 'complete'):
        raise ValueError('Only measured expression trials can be evaluated')
    expression = parse_expression(trial['expression'])
    if expression['order'] != ['wave', 'high5']:
        raise ValueError('Only executed Wave to High5 source order supported')
    identity = _identity(trial['source_identity'])
    metrics = trial.get('physical_metrics', trial['metrics'])
    if type(metrics) is not dict or set(metrics) != _METRIC_KEYS:
        raise ValueError('Original physical metrics required, not evaluated aliases')
    if (type(metrics['physical_gate_passed']) is not bool or
            type(metrics['actual_physics_steps']) is not int or metrics['actual_physics_steps'] != 8000 or
            type(metrics['online_onnx_call_count']) is not int or metrics['online_onnx_call_count'] != 400):
        raise ValueError('Physical gate and actual trial counters required')
    height = _number(metrics['min_root_height_m'], nonnegative=False)
    rmse = _number(metrics['mean_joint_rmse_rad'])
    if metrics['physical_gate_passed'] != (height >= .6 and rmse <= .35):
        raise ValueError('Physical flag contradicts unchanged numeric screen')
    if type(metrics['arm_rms_rad']) is not list or len(metrics['arm_rms_rad']) != 4:
        raise ValueError('Four visible original own-q0 diagnostics required')
    for v in metrics['arm_rms_rad']:
        _number(v)  # Visible integrity data only, never the new goal score.
    rows = trial['diagnostics']
    if type(rows) is not list or len(rows) != 4:
        raise ValueError('Exactly four measured diagnostic rows required')
    components = None
    for row, (phase, side) in zip(rows, ((0, 'left'), (0, 'right'), (1, 'left'), (1, 'right'))):
        if type(row) is not dict or set(row) != _ROW_KEYS:
            raise ValueError('Exact physical component diagnostic fields required')
        expected_window = [1.2, 3.2] if phase == 0 else [5.2, 7.2]
        if (type(row['phase']) is not int or row['phase'] != phase or
                type(row['side']) is not str or row['side'] != side or
                type(row['sample_count']) is not int or row['sample_count'] != 101 or
                type(row['window_s']) is not list or len(row['window_s']) != 2 or
                any(type(v) not in (int, float) for v in row['window_s']) or
                row['window_s'] != expected_window):
            raise ValueError('Fixed four phase/arm windows and 101 samples required')
        mean = _number(row['mean_pose_offset_rms_rad'])
        temporal = _number(row['centered_temporal_rms_rad'])
        peak = row['per_joint_peak_to_peak_rad']
        if type(peak) is not list or len(peak) != 7:
            raise ValueError('Seven joint peak-to-peak diagnostics required')
        for v in peak:
            _number(v)
        if phase == 1 and side == 'left':
            components = [mean, temporal]
    return expression, identity, components, metrics['physical_gate_passed']


def evaluate_goals(trial, baseline, goals):
    """Evaluate two fixed High5-left ratios after caller's runtime verification.

    Near-zero unit denominators are unassessable; zero candidate measurements
    are valid observations. Original own-q0 metrics never substitute components.
    """
    targets = parse_goals(goals)
    p, identity, actual, gate = _trial(trial)
    b, baseline_identity, denominator, baseline_gate = _trial(baseline)
    if baseline['index'] != -1 or any(b[k] != 1 for k in GAIN_FIELDS):
        raise ValueError('Fresh unit expression baseline required')
    if p['order'] != b['order'] or p['heading_policy'] != b['heading_policy'] or identity != baseline_identity:
        raise ValueError('Candidate and unit baseline source/heading must match')
    if trial['index'] == -1:
        if trial != baseline:
            raise ValueError('Baseline index cannot alias a different record')
    elif trial['id'] == baseline['id']:
        raise ValueError('Candidate cannot alias baseline identity')
    result = {
        'schema': 'pulsemorph.expression_goal_evaluation.v1',
        'goals': targets, 'targets': [targets['mean_pose_ratio'], targets['temporal_ratio']],
        'measurement': {'phase': 1, 'side': 'left', 'window_s': [5.2, 7.2], 'sample_count': 101,
                        'mean_pose_metric': 'mean_pose_offset_rms_rad',
                        'temporal_metric': 'centered_temporal_rms_rad',
                        'denominator_floor': DENOMINATOR_FLOOR},
        'tolerances': [TOLERANCE, TOLERANCE],
        'actual_components_rad': list(actual), 'baseline_components_rad': list(denominator),
        'physical_gate_passed': gate and baseline_gate, 'integrity_passed': True,
        'goal_count': 2, 'goal_total': 2, 'goals_passed': 0,
        'assessable': False, 'features': None, 'actual_ratios': None,
        'normalized_errors': None, 'worst_error': None, 'utility_l2': None,
        'assessment': 'unassessable', 'intent_passed': False,
        'qualification': 'Two measured High5-left component ratios only; verified source identity is a caller prerequisite, not an attestation. No own-q0 substitution or execution authority.'}
    if any(v < DENOMINATOR_FLOOR for v in denominator):
        result['reason'] = 'baseline_component_below_floor'
        return result
    features = [a / d for a, d in zip(actual, denominator)]
    errors = [abs(a - t) / TOLERANCE for a, t in zip(features, result['targets'])]
    utility = math.hypot(*errors)
    if not all(math.isfinite(v) for v in [*features, *errors, utility]):
        result['reason'] = 'nonfinite_ratio_evaluation'
        return result
    passed = sum(v <= 1 for v in errors)
    success = result['physical_gate_passed'] and passed == 2
    result.update(assessable=True, features=features,
                  actual_ratios={'mean_pose_ratio': features[0], 'temporal_ratio': features[1]},
                  normalized_errors=errors, worst_error=max(errors), utility_l2=utility,
                  goals_passed=passed, intent_passed=success,
                  assessment='target_met' if success else ('partial' if result['physical_gate_passed'] else 'physical_gate_failed'))
    return result


def select_final(trials, baseline, goals):
    """Return owned trial/evaluation; measured baseline is always eligible.

    Compiler rejects must have status rejected and metrics None. Invalid measured
    evidence raises rather than disappearing from the selection population.
    """
    if type(trials) is not list:
        raise ValueError('Ordered candidate list required')
    baseline_evaluation = evaluate_goals(baseline, baseline, goals)
    eligible = [(baseline, baseline_evaluation)]
    ids, indices = {baseline['id']}, {-1}
    for trial in trials:
        if type(trial) is not dict:
            raise ValueError('Candidate record required')
        if (type(trial.get('id')) is not str or not trial['id'] or type(trial.get('index')) is not int or
                trial['index'] < 0 or trial['id'] in ids or trial['index'] in indices):
            raise ValueError('Unique candidate IDs and nonnegative indices required')
        ids.add(trial['id']); indices.add(trial['index'])
        if trial.get('status') == 'rejected':
            if trial.get('metrics', 'missing') is not None or trial.get('physical_metrics') is not None:
                raise ValueError('Compiler rejection must not contain measured metrics')
            continue
        eligible.append((trial, evaluate_goals(trial, baseline, goals)))
    def key(item):
        trial, evaluation = item
        return (not evaluation['physical_gate_passed'],
                evaluation['worst_error'] if evaluation['assessable'] else math.inf,
                evaluation['utility_l2'] if evaluation['assessable'] else math.inf,
                trial['index'])
    trial, evaluation = min(eligible, key=key)
    return {'trial': copy.deepcopy(trial), 'evaluation': copy.deepcopy(evaluation)}
