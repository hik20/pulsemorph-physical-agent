"""Frozen nominal-target and matched-condition diagnostics; no simulation."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

_spec = importlib.util.spec_from_file_location('_robustness_generalization', Path(__file__).with_name('evaluate_generalization.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_require = _base._require


def _same_state(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def score_features(features, target, tolerance):
    """Pure scoring with explicit unavailable accent/ratio values."""
    errors = [None if value is None else float(value - wanted) for value, wanted in zip(features, target)]
    passed = [error is not None and abs(error) <= tol for error, tol in zip(errors, tolerance)]
    norms = None if any(error is None for error in errors) else np.abs(np.asarray(errors) / tolerance)
    return dict(features=features, feature_errors=errors, per_dimension_pass=passed,
                intent_passed=bool(all(passed)), feature_tolerances=tolerance.tolist(),
                utility=None if norms is None else float(np.max(norms)),
                utility_l2=None if norms is None else float(np.sum(norms**2)))


def _physical(audit, protocol):
    height = float(np.min(audit['sub'][:, 2]))
    rmse = float(audit['mean_joint_rmse_rad'])
    gate = protocol['physical_gate']
    return dict(min_root_height_m=height, mean_joint_rmse_rad=rmse,
                physical_gate_passed=bool(height >= gate['min_root_height_m'] and rmse <= gate['max_mean_joint_rmse_rad']))


def _initial(audit):
    manifest = audit['manifest']
    _require(manifest['schema'] == 'pulsemorph.agentic_generalization_trial.v1', 'generalization trial schema required')
    _require(manifest.get('reference_initial_posture_is_physics_state') is False, 'reference/physical initial state distinction required')
    record = manifest['initial_perturbation']
    _require(record['applied_once_at_time_s'] == 0 and record['reference_rotated'] is False, 'invalid startup perturbation application')
    _require(record['quaternion_order'] == 'wxyz' and record['root_linear_velocity_frame'] == 'world'
             and record['root_yaw_frame'] == 'world_left_multiply', 'initial perturbation frame mismatch')
    for name, shape in [('before_qpos', (36,)), ('after_qpos', (36,)), ('before_qvel', (35,)), ('after_qvel', (35,))]:
        values = np.asarray(record[name], dtype=float)
        _require(values.shape == shape and np.isfinite(values).all(), 'invalid '+name)
    for key in ('qpos', 'qvel'):
        _require(np.array_equal(record['after_'+key], audit['arrays'][key][0]), 'recorded actual initial '+key+' mismatch')
    return record


def evaluate_robustness(directory, nominal_baseline, condition_baseline, protocol, goal):
    """Audit all three artifacts, then score fixed and condition-relative targets.

    Baseline physical gate failures remain explicit data; the driver decides
    whether a failed nominal calibration prevents panel interpretation.
    """
    result = dict(schema='pulsemorph.agentic_robustness_evaluation.v1', integrity_passed=False,
                  integrity_errors=[], physical_gate_passed=False, primary=None, secondary=None,
                  limitation='Deterministic initial-state sensitivity of fixed outputs; not recovery, probability or aesthetics.')
    try:
        target, tolerance, _ = _base._configuration(protocol, goal)
        trial, nominal, condition = [_base._audit(path) for path in (directory, nominal_baseline, condition_baseline)]
        for key in ('onnx_sha256', 'motion_sha256', 'physics_contract', 'versions', 'preexecution_input_hashes'):
            _require(trial['manifest'][key] == nominal['manifest'][key] == condition['manifest'][key], 'source/controller pin mismatch: '+key)
        _require(trial['names'] == nominal['names'] == condition['names'], 'joint profile mismatch')
        initial = [_initial(audit) for audit in (trial, nominal, condition)]
        q, b0, bp = [audit['arrays']['qpos'] for audit in (trial, nominal, condition)]
        times = trial['arrays']['state_times']
        for audit in (nominal, condition):
            _require(np.array_equal(times, audit['arrays']['state_times']), 'baseline state time mismatch')
        for key in ('qpos', 'qvel'):
            _require(_same_state(trial['arrays'][key][0], condition['arrays'][key][0]), 'condition baseline initial '+key+' mismatch')
            _require(all(np.array_equal(row['before_'+key], nominal['arrays'][key][0]) for row in initial), 'nominal before-perturbation '+key+' mismatch')
        _require(initial[0]['spec'] == initial[2]['spec'], 'candidate/condition perturbation spec mismatch')
        groups = []
        for side in ('left', 'right'):
            group = [i for i, name in enumerate(trial['names']) if name.startswith(side+'_') and any(part in name for part in ('shoulder_', 'elbow_', 'wrist_'))]
            _require(len(group) == 7, 'expected seven '+side+' arm joints')
            groups.append(group)
        _require(set(groups[0]).isdisjoint(groups[1]), 'arm groups overlap')
        primary, secondary, dimensions = [], [], []
        windows = protocol['feature_windows_s']
        _require(len(windows) == 2, 'two feature windows required')
        for label, (start, end) in zip(('a', 'b'), windows):
            _require(0 <= start < end <= 4, 'invalid feature window')
            mask = (times >= start-1e-10) & (times <= end+1e-10)
            _require(mask.any(), 'empty feature window')
            for side, group in zip(('left', 'right'), groups):
                c0, cp = b0[0, 7:][group], bp[0, 7:][group]
                def rms(values, center):
                    return float(np.sqrt(np.mean((values[mask, 7:][:, group]-center)**2)))
                C0, B0, Bcondition0, Cp, Bp = rms(q,c0), rms(b0,c0), rms(bp,c0), rms(q,cp), rms(bp,cp)
                _require(B0 > 1e-10, 'zero nominal baseline amplitude')
                primary.append(C0/B0)
                secondary.append(Cp/Bp if Bp > 1e-10 else None)
                index = len(primary)-1
                dimensions.append(dict(name=side+'_ratio_'+label, candidate_rms_about_nominal_rad=C0,
                    nominal_baseline_rms_rad=B0, condition_baseline_rms_about_nominal_rad=Bcondition0,
                    candidate_rms_about_condition_rad=Cp, condition_baseline_rms_rad=Bp,
                    condition_center_minus_nominal_rad=(cp-c0).tolist(), joint_names=[trial['names'][i] for i in group],
                    primary_absolute_rms_target_rad=float(target[index]*B0),
                    primary_absolute_rms_tolerance_rad=float(tolerance[index]*B0),
                    secondary_available=Bp > 1e-10,
                    secondary_unavailable_reason=None if Bp > 1e-10 else 'zero condition baseline amplitude'))
        window = protocol['accent_window_s']
        _require(len(window) == 2 and 0 <= window[0] < window[1] <= 4, 'invalid accent window')
        prominence = float(protocol.get('accent_minimum_prominence', .15))
        _require(np.isfinite(prominence) and prominence > 0, 'positive accent prominence required')
        accent = _base.accent_metrics(trial['arrays']['qvel'][:,6:], times, window, prominence)
        primary.append(accent['peak_time_s']); secondary.append(accent['peak_time_s'])
        physical = _physical(trial, protocol)
        primary_score, secondary_score = score_features(primary,target,tolerance), score_features(secondary,target,tolerance)
        for score in (primary_score, secondary_score):
            score['joint_success'] = bool(physical['physical_gate_passed'] and score['intent_passed'])
        result.update(integrity_passed=True, **physical, primary=primary_score, secondary=secondary_score,
                      dimensions=dimensions, accent=accent,
                      initial=dict(candidate=initial[0], nominal_baseline=initial[1], condition_baseline=initial[2],
                                   primary_center='nominal baseline actual initial joints', secondary_center='condition baseline actual initial joints'),
                      nominal_baseline_physical=_physical(nominal,protocol), condition_baseline_physical=_physical(condition,protocol),
                      quality_diagnostics=_base.quality_diagnostics(trial,directory),
                      fresh_anchor_audits=[audit.get('fresh_anchor_audit') for audit in (trial,nominal,condition)])
    except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError, ET.ParseError, zipfile.BadZipFile) as error:
        result.update(integrity_passed=False, physical_gate_passed=False, primary=None, secondary=None)
        result['integrity_errors'].append(str(error))
    return result
