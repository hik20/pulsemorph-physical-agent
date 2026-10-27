"""Deterministic bounded, measured-feedback comparator for five motion controls.

The response surrogate is a numerical interpolation, not a physical model.
Peak identity can switch discontinuously; sparse RBF predictions carry no
convergence or quality guarantee. Every proposal must be executed and audited.
No budget, prior task outcomes, or unmeasured objective values enter this search.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc

PARAMETER_NAMES = ['left_gain_a', 'right_gain_a', 'left_gain_b', 'right_gain_b', 'timing_shift_s']


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (5,) or not np.isfinite(result).all():
        raise ValueError('invalid ' + name)
    return result


def _observation(row):
    evaluation = row.get('evaluation', row)
    # Missing audit/gate status is not affirmative physical evidence.
    if evaluation.get('integrity_passed') is not True or evaluation.get('physical_gate_passed') is not True:
        return None
    try:
        return _vector(evaluation.get('features'), 'observed features')
    except (ValueError, TypeError):
        return None


def next_candidate(history, protocol, goal):
    """Return one new bounded vector using only actually measured history.

    Required protocol: parameter_names, bounds (5x2), feature_tolerances (5).
    Goal: feature_targets (5). Rows: parameters and an evaluation (or its fields
    at row level), including explicit integrity/physical passes and features.
    Selection uses worst absolute tolerance error, then squared normalized L2.
    """
    target = _vector(goal['feature_targets'], 'feature targets')
    tolerance = _vector(protocol['feature_tolerances'], 'feature tolerances')
    bounds = np.asarray(protocol['bounds'], dtype=float)
    if (protocol['parameter_names'] != PARAMETER_NAMES or bounds.shape != (5, 2)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(tolerance <= 0)):
        raise ValueError('invalid search configuration')
    lower, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    anchor = np.array([1., 1., 1., 1., 0.])
    if np.any(anchor < lower) or np.any(anchor > bounds[:, 1]):
        raise ValueError('forced anchor outside bounds')
    if not history:
        return anchor.tolist()
    measured = np.array([_vector(row['parameters'], 'history parameters') for row in history])
    if np.any(measured < lower) or np.any(measured > bounds[:, 1]):
        raise ValueError('history parameters outside bounds')
    x = (measured - lower) / span
    observations = [_observation(row) for row in history]
    valid = [i for i, value in enumerate(observations) if value is not None]
    # A fixed low discrepancy design is independent of task outcomes and budget.
    global_pool = qmc.Halton(5, scramble=False).random(1025)[1:]
    pool = global_pool
    if valid:
        y = np.array([observations[i] for i in valid])
        errors = np.abs((y - target) / tolerance)
        best = min(range(len(valid)), key=lambda i: (float(errors[i].max()), float(errors[i] @ errors[i]), valid[i]))
        center = x[valid[best]]
        # Trust region contracts with observations but retains global candidates.
        radius = max(.12, .45 / np.sqrt(max(1, len(valid) - 1)))
        local = np.clip(center + (global_pool - .5) * 2 * radius, 0, 1)
        pool = np.vstack((local, global_pool))
        if len(history) == 1:
            observed = y[0]
            proposal = measured[valid[0]].copy()
            positive = observed[:4] > 1e-10
            proposal[:4][positive] *= target[:4][positive] / observed[:4][positive]
            # Positive source-time shift advances an event in execution time.
            proposal[4] += observed[4] - target[4]
            pool = np.vstack(((np.clip(proposal, lower, lower + span) - lower) / span, pool))
    distance = np.linalg.norm(pool[:, None, :] - x[None, :, :], axis=2).min(axis=1)
    fresh = distance > 1e-9
    pool, distance = pool[fresh], distance[fresh]
    if not len(pool):
        raise ValueError('deterministic candidate pool exhausted')
    if valid and len(history) == 1:
        chosen = 0
    # Explicit maximin slots are zero-based proposal indices 4, 8, 12, ...;
    # the anchor occupies index 0. Other insufficient-data slots also explore.
    elif len(valid) >= 2 and len(history) % 4 != 0:
        # Smoothing and degree zero avoid an underdetermined 5D affine tail.
        # Duplicate observations are averaged, never replaced with fake feedback.
        unique, inverse = np.unique(x[valid], axis=0, return_inverse=True)
        means = np.array([y[inverse == i].mean(axis=0) for i in range(len(unique))])
        if len(unique) < 2:
            chosen = int(np.argmax(distance))
        else:
            prediction = RBFInterpolator(unique, means, kernel='linear', degree=0, smoothing=1e-6)(pool)
            error = np.abs((prediction - target) / tolerance)
            chosen = min(range(len(pool)), key=lambda i: (float(error[i].max()), float(error[i] @ error[i]), -float(distance[i]), i))
    else:
        chosen = int(np.argmax(distance))
    return np.clip(lower + span * pool[chosen], lower, lower + span).tolist()
