"""Frozen bounded witness search; predictions are never physical evidence.

Only measured high5_b warmstarts are admitted by the caller with provenance.
An unsuccessful finite search is not an impossibility certificate. Timing is
fit as measured (including quantization/peak switches), never smoothed in logs.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.stats import qmc

PARAMETER_NAMES = ['left_gain_a', 'right_gain_a', 'left_gain_b', 'right_gain_b', 'timing_shift_s']
ALGORITHM_VERSION = 'capability_witness_v1'
SCHEDULE = ('local_minmax', 'coordinate_probe', 'local_minmax', 'global_rbf',
            'local_minmax', 'coordinate_probe', 'global_rbf', 'halton_exploration')


def _vector(value, label):
    a = np.asarray(value, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all():
        raise ValueError('invalid ' + label)
    return a


def _observation(row):
    e = row.get('evaluation', row)
    if e.get('integrity_passed') is not True or e.get('physical_gate_passed') is not True:
        return None
    try:
        return _vector(e.get('features'), 'features')
    except (ValueError, TypeError):
        return None


def next_candidate(prior, new_history, protocol, goal):
    """Return ``{parameters, decision}`` without executing or assuming physics.

    Rows follow generalization_search: parameters and evaluation with features,
    integrity_passed and physical_gate_passed. Every prior row additionally needs
    nonempty provenance. Failed fresh trials consume budget and exclude repeats.
    Verification repeats are intentionally outside this proposal interface.
    All coordinates are continuous tool inputs; execution retains existing
    section boundaries, warp semantics and physics time quantization.
    """
    bounds = np.asarray(protocol['bounds'], dtype=float)
    target = _vector(goal['feature_targets'], 'target')
    tol = _vector(protocol['feature_tolerances'], 'tolerances')
    budget = protocol.get('new_candidate_budget', 64)
    if (protocol['parameter_names'] != PARAMETER_NAMES or bounds.shape != (5, 2)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(tol <= 0) or not isinstance(budget, int) or not 1 <= budget <= 64):
        raise ValueError('invalid capability protocol')
    if len(new_history) >= budget:
        raise ValueError('new candidate budget exhausted')
    if any(not r.get('provenance') for r in prior):
        raise ValueError('warmstart provenance required')
    rows = list(prior) + list(new_history)
    lo, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    raw = np.array([_vector(r['parameters'], 'parameters') for r in rows]).reshape(-1, 5)
    if np.any(raw < lo) or np.any(raw > lo + span):
        raise ValueError('history outside bounds')
    x = (raw - lo) / span
    # Duplicate fresh measurements cannot purchase additional search slots.
    if len(new_history) and len(np.unique(np.round(raw[len(prior):], 12), axis=0)) != len(new_history):
        raise ValueError('duplicate fresh candidate')
    obs = [_observation(r) for r in rows]
    valid = [i for i, y in enumerate(obs) if y is not None]
    n = len(new_history)
    strategy = SCHEDULE[n % len(SCHEDULE)]
    radius = (.20, .12, .07, .04)[min(n // 16, 3)]
    halton = qmc.Halton(5, scramble=False).random(2049)[1:]
    pool = halton.copy()
    prediction = None
    center_index = None
    model_points = 0
    if valid:
        y = np.array([(obs[i] - target) / tol for i in valid])
        best = min(range(len(valid)), key=lambda j: (float(abs(y[j]).max()), float(y[j] @ y[j]), valid[j]))
        center_index = valid[best]
        center = x[center_index]
        local = np.clip(center + (halton[:512] * 2 - 1) * radius, 0, 1)
        probes = np.clip(center + np.vstack((np.eye(5), -np.eye(5))) * radius, 0, 1)
        pool = np.vstack((local, halton))
        unique, inverse = np.unique(x[valid], axis=0, return_inverse=True)
        means = np.array([y[inverse == i].mean(axis=0) for i in range(len(unique))])
        model_points = len(unique)
        if strategy == 'coordinate_probe':
            # Rotate axes/signs deterministically, retaining both timing signs.
            pool = np.roll(probes, -(n // 8 * 2 + n % 8 // 4), axis=0)
        elif strategy == 'local_minmax' and len(unique) >= 6:
            distances = np.linalg.norm(unique - center, axis=1)
            near = np.argsort(distances, kind='stable')[:min(24, len(unique))]
            design = np.column_stack((np.ones(len(near)), unique[near] - center))
            weights = 1 / (1 + (distances[near] / radius) ** 2)
            penalty = np.diag([0.] + [1e-5] * 5)
            coef = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                                   design.T @ (weights[:, None] * means[near]))
            def predict(z):
                return coef[0] + (z - center) @ coef[1:]
            lower, upper = np.maximum(0, center - radius), np.minimum(1, center + radius)
            initial = np.r_[center, np.max(abs(predict(center)))]
            result = minimize(lambda z: z[5], initial, method='SLSQP',
                bounds=list(zip(lower, upper)) + [(0, None)],
                constraints=[{'type': 'ineq', 'fun': lambda z: z[5] - predict(z[:5])},
                             {'type': 'ineq', 'fun': lambda z: z[5] + predict(z[:5])}],
                options={'maxiter': 100, 'ftol': 1e-10})
            if result.success and np.isfinite(result.x).all():
                pool = np.vstack((result.x[:5], local))
            else:
                pool = local
            prediction = predict(pool)
        elif strategy != 'halton_exploration' and len(unique) >= 2:
            prediction = RBFInterpolator(unique, means, kernel='linear', degree=0, smoothing=1e-5)(pool)
    if strategy == 'coordinate_probe' and not valid:
        strategy = 'halton_exploration'
    distance = np.linalg.norm(pool[:, None] - x[None, :], axis=2).min(axis=1) if len(x) else np.ones(len(pool))
    fresh = distance > 1e-9
    if not fresh.any():
        pool = halton
        distance = np.linalg.norm(pool[:, None] - x[None, :], axis=2).min(axis=1)
        fresh = distance > 1e-9
        prediction = None
        strategy = 'halton_fallback'
    indices = np.flatnonzero(fresh)
    if prediction is not None:
        chosen = min(indices, key=lambda i: (float(abs(prediction[i]).max()), float(prediction[i] @ prediction[i]), -float(distance[i]), int(i)))
    elif strategy == 'coordinate_probe':
        chosen = int(indices[0])
    else:
        chosen = int(indices[np.argmax(distance[indices])])
    return dict(parameters=np.clip(lo + span * pool[chosen], lo, lo + span).tolist(),
        decision=dict(algorithm_version=ALGORITHM_VERSION, new_candidate_index=n,
            strategy=strategy, trust_radius_normalized=radius, warmstart_rows=len(prior),
            fresh_rows=n, valid_measured_rows=len(valid), unique_model_points=model_points,
            center_history_index=center_index, predicted_normalized_residual=None if prediction is None else prediction[chosen].tolist(),
            prediction_is_physical_evidence=False))
