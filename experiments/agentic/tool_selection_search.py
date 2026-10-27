"""Development-only eight-slot mixed-tool search; no execution or old labels.

Exact zero-based schedule: global task seed, local task seed, global maximin,
local maximin, model selection, maximin exploration, model selection, maximin
exploration. Seeds multiply identity gains by target/calibration feature ratios
and set shift to observed minus requested peak, clipped to the domain. This is
a directional heuristic, not an inferred nonzero-clock response. Both tools
share ONLY the measured zero-shift identity calibration (one provenance/evidence
item); models otherwise use exclusively their own valid measured history.

Pools: 1024 unscrambled Halton points (skip origin) per tool, plus 256 points
within normalized radius .25 of that tool's best measured point. Per-tool
linear-kernel degree-zero RBF, smoothing 1e-5, predicts tolerance-normalized
residuals; at most five points per tool under this eight-slot schedule. Model
slots first prioritize tools with fewer committed attempts, then minimize
max(abs(residual)) - .15 * nearest-distance, squared residual, tool order,
pool order. Exploration prioritizes fewer attempts then largest distance.
Missing model support falls back to maximin for all eligible tools. Failed
proposals consume slots; no fake features, retry, cross-tool labels, or physics.
Held-out sources are not yet available: algorithm development until freeze.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc

PARAMETER_NAMES = ['left_gain_a', 'right_gain_a', 'left_gain_b', 'right_gain_b', 'timing_shift_s']
TOOLS = ('global_clock_v1', 'local_clock_1p4_2p8_v1')
BOUNDS = np.array([[.35, 1.6]] * 4 + [[-.3, .3]])
IDENTITY = np.array([1., 1., 1., 1., 0.])
ALGORITHM_VERSION = 'mixed_tool_search_development_v1'


def _vector(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all():
        raise ValueError('expected five finite numbers')
    return a


def _parameters(row):
    try:
        a = _vector(row.get('parameters'))
        return a if np.all(a >= BOUNDS[:, 0]) and np.all(a <= BOUNDS[:, 1]) else None
    except (ValueError, TypeError):
        return None


def _observation(row):
    e = row.get('evaluation', row)
    if not isinstance(e, dict) or e.get('integrity_passed') is not True or e.get('physical_gate_passed') is not True:
        return None
    if (e.get('control_tool') is not None and row.get('control_tool') is not None
            and e['control_tool'] != row['control_tool']):
        return None
    try:
        return _vector(e.get('features'))
    except (ValueError, TypeError):
        return None


def next_candidate(history, protocol, goal, calibration):
    """Return control_tool, parameters, and auditable decision metadata.

    Protocol uses parameter_names, bounds, feature_tolerances; optional
    new_candidate_budget must equal 8. Calibration needs identity parameters,
    nonempty provenance and affirmative evaluation gates/features. History is
    committed slots, including malformed/failed proposals. Unknown-tool rows
    cannot train models. Duplicate (tool, parameters) rows are rejected, while
    identical coordinates across different tools are legitimate experiments.
    """
    target, tol = _vector(goal['feature_targets']), _vector(protocol['feature_tolerances'])
    if (protocol['parameter_names'] != PARAMETER_NAMES
            or not np.array_equal(np.asarray(protocol['bounds'], dtype=float), BOUNDS)
            or np.any(tol <= 0) or protocol.get('new_candidate_budget', 8) != 8):
        raise ValueError('invalid mixed-tool protocol')
    n = len(history)
    if n >= 8:
        raise ValueError('eight candidate budget exhausted')
    cal_x, cal_y = _parameters(calibration), _observation(calibration)
    if (not calibration.get('provenance') or cal_x is None
            or not np.array_equal(cal_x, IDENTITY) or cal_y is None):
        raise ValueError('valid provenance-bearing identity calibration required')
    lo, span = BOUNDS[:, 0], BOUNDS[:, 1] - BOUNDS[:, 0]
    anchor = (IDENTITY - lo) / span
    data, tried, counts = {}, {}, {}
    for tool in TOOLS:
        rows = [r for r in history if r.get('control_tool') == tool]
        counts[tool] = len(rows)
        pars = [p for r in rows if (p := _parameters(r)) is not None]
        if len({tuple(np.round(p, 12)) for p in pars}) != len(pars):
            raise ValueError('duplicate tool and parameters')
        tried[tool] = np.array([(p - lo) / span for p in pars]).reshape(-1, 5)
        # Calibration is shared evidence, never two independent observations.
        xs, ys = [anchor], [(cal_y - target) / tol]
        for row in rows:
            p, y = _parameters(row), _observation(row)
            if p is not None and y is not None and not np.allclose(p, IDENTITY, atol=1e-12, rtol=0):
                xs.append((p - lo) / span)
                ys.append((y - target) / tol)
        data[tool] = (np.array(xs), np.array(ys))
    halton = qmc.Halton(5, scramble=False).random(1025)[1:]
    candidates = []
    for ti, tool in enumerate(TOOLS):
        x, y = data[tool]
        best = min(range(len(x)), key=lambda i: (float(abs(y[i]).max()), float(y[i] @ y[i]), i))
        pool = np.vstack((halton, np.clip(x[best] + (halton[:256] * 2 - 1) * .25, 0, 1)))
        if n < 2:
            seed = IDENTITY.copy()
            positive = cal_y[:4] > 1e-10
            seed[:4][positive] *= target[:4][positive] / cal_y[:4][positive]
            seed[4] = cal_y[4] - target[4]
            pool = np.vstack(((np.clip(seed, lo, lo + span) - lo) / span, pool))
        distance = np.linalg.norm(pool[:, None] - np.vstack((anchor, tried[tool]))[None, :], axis=2).min(axis=1)
        fresh = distance > 1e-9
        # Identity is the free shared calibration and is never re-proposed.
        indices = np.flatnonzero(fresh)
        pred = RBFInterpolator(x, y, kernel='linear', degree=0, smoothing=1e-5)(pool) if len(x) >= 2 else None
        candidates.append((tool, pool, distance, indices, pred))
    strategy = 'task_informed_seed' if n < 2 else 'maximin_probe' if n < 4 else 'model_selection' if n % 2 == 0 else 'maximin_exploration'
    eligible = [n % 2] if n < 4 else [i for i, t in enumerate(TOOLS) if counts[t] == min(counts.values())]
    model_mode = strategy == 'model_selection' and all(candidates[i][4] is not None for i in eligible)
    if strategy == 'model_selection' and not model_mode:
        strategy = 'insufficient_data_maximin'
    choices = []
    for ti in eligible:
        tool, pool, distance, indices, pred = candidates[ti]
        for j in indices:
            if n < 2:
                key = (int(j), ti)
            elif model_mode:
                key = (float(abs(pred[j]).max() - .15 * distance[j]), float(pred[j] @ pred[j]), ti, int(j))
            else:
                key = (-float(distance[j]), ti, int(j))
            choices.append((key, ti, int(j)))
    if not choices:
        raise ValueError('candidate pool exhausted')
    _, ti, j = min(choices)
    tool, pool, distance, indices, pred = candidates[ti]
    return dict(control_tool=tool, parameters=np.clip(lo + span * pool[j], lo, lo + span).tolist(),
        decision=dict(algorithm_version=ALGORITHM_VERSION, new_candidate_index=n, strategy=strategy,
            prediction_is_physical_evidence=False, development_only=True,
            calibration_evidence_count=1, calibration_provenance=calibration['provenance'],
            model_points_by_tool={t: len(data[t][0]) for t in TOOLS},
            committed_attempts_by_tool=counts,
            predicted_normalized_residual=pred[j].tolist() if model_mode else None))
