"""Warm-start development comparator: local affine transfer and minimax repair.

Nominal derivatives and incumbent state offsets are surrogate assumptions, not
physical evidence. Only shared pinned priors and this arm's measured history
enter fits. No simulator, provider, or filesystem access occurs here.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog, minimize

PARAMETER_NAMES = ['left_gain_a', 'right_gain_a', 'left_gain_b', 'right_gain_b', 'timing_shift_s']
TOOLS = ('global_clock_v1', 'local_clock_1p4_2p8_v1')
BOUNDS = np.array([[.35, 1.6]] * 4 + [[-.3, .3]])
ALGORITHM_VERSION = 'repair_weighted_affine_minimax_v1'


def _vector(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all():
        raise ValueError('expected five finite numbers')
    return a


def _parameters(row):
    a = _vector(row['parameters'])
    if np.any(a < BOUNDS[:, 0]) or np.any(a > BOUNDS[:, 1]):
        raise ValueError('parameters outside bounds')
    return a


def _states(row):
    result = {}
    for e in row.get('state_evaluations', []):
        sid = e['scenario_id']
        if sid in result:
            raise ValueError('duplicate scenario observation')
        result[sid] = None
        if e.get('integrity_passed') is True and e.get('physical_gate_passed') is True:
            try:
                # np.asarray(None) can produce nan: never invent a missing peak.
                result[sid] = _vector(e.get('features'))
            except (TypeError, ValueError):
                pass
    return result


def _score(residual):
    return float(np.max(np.abs(residual))), float(np.sum(residual ** 2))


def _fit_rows(rows, tool, scenario_id):
    """Count repeated labels once without discarding their input provenance.

    Exact parameter/state/feature matches are redundant fit evidence. Different
    measured outcomes at identical coordinates remain separate observations.
    """
    seen, unique = set(), []
    for row in rows:
        y = row['states'].get(scenario_id)
        if row['tool'] != tool or y is None:
            continue
        key = (tool, tuple(row['raw']['parameters']), scenario_id, tuple(y))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _solve(residual, jacobian, lower, upper):
    """Lexicographic surrogate minimax, then sumsq inside the optimal slab."""
    r, j = residual.ravel(), jacobian.reshape(-1, 5)
    lp = linprog(np.r_[np.zeros(5), 1.],
        A_ub=np.vstack((np.column_stack((j, -np.ones(len(r)))),
                       np.column_stack((-j, -np.ones(len(r)))))),
        b_ub=np.r_[-r, r], bounds=[*zip(lower, upper), (0, None)], method='highs')
    if not lp.success:
        raise ValueError('minimax LP failed: ' + lp.message)
    ceiling = float(lp.x[5]) + 1e-9
    objective = lambda d: float(np.sum((r + j @ d) ** 2))
    tie = minimize(objective, lp.x[:5], jac=lambda d: 2 * j.T @ (r + j @ d),
        bounds=list(zip(lower, upper)), constraints=[dict(type='ineq',
            fun=lambda d: np.r_[ceiling - r - j @ d, ceiling + r + j @ d],
            jac=lambda d: np.vstack((-j, j)))], method='SLSQP',
        options=dict(ftol=1e-11, maxiter=200))
    d = tie.x if tie.success and np.max(np.abs(r + j @ tie.x)) <= ceiling + 1e-8 else lp.x[:5]
    return np.clip(d, lower, upper), bool(tie.success)


def next_candidate(history, protocol, goal, prior_observations):
    """Propose one of eight candidates; all inputs/outputs are JSON compatible.

    Prior/history rows contain control_tool, five parameters, state_evaluations
    and nonempty provenance. History rows may additionally carry the returned
    search_diagnostics for transparent trust-radius adaptation. Nominal priors
    estimate tool-specific derivatives; the robust incumbent supplies offsets
    across states, including to the alternate tool until measured there.
    """
    target, tol = _vector(goal['feature_targets']), _vector(protocol['feature_tolerances'])
    conditions = protocol['development_condition_ids']
    if (protocol['parameter_names'] != PARAMETER_NAMES
            or not np.array_equal(np.asarray(protocol['bounds'], dtype=float), BOUNDS)
            or np.any(tol <= 0) or protocol.get('new_candidate_budget', 8) != 8
            or not isinstance(conditions, list) or len(conditions) != 4
            or len(set(conditions)) != 4 or 'nominal' not in conditions):
        raise ValueError('invalid repair protocol')
    if len(history) >= 8:
        raise ValueError('eight candidate budget exhausted')
    lo, span = BOUNDS[:, 0], BOUNDS[:, 1] - BOUNDS[:, 0]
    rows = []
    for source, records in [('prior', prior_observations), ('history', history)]:
        for index, row in enumerate(records):
            if row.get('control_tool') not in TOOLS or not row.get('provenance'):
                raise ValueError('known tool and provenance required')
            rows.append(dict(x=(_parameters(row)-lo)/span, states=_states(row),
                             tool=row['control_tool'], source=source, index=index, raw=row))
    if not prior_observations:
        raise ValueError('shared measured prior required')
    robust = [r for r in rows if all(r['states'].get(s) is not None for s in conditions)]
    if not robust:
        raise ValueError('complete physical robust incumbent required')
    def residual(row):
        return np.array([(row['states'][s] - target) / tol for s in conditions])
    incumbent = min(robust, key=lambda r: (*_score(residual(r)), r['source'] != 'prior', r['index']))
    offsets = {s: incumbent['states'][s] - incumbent['states']['nominal'] for s in conditions}
    # Failed measured attempts consume budget and remain excluded coordinates.
    tried = {t: [r['x'] for r in rows if r['tool'] == t] for t in TOOLS}
    history_pairs = [(r['tool'], tuple(r['x'])) for r in rows if r['source'] == 'history']
    if len(set(history_pairs)) != len(history_pairs):
        raise ValueError('duplicate history tool and parameters')
    bad_predictions = 0
    for r in robust:
        diagnostic = r['raw'].get('search_diagnostics', {})
        predicted = diagnostic.get('predicted_worst_normalized_error')
        center = diagnostic.get('center_worst_normalized_error')
        if r['source'] == 'history' and isinstance(predicted, (int, float)) and isinstance(center, (int, float)):
            actual = _score(residual(r))[0]
            if actual > predicted + .5 * max(0., center - predicted) + .05:
                bad_predictions += 1
    radius = max(.025, .2 * (.5 ** bad_predictions))
    choices = []
    diagnostics_by_tool = {}
    for ti, tool in enumerate(TOOLS):
        nominal = _fit_rows(rows, tool, 'nominal')
        if not nominal:
            raise ValueError('nominal measured prior required for each tool')
        complete = [r for r in robust if r['tool'] == tool]
        if complete:
            center = min(complete, key=lambda r: (*_score(residual(r)), r['source'] != 'prior', r['index']))
            base = np.array([center['states'][s] for s in conditions])
        else:
            center = min(nominal, key=lambda r: (*_score(np.array([(r['states']['nominal'] + offsets[s] - target)/tol for s in conditions])), r['index']))
            base = np.array([center['states']['nominal'] + offsets[s] for s in conditions])
        x0 = center['x']
        # Affine slopes use differences from a measured nominal center. Nearby
        # samples dominate; ridge makes rank-deficient fits explicit and finite.
        dx = np.array([r['x'] - x0 for r in nominal])
        dy = np.array([(r['states']['nominal'] - center['states']['nominal']) / tol for r in nominal])
        weights = np.exp(-np.sum(dx ** 2, axis=1) / .5 ** 2)
        normal = dx.T @ (weights[:, None] * dx) + np.eye(5) * 1e-4
        nominal_j = np.linalg.solve(normal, dx.T @ (weights[:, None] * dy)).T
        js = []
        for si, sid in enumerate(conditions):
            # Own robust feedback estimates a correction to the shared nominal
            # derivative without importing sibling methods' new outcomes.
            same = _fit_rows(rows, tool, sid)
            sx = np.array([r['x']-x0 for r in same]).reshape(-1, 5)
            sy = np.array([(r['states'][sid]-base[si])/tol for r in same]).reshape(-1, 5)
            sw = np.exp(-np.sum(sx ** 2, axis=1)/.5 ** 2)
            correction = np.linalg.solve(sx.T @ (sw[:, None]*sx) + np.eye(5)*1e-3,
                sx.T @ (sw[:, None]*(sy-sx @ nominal_j.T))).T
            js.append(nominal_j + correction)
        jac = np.array(js)
        r0 = (base-target)/tol
        lower, upper = np.maximum(-radius, -x0), np.minimum(radius, 1-x0)
        delta, tie_ok = _solve(r0, jac, lower, upper)
        pool = [x0+delta]
        # Deterministic bounded coordinate polls prevent re-proposing a prior or
        # failed point when the surrogate predicts the center is already best.
        for scale in (1., .5, .25):
            for axis in range(5):
                for sign in (1., -1.):
                    d = np.zeros(5); d[axis] = sign*radius*scale
                    pool.append(np.clip(x0+d, 0, 1))
        fresh = [(i, x) for i, x in enumerate(pool) if all(np.linalg.norm(x-old) > 1e-9 for old in tried[tool])]
        info = dict(nominal_points=len(nominal), complete_robust_points=len(complete),
                    nominal_design_rank=int(np.linalg.matrix_rank(dx)), tie_solver_succeeded=tie_ok)
        diagnostics_by_tool[tool] = info
        for pi, x in fresh:
            pred = r0 + np.einsum('sfp,p->sf', jac, x-x0)
            score = _score(pred)
            choices.append(((*score, ti, pi), tool, x, score, _score(r0), pi))
    if not choices:
        raise ValueError('candidate pool exhausted')
    _, tool, x, predicted, center_score, pi = min(choices, key=lambda v: v[0])
    return dict(control_tool=tool, parameters=np.clip(lo+span*x, lo, lo+span).tolist(),
        search_diagnostics=dict(algorithm_version=ALGORITHM_VERSION, new_candidate_index=len(history),
            strategy='affine_minimax' if pi == 0 else 'coordinate_poll',
            prediction_is_physical_evidence=False, warm_start_labeled_development=True,
            state_offset_transfer_assumption=True, nominal_derivative_transfer_assumption=True,
            trust_radius_normalized=radius, bad_prediction_count=bad_predictions,
            predicted_worst_normalized_error=predicted[0], predicted_normalized_sumsq=predicted[1],
            center_worst_normalized_error=center_score[0], models_by_tool=diagnostics_by_tool,
            shared_prior_observation_count=len(prior_observations)))
