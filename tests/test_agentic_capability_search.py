"""Numerical search invariants and analytical witness, without simulator calls."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location('capability_search', Path(__file__).parents[1] / 'experiments/agentic/capability_search.py')
search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search)


def config():
    return dict(parameter_names=search.PARAMETER_NAMES.copy(), bounds=[[.35, 1.6]] * 4 + [[-.3, .3]],
                feature_tolerances=[.08] * 4 + [.1]), dict(feature_targets=[1.1, .75, .75, 1.1, 2.2948072562358277])


def response(p):
    return np.r_[p[:4], 2.1 - p[4]]


def row(p, measured=True):
    return dict(parameters=list(p), provenance={'source': 'synthetic analytical fixture'},
                evaluation=dict(features=response(p).tolist(), integrity_passed=measured, physical_gate_passed=measured))


def prior_design():
    center = np.array([1., 1., 1., 1., 0.])
    return [row(center)] + [row(center + direction * .1) for direction in np.vstack((np.eye(5), -np.eye(5)))]


def test_affine_minmax_improves_all_features_and_correct_timing_sign():
    protocol, goal = config()
    prior = prior_design()
    out = search.next_candidate(prior, [], protocol, goal)
    p = np.array(out['parameters'])
    tol = np.array(protocol['feature_tolerances'])
    old = min(np.max(abs((response(np.array(r['parameters'])) - goal['feature_targets']) / tol)) for r in prior)
    assert np.max(abs((response(p) - goal['feature_targets']) / tol)) < old
    assert p[4] < 0  # Later target needs a negative source-time shift.
    assert out['decision']['strategy'] == 'local_minmax'
    assert out['decision']['prediction_is_physical_evidence'] is False


def test_deterministic_bounded_no_repeat_and_no_mutation():
    protocol, goal = config()
    prior, history = prior_design(), []
    for _ in range(12):
        before = copy.deepcopy((prior, history, protocol, goal))
        out = search.next_candidate(prior, history, protocol, goal)
        assert out == search.next_candidate(prior, history, protocol, goal)
        assert (prior, history, protocol, goal) == before
        p = np.array(out['parameters'])
        assert np.all(p >= np.array(protocol['bounds'])[:, 0])
        assert np.all(p <= np.array(protocol['bounds'])[:, 1])
        assert all(np.linalg.norm(p - r['parameters']) > 1e-9 for r in prior + history)
        history.append(row(p))


@pytest.mark.parametrize('failure', ['integrity_passed', 'physical_gate_passed', 'features', 'missing_status'])
def test_missing_or_failed_physics_excluded_from_models_but_not_history(failure):
    protocol, goal = config()
    p = [1., 1., 1., 1., 0.]
    bad = row(p)
    if failure == 'features':
        bad['evaluation']['features'][-1] = None
    elif failure == 'missing_status':
        del bad['evaluation']['physical_gate_passed']
    else:
        bad['evaluation'][failure] = False
    out = search.next_candidate([], [bad], protocol, goal)
    assert out['decision']['valid_measured_rows'] == 0
    assert out['decision']['predicted_normalized_residual'] is None
    assert out['parameters'] != p


def test_budget_and_provenance_enforced():
    protocol, goal = config()
    bad = row([1, 1, 1, 1, 0])
    del bad['provenance']
    with pytest.raises(ValueError, match='provenance'):
        search.next_candidate([bad], [], protocol, goal)
    protocol['new_candidate_budget'] = 1
    with pytest.raises(ValueError, match='exhausted'):
        search.next_candidate([], [row([1, 1, 1, 1, 0])], protocol, goal)


def test_timing_quantization_is_retained_in_measured_rows():
    protocol, goal = config()
    prior = prior_design()
    for r in prior:
        r['evaluation']['features'][-1] = round(r['evaluation']['features'][-1] / .02) * .02
    before = copy.deepcopy(prior)
    search.next_candidate(prior, [], protocol, goal)
    assert prior == before
