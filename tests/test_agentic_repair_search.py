"""Synthetic numerical repair invariants; no physics or provider calls."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('repair_search', Path(__file__).parents[1]/'experiments/agentic/repair_search.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
CONDITIONS = ['nominal', 'joint_position_minus', 'combined_plus', 'combined_minus']


def features(x, state):
    x = np.asarray(x)
    y = x.copy(); y[4] = 2-x[4]
    if state != 'nominal': y[0] += {'joint_position_minus': .02, 'combined_plus': -.01, 'combined_minus': .03}[state]
    return y.tolist()


def row(tool, x, states=('nominal',)):
    return dict(control_tool=tool, parameters=list(x), provenance='synthetic-test',
        state_evaluations=[dict(scenario_id=state, features=features(x, state),
            integrity_passed=True, physical_gate_passed=True) for state in states])


def setup():
    p = dict(parameter_names=s.PARAMETER_NAMES, bounds=s.BOUNDS.tolist(),
        feature_tolerances=[.08]*4+[.1], new_candidate_budget=8, development_condition_ids=CONDITIONS)
    g = dict(feature_targets=[.85, 1.1, 1.1, .9, 2.08])
    anchor = np.array([1., 1., 1., 1., 0.])
    prior = []
    for tool in s.TOOLS:
        prior.append(row(tool, anchor))
        for axis in range(5):
            x = anchor.copy(); x[axis] += .1
            prior.append(row(tool, x))
    prior.append(row(s.TOOLS[0], anchor, CONDITIONS))
    return p, g, prior


def test_determinism_bounds_budget_no_mutation_and_finite_json():
    p, g, prior = setup(); h = []
    for _ in range(8):
        before = copy.deepcopy((p, g, prior, h))
        a = s.next_candidate(h, p, g, prior)
        assert a == s.next_candidate(h, p, g, prior)
        assert before == (p, g, prior, h)
        json.dumps(a, allow_nan=False)
        x = np.array(a['parameters'])
        assert np.all(x >= s.BOUNDS[:, 0]) and np.all(x <= s.BOUNDS[:, 1])
        assert all(a['control_tool'] != r['control_tool'] or not np.allclose(x, r['parameters'], atol=1e-9, rtol=0) for r in prior+h)
        measured = row(a['control_tool'], x, CONDITIONS)
        measured['search_diagnostics'] = a['search_diagnostics']; h.append(measured)
    with pytest.raises(ValueError, match='budget'): s.next_candidate(h, p, g, prior)


def test_affine_joint_repair_reduces_actual_worst_multistate_error():
    p, g, prior = setup()
    a = s.next_candidate([], p, g, prior)
    target = np.array(g['feature_targets']); tol = np.array(p['feature_tolerances'])
    actual = np.array([(np.array(features(a['parameters'], sid))-target)/tol for sid in CONDITIONS])
    assert np.max(np.abs(actual)) < .35
    assert a['search_diagnostics']['prediction_is_physical_evidence'] is False
    assert a['search_diagnostics']['models_by_tool'][s.TOOLS[0]]['nominal_design_rank'] == 5


@pytest.mark.parametrize('failure', ['physical_gate_passed', 'integrity_passed', 'features'])
def test_invalid_history_labels_excluded_but_slot_and_point_consumed(failure):
    p, g, prior = setup()
    bad = row(s.TOOLS[1], [1.3, 1.3, 1.3, 1.3, .2], CONDITIONS)
    for e in bad['state_evaluations']: e[failure] = None
    a = s.next_candidate([bad], p, g, prior)
    other = copy.deepcopy(bad)
    for e in other['state_evaluations']: e['features'] = [None]*5 if failure == 'features' else [999]*5
    assert a == s.next_candidate([other], p, g, prior)
    assert a['search_diagnostics']['new_candidate_index'] == 1


def test_lp_minimax_precedes_squared_error():
    # Scalar minimax balances -2 and +1 at d=.5, not a pass-count heuristic.
    residual = np.array([[-2., 1., 0., 0., 0.]])
    jacobian = np.zeros((1, 5, 5)); jacobian[0, :2, 0] = 1
    delta, _ = s._solve(residual, jacobian, np.full(5, -1.), np.full(5, 1.))
    assert delta[0] == pytest.approx(.5, abs=1e-7)


def test_required_prior_and_protocol_and_duplicate_history_rejected():
    p, g, prior = setup()
    with pytest.raises(ValueError, match='prior'): s.next_candidate([], p, g, [])
    invalid = copy.deepcopy(prior); invalid[0]['provenance'] = ''
    with pytest.raises(ValueError, match='provenance'): s.next_candidate([], p, g, invalid)
    bad = copy.deepcopy(p); bad['development_condition_ids'] = CONDITIONS[:3]
    with pytest.raises(ValueError, match='protocol'): s.next_candidate([], bad, g, prior)
    h = row(s.TOOLS[0], [1.3]*4+[.1], CONDITIONS)
    with pytest.raises(ValueError, match='duplicate history'): s.next_candidate([h, h], p, g, prior)


def test_missing_peak_is_not_valid_robust_incumbent():
    p, g, prior = setup()
    prior[-1]['state_evaluations'][-1]['features'][-1] = None
    with pytest.raises(ValueError, match='robust incumbent'): s.next_candidate([], p, g, prior)



def test_exact_duplicate_prior_label_does_not_change_choice_or_fit_weight():
    p, g, prior = setup()
    before = copy.deepcopy(prior)
    a = s.next_candidate([], p, g, prior)
    repeated = copy.deepcopy(prior)
    duplicate = copy.deepcopy(prior[2]); duplicate['provenance'] = 'another-path-same-label'
    repeated.extend([duplicate, copy.deepcopy(prior[-1])])
    b = s.next_candidate([], p, g, repeated)
    assert prior == before
    assert a['control_tool'] == b['control_tool']
    assert a['parameters'] == b['parameters']
    assert a['search_diagnostics']['predicted_worst_normalized_error'] == b['search_diagnostics']['predicted_worst_normalized_error']
    assert a['search_diagnostics']['models_by_tool'][s.TOOLS[0]]['nominal_points'] == b['search_diagnostics']['models_by_tool'][s.TOOLS[0]]['nominal_points']
    # Raw prior provenance/count remains present; distinct measured outcomes
    # at identical coordinates must continue to influence the fit.
    assert b['search_diagnostics']['shared_prior_observation_count'] == len(repeated)
    duplicate['state_evaluations'][0]['features'][0] += .2
    changed = s.next_candidate([], p, g, prior+[duplicate])
    assert changed['search_diagnostics']['models_by_tool'][s.TOOLS[0]]['nominal_points'] == a['search_diagnostics']['models_by_tool'][s.TOOLS[0]]['nominal_points'] + 1
