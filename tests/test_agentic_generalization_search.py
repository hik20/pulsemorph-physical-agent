"""Numerical comparator invariants; no simulator or model calls."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location('generalization_search', Path(__file__).parents[1] / 'experiments/agentic/generalization_search.py')
search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search)


def config():
    return dict(parameter_names=search.PARAMETER_NAMES.copy(), bounds=[[.35, 1.6]] * 4 + [[-.3, .3]],
                feature_tolerances=[.1] * 4 + [.08]), dict(feature_targets=[.7, 1.2, .8, 1.1, 2.1])


def row(parameters, features, **status):
    return dict(parameters=parameters, evaluation=dict(features=features, integrity_passed=True,
                physical_gate_passed=True, **status))


def test_anchor_and_task_informed_proposal():
    protocol, goal = config()
    assert search.next_candidate([], protocol, goal) == [1, 1, 1, 1, 0]
    history = [row([1, 1, 1, 1, 0], [1, 1, 1, 1, 2])]
    assert search.next_candidate(history, protocol, goal) == pytest.approx([.7, 1.2, .8, 1.1, -.1])


@pytest.mark.parametrize('observed_peak,expected_shift', [(2.3, .2), (1.9, -.2)])
def test_late_event_advances_and_early_event_delays_source_time(observed_peak, expected_shift):
    protocol, goal = config()
    history = [row([1, 1, 1, 1, 0], [1, 1, 1, 1, observed_peak])]
    proposed = search.next_candidate(history, protocol, goal)
    # In tool source_time=t+shift*sin²(...), positive shift advances a peak.
    # This is a directional heuristic, not an exact inverse of that time warp.
    assert proposed[4] == pytest.approx(expected_shift)


def test_deterministic_bounded_fresh_and_no_mutation():
    protocol, goal = config()
    history = []
    for _ in range(14):
        before = copy.deepcopy((history, protocol, goal))
        candidate = search.next_candidate(history, protocol, goal)
        assert candidate == search.next_candidate(history, protocol, goal)
        assert (history, protocol, goal) == before
        assert np.all(np.asarray(candidate) >= np.asarray(protocol['bounds'])[:, 0])
        assert np.all(np.asarray(candidate) <= np.asarray(protocol['bounds'])[:, 1])
        assert all(not np.allclose(candidate, entry['parameters'], atol=1e-10, rtol=0) for entry in history)
        # Coupled smooth synthetic response checks multivariate fitting paths.
        x = np.array(candidate)
        features = (x + .05 * np.roll(x, 1)).tolist()
        features[4] += 2
        history.append(row(candidate, features))


@pytest.mark.parametrize('failure', ['missing_peak', 'integrity', 'physical', 'missing_status'])
def test_invalid_observations_never_fit_or_invent_features(failure):
    protocol, goal = config()
    bad = row([1, 1, 1, 1, 0], [1, 1, 1, 1, 2])
    if failure == 'missing_peak':
        bad['evaluation']['features'][4] = None
    elif failure == 'missing_status':
        del bad['evaluation']['integrity_passed']
    else:
        bad['evaluation'][failure + ('_passed' if failure == 'integrity' else '_gate_passed')] = False
    result = search.next_candidate([bad], protocol, goal)
    other = copy.deepcopy(bad)
    other['evaluation']['features'] = None
    assert result == search.next_candidate([other], protocol, goal)


def test_duplicate_observations_and_invalid_configuration():
    protocol, goal = config()
    repeated = row([1, 1, 1, 1, 0], [1, 1, 1, 1, 2])
    assert np.isfinite(search.next_candidate([repeated, repeated], protocol, goal)).all()
    protocol['feature_tolerances'][0] = 0
    with pytest.raises(ValueError):
        search.next_candidate([], protocol, goal)
