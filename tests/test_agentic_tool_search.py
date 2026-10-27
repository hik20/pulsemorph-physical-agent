"""Mixed categorical search invariants; synthetic measurements only."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('tool_search', Path(__file__).parents[1] / 'experiments/agentic/tool_selection_search.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def setup():
    protocol = dict(parameter_names=s.PARAMETER_NAMES.copy(), bounds=s.BOUNDS.tolist(), feature_tolerances=[.08] * 4 + [.1])
    goal = dict(feature_targets=[.8, 1.2, 1.2, .8, 2.1])
    cal = dict(parameters=s.IDENTITY.tolist(), provenance='source:identity:sha256-test',
               evaluation=dict(features=[1, 1, 1, 1, 2.2], integrity_passed=True, physical_gate_passed=True))
    return protocol, goal, cal


def measured(candidate):
    row = copy.deepcopy(candidate)
    x = np.asarray(row['parameters'])
    row['evaluation'] = dict(features=[*x[:4], 2.2 - x[4]], integrity_passed=True, physical_gate_passed=True)
    return row


def test_schedule_budget_bounds_determinism_and_no_mutation():
    p, g, c = setup()
    history = []
    strategies = []
    for n in range(8):
        before = copy.deepcopy((history, p, g, c))
        candidate = s.next_candidate(history, p, g, c)
        assert candidate == s.next_candidate(history, p, g, c)
        assert before == (history, p, g, c)
        x = np.array(candidate['parameters'])
        assert np.all(x >= s.BOUNDS[:, 0]) and np.all(x <= s.BOUNDS[:, 1])
        assert all(candidate['control_tool'] != r['control_tool'] or not np.allclose(x, r['parameters'], atol=1e-9, rtol=0) for r in history)
        assert not candidate['decision']['prediction_is_physical_evidence']
        assert candidate['decision']['calibration_evidence_count'] == 1
        assert max(candidate['decision']['model_points_by_tool'].values()) <= 5
        strategies.append(candidate['decision']['strategy'])
        history.append(measured(candidate))
    assert [r['control_tool'] for r in history[:4]] == list(s.TOOLS) * 2
    assert strategies == ['task_informed_seed'] * 2 + ['maximin_probe'] * 2 + ['model_selection', 'maximin_exploration'] * 2
    with pytest.raises(ValueError, match='budget'):
        s.next_candidate(history, p, g, c)


def test_task_seed_uses_shared_identity_without_duplicate_physics():
    p, g, c = setup()
    a = s.next_candidate([], p, g, c)
    b = s.next_candidate([measured(a)], p, g, c)
    assert a['parameters'] == pytest.approx([.8, 1.2, 1.2, .8, .1])
    assert a['parameters'] == b['parameters']
    assert a['control_tool'] != b['control_tool']


def test_other_tool_feedback_does_not_fit_this_tools_model():
    p, g, c = setup()
    history = []
    for _ in range(4):
        history.append(measured(s.next_candidate(history, p, g, c)))
    a = s.next_candidate(history, p, g, c)
    changed = copy.deepcopy(history)
    for r in changed:
        if r['control_tool'] == s.TOOLS[1]:
            r['evaluation']['features'] = [99] * 5
    # Tool balance leaves both eligible, so force the next global probe instead:
    assert s.next_candidate(changed[:2], p, g, c) == s.next_candidate(history[:2], p, g, c)
    assert a['decision']['model_points_by_tool'] == {s.TOOLS[0]: 3, s.TOOLS[1]: 3}
    # At a model slot, fewer committed global attempts makes global eligible;
    # changing only local labels must not affect its fitted proposal.
    history[2]['control_tool'] = s.TOOLS[1]
    history[2]['parameters'][0] = .73
    isolated = s.next_candidate(history, p, g, c)
    changed = copy.deepcopy(history)
    for r in changed:
        if r['control_tool'] == s.TOOLS[1]:
            r['evaluation']['features'] = [99] * 5
    assert isolated == s.next_candidate(changed, p, g, c)


def test_mislabeled_tool_evaluation_is_excluded():
    p, g, c = setup()
    r = measured(s.next_candidate([], p, g, c))
    r['evaluation']['control_tool'] = s.TOOLS[1]
    result = s.next_candidate([r], p, g, c)
    assert result['decision']['model_points_by_tool'][s.TOOLS[0]] == 1


@pytest.mark.parametrize('failure', ['integrity_passed', 'physical_gate_passed', 'features'])
def test_invalid_labels_excluded_but_slot_and_coordinates_retained(failure):
    p, g, c = setup()
    h = [measured(s.next_candidate([], p, g, c))]
    h[0]['evaluation'][failure] = None
    a = s.next_candidate(h, p, g, c)
    h[0]['evaluation']['features'] = [999] * 5 if failure != 'features' else [None] * 5
    assert a == s.next_candidate(h, p, g, c)
    assert a['decision']['new_candidate_index'] == 1
    assert a['decision']['model_points_by_tool'][s.TOOLS[0]] == 1


def test_malformed_proposal_consumes_slot_and_unknown_tool_never_fits():
    p, g, c = setup()
    h = [dict(control_tool='unknown', parameters=None, evaluation=dict(features=[99] * 5, integrity_passed=True, physical_gate_passed=True))]
    result = s.next_candidate(h, p, g, c)
    assert result['control_tool'] == s.TOOLS[1]
    assert set(result['decision']['model_points_by_tool'].values()) == {1}


def test_duplicate_pair_rejected_cross_tool_allowed_and_calibration_required():
    p, g, c = setup()
    r = measured(s.next_candidate([], p, g, c))
    with pytest.raises(ValueError, match='duplicate'):
        s.next_candidate([r, r], p, g, c)
    other = copy.deepcopy(r)
    other['control_tool'] = s.TOOLS[1]
    s.next_candidate([r, other], p, g, c)
    for key in ['provenance', 'parameters', 'evaluation']:
        invalid = copy.deepcopy(c)
        invalid.pop(key)
        with pytest.raises(ValueError, match='calibration'):
            s.next_candidate([], p, g, invalid)
