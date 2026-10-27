"""Static panel invariants; no physics or policy calls."""
import importlib
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/agentic'))
panel = importlib.import_module('robustness_scenarios')


def test_json_deterministic_sign_pairs_and_levels():
    scenarios = panel.build_scenarios(['hip', 'knee', 'shoulder'])
    assert len(scenarios) == len({s['scenario_id'] for s in scenarios}) == 13
    assert json.dumps(scenarios, allow_nan=False) == json.dumps(panel.build_scenarios(['hip', 'knee', 'shoulder']))
    keys = ('joint_position_offset_rad', 'joint_velocity_offset_rad_s', 'root_linear_velocity_offset_world_m_s')
    assert set(scenarios[0]) == {'schema_version', 'scenario_id', 'kind', 'sign', 'seed', 'joint_names', 'initial_perturbation'}
    for scenario in scenarios:
        assert set(scenario['initial_perturbation']) == set(keys) | {'root_yaw_offset_rad'}
    values = [s['initial_perturbation'] for s in scenarios]
    zero = values[0]
    assert zero['root_yaw_offset_rad'] == 0
    assert all(v == 0 for key in keys for v in zero[key])
    for plus, minus in zip(scenarios[1::2], scenarios[2::2]):
        assert plus['kind'] == minus['kind']
        for key in keys:
            assert plus['initial_perturbation'][key] == [-v for v in minus['initial_perturbation'][key]]
        assert plus['initial_perturbation']['root_yaw_offset_rad'] == -minus['initial_perturbation']['root_yaw_offset_rad']
    assert all(abs(v) == .01 for v in values[1][keys[0]])
    assert all(abs(v) == .05 for v in values[3][keys[1]])
    assert values[5][keys[2]] == [.03, 0, 0]
    assert values[7][keys[2]] == [0, .03, 0]
    assert values[9]['root_yaw_offset_rad'] == math.pi/180
    combined = values[11]
    assert combined[keys[0]] == values[1][keys[0]]
    assert combined[keys[1]] == values[3][keys[1]]
    assert combined[keys[2]] == [.03, .03, 0]


def test_named_hash_is_order_independent_and_copies_are_independent():
    a = panel.build_scenarios(['hip', 'knee'])
    b = panel.build_scenarios(['knee', 'hip'])
    assert a[1]['initial_perturbation']['joint_position_offset_rad'] == b[1]['initial_perturbation']['joint_position_offset_rad'][::-1]
    assert a[3]['initial_perturbation']['joint_velocity_offset_rad_s'] == b[3]['initial_perturbation']['joint_velocity_offset_rad_s'][::-1]
    a[0]['joint_names'].append('extra')
    assert a[1]['joint_names'] == ['hip', 'knee']


@pytest.mark.parametrize('names,seed', [([], 's'), (['x', 'x'], 's'), ([''], 's'), ([3], 's'), (['x'], '')])
def test_reject_ambiguous_inputs(names, seed):
    with pytest.raises(ValueError):
        panel.build_scenarios(names, seed)


def test_world_yaw_left_product_wxyz_and_inverse():
    # 90-degree x roll makes multiplication side observable.
    c = math.sqrt(.5)
    q = [c, c, 0, 0]
    rotated = panel.yaw_left_multiply_wxyz(q, math.pi/2)
    assert rotated == pytest.approx([.5, .5, .5, .5])
    assert panel.yaw_left_multiply_wxyz(rotated, -math.pi/2) == pytest.approx(q)
    assert sum(v*v for v in rotated) == pytest.approx(1)
    assert panel.yaw_left_multiply_wxyz(q, 0) == q
    with pytest.raises(ValueError):
        panel.yaw_left_multiply_wxyz([1, 1, 0, 0], 0)
