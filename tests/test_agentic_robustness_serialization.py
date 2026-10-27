"""Serialization-only repair: actual quarantined artifacts are read, never written."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/agentic'))
import evaluate_robustness_v2 as evaluator


def test_numpy_scalars_become_native_json_without_numeric_change():
    source = {'passes':[np.bool_(True),np.bool_(False)],'number':np.float64(.12345678901234568),
              'int':np.int64(123),'none':None,'nested':({'x':np.float32(.3)},)}
    result = evaluator.to_json_native(source)
    assert result == json.loads(json.dumps(result,allow_nan=False))
    assert [type(value) for value in result['passes']] == [bool,bool]
    assert result['number'].hex() == float(source['number']).hex()
    assert result['nested'][0]['x'].hex() == float(source['nested'][0]['x']).hex()
    assert result['none'] is None and type(result['int']) is int


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf'),np.float32('nan'),np.float64('inf')])
def test_no_hidden_nonfinite_numbers(bad):
    with pytest.raises(ValueError,match='nonfinite'):
        evaluator.to_json_native({'nested':[None,{'metric':bad}]})


def test_score_features_actual_numpy_boolean_branch_json_roundtrip():
    features = [1.,1.,1.,1.,2.5]
    result = evaluator.score_features(features,np.array(features),np.ones(5)*.1)
    assert all(type(value) is bool for value in result['per_dimension_pass'])
    assert json.loads(json.dumps(result,allow_nan=False)) == result
    assert result['intent_passed'] and result['utility']==0.


def test_existing_three_nominal_artifacts_read_evaluate_json_roundtrip():
    panel = ROOT/'artifacts/runs/agentic/robustness_initial_state_v1'
    trials = [panel/'trials'/identity for identity in ('nominal__calibration_wave',
        'nominal__calibration_high5','nominal__selected_wave_a')]
    # Completed v1 runtime artifacts exist despite the later report-write failure.
    assert all((trial/'manifest.json').is_file() for trial in trials)
    before = {str(path):path.read_bytes() for trial in trials for path in trial.glob('*.json')}
    audited = [evaluator._original._base._audit(trial) for trial in trials]
    assert all(value['manifest']['status']=='success' for value in audited)
    protocol = json.loads((panel/'protocol.json').read_text())
    goal = protocol['goals']['wave_a']
    original = evaluator._original.evaluate_robustness(trials[2],trials[0],trials[0],protocol,goal)
    repaired = evaluator.evaluate_robustness(trials[2],trials[0],trials[0],protocol,goal)
    assert repaired['integrity_passed'],repaired
    assert repaired == evaluator.to_json_native(original)
    assert json.loads(json.dumps(repaired,allow_nan=False)) == repaired
    assert all(type(value) is bool for value in repaired['primary']['per_dimension_pass'])
    assert repaired['primary']==repaired['secondary']
    import audit_robustness_v2 as auditor
    center_check = auditor.check_center_definitions(repaired,audited[2]['arrays'],audited[0]['arrays'],
        audited[0]['arrays'],audited[2]['names'],protocol,goal)
    audit_report = {'selected_rows':[{'evaluation':repaired,'independent_center_check':center_check}]}
    assert json.loads(json.dumps(audit_report,allow_nan=False))==audit_report
    assert all(Path(path).read_bytes()==content for path,content in before.items())


def test_v2_auditor_preserves_quarantine_cost_and_all_v1_source_pins():
    import audit_robustness_v2 as auditor
    failed = ROOT/'artifacts/runs/agentic/robustness_initial_state_v1'
    quarantine = ROOT/'artifacts/runs/agentic/robustness_initial_state_v1_quarantine_v1'
    protocol = json.loads((failed/'protocol.json').read_text())
    protocol['schema'] = 'pulsemorph.agentic_initial_state_robustness_protocol.v2'
    def link(path): return {'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':auditor.sha(path)}
    protocol['supersedes_failure'] = link(failed/'summary.json')
    protocol['quarantine'] = link(quarantine/'summary.json')
    protocol['quarantine_inventory'] = link(quarantine/'inventory.json')
    protocol['quarantined_cost_excluded_from_this_panel'] = dict(physical_trials=3,control_steps=600,
        physics_steps=12000,new_llm_calls=0,nonzero_physical_trials=0)
    for name in ('robustness_plan_v2.py','robustness_driver_v2.py','evaluate_robustness_v2.py','audit_robustness_v2.py'):
        relative = 'experiments/agentic/'+name
        protocol['execution_code_sha256'][relative] = auditor.sha(ROOT/relative)
    checked = auditor.check_quarantine(protocol)
    assert checked['preserved_failure'] and checked['actual_aborted_cost']['physics_steps']==12000
    protocol['feature_tolerances'][0] += .01
    with pytest.raises(ValueError,match='changed frozen experiment'):
        auditor.check_quarantine(protocol)
