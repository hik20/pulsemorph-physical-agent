"""Use actual saved Windows manifests; no physics execution."""
import importlib
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
audit=importlib.import_module('audit_tool_compatibility_smoke')
runner=importlib.import_module('run_tool_compatibility_smoke_v2')


def test_actual_saved_windows_pin_keys_normalize_exact_switch():
    root=audit.ROOT/'artifacts/runs/agentic'
    assert audit.sha256(str(root/'tool_selection_development_v2/protocol.json'))==audit.sha256(root/'tool_selection_development_v2/protocol.json')
    old=audit.read(root/'tool_selection_development_v2/trials/wave_a__llm_feedback__7/manifest.json')
    new=audit.read(root/'tool_selection_compatible_smoke_v1/trials/compatibility_smoke__wave_a/manifest.json')
    assert any('\\' in key for key in old['preexecution_input_hashes'])
    assert any('\\' in key for key in new['preexecution_input_hashes'])
    assert audit.input_pin_parity(old,new)['only_declared_executor_switch']
    key=next(k for k in new['preexecution_input_hashes'] if k.endswith('g1_holo_compat.xml'))
    new['preexecution_input_hashes'][key]='0'*64
    with pytest.raises(ValueError,match='shared input pin'):audit.input_pin_parity(old,new)


def test_normalization_collision_rejected():
    with pytest.raises(ValueError,match='collision'):
        audit.normalized_pins({'a/b':'x','a\\b':'x'})


def test_new_runner_one_local_case_uses_corrected_normalized_verifier():
    assert runner.CASES==(('high5_b','local_clock_1p4_2p8_v1'),)
    assert runner.input_pin_parity is audit.input_pin_parity
    text=(audit.ROOT/'experiments/agentic/run_tool_compatibility_smoke_v2.py').read_text()
    assert "'max_physical_trials':1" in text
    assert "choices=['high5_b']" in text
    assert 'audit_tool_compatibility_smoke.py' in text
