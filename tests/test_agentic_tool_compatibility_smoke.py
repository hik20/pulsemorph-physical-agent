"""Smoke helper parity checks only; no simulation/model calls."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
smoke=importlib.import_module('run_tool_compatibility_smoke')


def test_full_trace_parity_checks_decompressed_payload_and_dtype(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    for d in (a,b):
        (d/'common_reference').mkdir(parents=True)
        for n in ('qpos.npy','qvel.npy','applied_joint_targets.npy','state_times.npy'):
            np.save(d/n,np.ones((2,2)))
        for n in ('policy_inputs.npz','policy_outputs.npz','reference_cache.npz','common_reference/arrays.npz'):
            np.savez_compressed(d/n,x=np.ones((2,2)))
    assert smoke.trace_parity(a,b)['common_all_four_arrays_identical']
    np.savez_compressed(b/'policy_outputs.npz',x=np.ones((2,2),dtype=np.float32))
    with pytest.raises(ValueError,match='dtype/shape/bytes'):smoke.trace_parity(a,b)


def test_preexecution_requires_exact_executor_switch_only(monkeypatch):
    monkeypatch.setattr(smoke,'sha256',lambda p:'new')
    shared={key:'same' for key in ('onnx_sha256','motion_sha256','body_profile_sha256',
        'controller_profile_sha256','checkpoint_sha256','physics_contract','versions')}
    a=dict(shared,preexecution_input_hashes={smoke.OLD_RUNTIME:'old','body':'body'})
    b=dict(shared,preexecution_input_hashes={smoke.RUNTIME:'new','body':'body'})
    assert smoke.input_pin_parity(a,b)['only_declared_executor_switch']
    b['preexecution_input_hashes']['body']='changed'
    with pytest.raises(ValueError,match='shared preexecution'):smoke.input_pin_parity(a,b)


def test_smoke_refuses_incomplete_comparison_before_output(tmp_path):
    from musical_tasks import write_new_json
    pilot=tmp_path/'pilot';pilot.mkdir()
    write_new_json(pilot/'comparison.json',{'comparison_valid':False,'physical_trials':1})
    write_new_json(pilot/'protocol.json',{})
    monkey_output=smoke.ROOT/'artifacts/runs/agentic/never_created_smoke_test'
    with pytest.raises(ValueError,match='completed original100'):smoke.initialize(pilot,monkey_output)
    assert not monkey_output.exists()
