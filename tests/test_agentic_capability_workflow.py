"""Mock execution proves accounting and fail-closed stopping without physics."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('capability_driver',Path(__file__).parents[1]/'experiments/agentic/capability_driver.py')
d=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(d)


def fixture(tmp_path, monkeypatch, outcomes, budget=3, difference=False, calibration_difference=False):
    out=tmp_path/'pilot';out.mkdir(); old=tmp_path/'old';old.mkdir()
    pins={'preexecution_input_hashes':{},'onnx_sha256':'x','motion_sha256':'y','physics_contract':{},'versions':{}}
    d.write(old/'manifest.json',dict(pins,candidate_id='old',status='success'))
    for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy'):np.save(old/name,np.zeros((2,2)))
    protocol={'task':{'task_id':'high5_b','source_motion':{'path':'source','sha256':'x'}},'goal':{},
              'new_candidate_budget':budget,'calibration_trials':1,'confirmation_trials':1,'execution_code_sha256':{'pin':'x'},
              'prior_trial_records':[],'calibration_proposal':'proposals/calibration.json','bounds':[[0,2]]*5,'anchor':[1,1,1,1,0],
              'old_protocol':d.artifact(old/'manifest.json'),'old_audit':d.artifact(old/'manifest.json'),
              'old_calibration':{'execution_manifest':d.artifact(old/'manifest.json')}}
    d.write(out/'protocol.json',protocol)
    d.write(out/'proposals/calibration.json',{'candidate_id':'calibration','source_motion':protocol['task']['source_motion'],
                                            'parameters':dict(zip(d.NAMES,protocol['anchor'])),'method':'calibration',
                                            'task_id':'high5_b','duration_s':4,'source_start_s':0,'protocol_sha256':d.sha(out/'protocol.json')})
    monkeypatch.setattr(d,'verify_code',lambda p:None)
    calls=[]
    def execute(path,directory,video=False):
        calls.append(d.read(path));directory.mkdir(parents=True)
        manifest=dict(pins,status='success',actual_control_steps=200,actual_physics_steps=4000,online_onnx_call_count=200,warmup_onnx_call_count=1)
        d.write(directory/'manifest.json',manifest)
        for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy'):
            arr=np.zeros((2,2));arr[-1,-1]=int((difference and 'confirm' in directory.name) or (calibration_difference and directory.name=='calibration'));np.save(directory/name,arr)
        return manifest
    def evaluate(directory,baseline,protocol,goal):
        good=False if directory.name=='calibration' else outcomes.pop(0)
        return {'integrity_passed':True,'physical_gate_passed':good,'intent_passed':good,'per_dimension_pass':[good]*5,'utility':0 if good else 2}
    suggest=lambda *args:{'parameters':[1]*5,'decision':{'algorithm':'mock'}}
    return out, calls, lambda:d.run(out,execute,evaluate,suggest)


def test_budget_and_counting(tmp_path,monkeypatch):
    out,calls,run=fixture(tmp_path,monkeypatch,[False]*3)
    result=run()
    assert result['status']=='no_witness_within_budget'
    assert result['total_new_trials']==4 and result['search_candidates']==3
    assert result['warmup_onnx_call_count']==4 and result['actual_physics_steps']==16000


def test_first_witness_confirmed_and_stops(tmp_path,monkeypatch):
    out,calls,run=fixture(tmp_path,monkeypatch,[True,True])
    result=run()
    assert result['status']=='confirmed_witness' and len(calls)==3
    assert calls[1]['parameters']==calls[2]['parameters']
    assert calls[1]['method']=='surrogate_search' and calls[1]['search_algorithm']=={'algorithm':'mock'}
    with pytest.raises(FileExistsError):run()


@pytest.mark.parametrize('difference,outcomes',[(True,[True,True]),(False,[True,False])])
def test_confirmation_failure_no_retry(tmp_path,monkeypatch,difference,outcomes):
    out,calls,run=fixture(tmp_path,monkeypatch,outcomes,difference=difference)
    result=run()
    assert result['status']=='execution_error' and len(calls)==3


def test_invalid_prior_link_stops_before_execution(tmp_path,monkeypatch):
    out,calls,run=fixture(tmp_path,monkeypatch,[])
    protocol=d.read(out/'protocol.json');protocol['prior_trial_records']=[{'candidate_id':'bad',**{k:{'path':str(out/'protocol.json'),'sha256':'wrong'} for k in ('proposal','evaluation','execution_manifest')}}]
    (out/'protocol.json').write_text(json.dumps(protocol))
    result=run()
    assert result['status']=='execution_error' and not calls


def test_valid_requires_all_five_dimensions():
    result={'integrity_passed':True,'physical_gate_passed':True,'intent_passed':True,'per_dimension_pass':[True]*4+[False]}
    assert not d.valid(result)


def test_calibration_later_divergence_same_initial_stops(tmp_path,monkeypatch):
    out,calls,run=fixture(tmp_path,monkeypatch,[],calibration_difference=True)
    result=run()
    assert result['status']=='execution_error' and len(calls)==1
    assert result['search_candidates']==0 and 'exact parity failed' in result['error']


def test_stale_calibration_protocol_no_execution(tmp_path,monkeypatch):
    out,calls,run=fixture(tmp_path,monkeypatch,[])
    path=out/'proposals/calibration.json'; proposal=d.read(path);proposal['protocol_sha256']='stale'
    path.write_text(json.dumps(proposal))
    result=run()
    assert result['status']=='execution_error' and not calls


def test_parity_rejects_signed_zero(tmp_path):
    first=tmp_path/'a';second=tmp_path/'b';first.mkdir();second.mkdir()
    for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy'):
        np.save(first/name,np.array([0.]));np.save(second/name,np.array([-0.]))
    with pytest.raises(ValueError,match='exact parity'):d.parity(first,second)
