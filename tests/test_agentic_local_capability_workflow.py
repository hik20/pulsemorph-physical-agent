"""Mock execution proves accounting and fail-closed stopping without physics."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('local_capability_driver',Path(__file__).parents[1]/'experiments/agentic/local_capability_driver.py')
d=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(d)


def fixture(tmp_path, monkeypatch, outcomes, budget=3, difference=False, calibration_difference=False):
    out=tmp_path/'pilot';out.mkdir(); old=tmp_path/'old';old.mkdir()
    pins={'preexecution_input_hashes':{},'onnx_sha256':'x','motion_sha256':'y','physics_contract':{},'versions':{}}
    d.write(old/'manifest.json',dict(pins,candidate_id='old',status='success'))
    for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy'):np.save(old/name,np.zeros((2,2)))
    protocol={'task':{'task_id':'high5_b','source_motion':{'path':'source','sha256':'x'}},'goal':{'feature_targets':[1.1,.9,1.1,.9,2.5]},
              'new_candidate_budget':budget,'calibration_trials':1,'confirmation_trials':1,'execution_code_sha256':{'pin':'x'},
              'prior_trial_records':[],'calibration_proposal':'proposals/calibration.json','bounds':[[0,2]]*5,'anchor':[1,1,1,1,0],'fixed_control_tool':'local_clock_1p4_2p8_v1','declared_input_hash_extras':[],
              'old_protocol':d.artifact(old/'manifest.json'),'old_audit':d.artifact(old/'manifest.json'),
              'old_calibration':{'execution_manifest':d.artifact(old/'manifest.json')}}
    d.write(out/'protocol.json',protocol)
    d.write(out/'proposals/calibration.json',{'candidate_id':'calibration','source_motion':protocol['task']['source_motion'],
                                            'parameters':dict(zip(d.NAMES,protocol['anchor'])),'method':'calibration',
                                            'control_tool':'local_clock_1p4_2p8_v1','task_id':'high5_b','duration_s':4,'source_start_s':0,'protocol_sha256':d.sha(out/'protocol.json')})
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
        return {'integrity_passed':True,'physical_gate_passed':good,'intent_passed':good,'per_dimension_pass':[good]*5,'utility':0 if good else 2,'features':[1,1,1,1,2.52]}
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
    assert calls[1]['method']=='surrogate_search' and calls[1]['search_algorithm']['strategy']=='target_informed_seed'
    assert calls[1]['trial_index']==calls[2]['trial_index']==0
    assert calls[2]['stage']=='confirmation'
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


def test_seed_and_fixed_probes_replay():
    protocol={'new_candidate_budget':64,'bounds':[[.2,1.8]]*4+[[-.3,.3]]}
    goal={'feature_targets':[1.2,.8,1.1,.9,2.5]}
    calibration=[{'features':[1,1,1,1,2.54],'provenance':{'proof':'fresh'}}]
    seed=d.next_local_candidate(calibration,[],protocol,goal)['parameters']
    assert seed[:4]==[1.2,.8,1.1,.9]
    assert seed[4]==pytest.approx(.04/np.sin(np.pi*(2.5-1.4)/1.4)**2)
    for n in range(1,6):
        values=d.next_local_candidate(calibration,[{}]*n,protocol,goal)['parameters']
        axis=n-1
        assert all(values[i]==seed[i] for i in range(5) if i!=axis)
    captured=[]
    def suggest(prior,history,p,g):
        captured.append((prior,history));return {'parameters':seed,'decision':{}}
    d.next_local_candidate(calibration,[{}]*6,protocol,goal,suggest)
    assert captured[0][0] is calibration and len(captured[0][1])==6


def test_probe_flips_at_upper_bound():
    p={'new_candidate_budget':64,'bounds':[[.2,1.8]]*4+[[-.3,.3]]}
    c=[{'features':[1,1,1,1,2.5],'provenance':{'fresh':True}}]
    g={'feature_targets':[1.8,1,1,1,2.5]}
    result=d.next_local_candidate(c,[{}],p,g)
    assert result['parameters'][0]==pytest.approx(1.6)



def test_driver_proposal_matches_actual_local_producer(tmp_path,monkeypatch):
    out,calls,run=fixture(tmp_path,monkeypatch,[True,True])
    result=run()
    runtime_path=Path(__file__).parents[1]/'experiments/agentic/execute_local_capability.py'
    spec=importlib.util.spec_from_file_location('local_runtime_integration',runtime_path)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    assert result['status']=='confirmed_witness'
    assert result['imported_prior_candidates']==0 and result['fresh_calibration_as_search_prior']==1
    for proposal in calls:
        producer=runtime.proposal_producer(proposal)
        assert producer['id']=='local-capability-'+proposal['method']
        assert proposal['control_tool']==d.CONTROL_TOOL
