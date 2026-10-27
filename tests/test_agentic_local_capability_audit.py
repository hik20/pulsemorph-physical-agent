"""Local phase audit checks clean index metadata and fresh-measurement seeding."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
from audit_local_capability import check_index,check_metadata,check_stopping,check_input_extras,ROOT,sha,CONTROL_TOOL
from local_capability_driver import next_local_candidate


def test_clean_index_and_stage_required():
    p=dict(candidate_id='high5_b__local_capability__07',trial_index=7,stage='search',search_algorithm={'new_candidate_index':7})
    check_index(p,7)
    for change in ({'trial_index':0},{'stage':'confirmation'},{'candidate_id':'high5_b__local_capability__06'},{'search_algorithm':{'new_candidate_index':0}}):
        with pytest.raises(ValueError):check_index(dict(p,**change),7)


def test_only_declared_two_extra_inputs_and_unchanged_old_subset():
    extras=['experiments/agentic/execute_local_capability.py','src/pulsemorph/agentic/local_multifeature_reference.py']
    old={'existing/file':'old'};new={**old,**{k:sha(ROOT/k) for k in extras}}
    check_input_extras(old,new,extras)
    for bad in (dict(new,**{'existing/file':'changed'}),dict(new,undeclared='x')):
        with pytest.raises(ValueError):check_input_extras(old,bad,extras)


def test_seed_uses_fresh_measured_denominator_and_peak():
    protocol={'new_candidate_budget':64,'bounds':[[.35,1.6]]*4+[[-.3,.3]]}
    features=[1.2,.8,1.1,.9,2.1];target=[.9,1.,1.,.8,2.2]
    calibration=[dict(features=features,provenance={'candidate_id':'calibration'})]
    result=next_local_candidate(calibration,[],protocol,{'feature_targets':target})
    phase=np.sin(np.pi*(target[4]-1.4)/1.4)**2
    expected=np.r_[np.asarray(target[:4])/features[:4],(features[4]-target[4])/phase]
    assert np.array_equal(result['parameters'],expected)
    assert result['decision']['prediction_is_physical_evidence'] is False
    with pytest.raises(ValueError):next_local_candidate([],[],protocol,{'feature_targets':target})


def test_stopping_and_local_method_linkage():
    row=dict(integrity_passed=True,physical_gate_passed=True,intent_passed=True,per_dimension_pass=[True]*5)
    check_stopping({'status':'confirmed_witness'},[row],[row],64)
    with pytest.raises(ValueError):check_stopping({'status':'confirmed_witness'},[row,row],[row],64)
    with pytest.raises(ValueError):check_stopping({'status':'no_witness_within_budget'},[],[],64)
    d=dict(algorithm_version='local_capability_search_v1',prediction_is_physical_evidence=False)
    p=dict(method='surrogate_search',search_algorithm=d)
    m=dict(schema='pulsemorph.agentic_generalization_trial.v1',producer=dict(kind='baseline',id='local-capability-surrogate_search',version='2',control_tool=CONTROL_TOOL))
    check_metadata(p,m,d)
    with pytest.raises(ValueError):check_metadata(dict(p,method='llm_feedback'),m,d)
    with pytest.raises(ValueError):check_metadata(p,dict(m,producer=dict(m['producer'],control_tool='global')),d)


def test_audit_source_capture_pins_copied_bytes_and_detects_change(tmp_path):
    from audit_local_capability import capture_sources,source_hashes
    root=tmp_path/'source';root.mkdir();(root/'helper.py').write_text('original')
    output=tmp_path/'audit';output.mkdir()
    before=capture_sources(output,root,('helper.py',))
    assert (output/'audit_source_snapshot/helper.py').read_text()=='original'
    assert source_hashes(root,('helper.py',))==before
    (root/'helper.py').write_text('changed')
    assert source_hashes(root,('helper.py',))!=before
    assert (output/'audit_source_snapshot/helper.py').read_text()=='original'


def test_frozen_local_runtime_producer_integration_rejects_global_identity():
    from execute_local_capability import proposal_producer
    from pulsemorph.motion.contracts import load_reference
    pilot=ROOT/'artifacts/runs/agentic/high5_local_capability_v1'
    from audit_local_capability import read
    proposal=read(pilot/'proposals/high5_b__local_capability__00.json')
    manifest=read(pilot/'trials/high5_b__local_capability__00/manifest.json')
    expected=proposal_producer(proposal)
    expected['control_tool']=CONTROL_TOOL
    assert expected==manifest['producer']==load_reference(pilot/'trials/high5_b__local_capability__00/common_reference').metadata['producer']
    check_metadata(proposal,manifest,proposal['search_algorithm'])
    wrong=dict(manifest,producer=dict(expected,id='generalization-surrogate_search'))
    with pytest.raises(ValueError,match='producer linkage'):check_metadata(proposal,wrong,proposal['search_algorithm'])
