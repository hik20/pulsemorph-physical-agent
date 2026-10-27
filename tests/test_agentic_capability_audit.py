"""Capability audit acceptance must follow measured outcomes and frozen metadata."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
from audit_capability import best,check_metadata,check_stopping,check_index


def row(identity='x',passing=False,utility=2.,l2=5.):
    return dict(candidate_id=identity,integrity_passed=True,physical_gate_passed=True,intent_passed=passing,per_dimension_pass=[passing]*5,utility=utility,utility_l2=l2)


def test_requires_all64_for_negative_claim():
    with pytest.raises(ValueError,match='all64'):
        check_stopping({'status':'no_witness_within_budget'},[row()]*63,[],64)
    check_stopping({'status':'no_witness_within_budget'},[row()]*64,[],64)


def test_stop_first_and_one_confirmation():
    summary={'status':'confirmed_witness'}
    check_stopping(summary,[row(passing=True)],[row(passing=True)],64)
    for history,confirm in [([row(passing=True),row()], [row(passing=True)]),([row(passing=True)],[]),([row(passing=True)],[row(passing=True)]*2)]:
        with pytest.raises(ValueError):check_stopping(summary,history,confirm,64)
    with pytest.raises(ValueError,match='budget'):check_stopping(summary,[row()]*65,[],64)


def test_failure_is_not_negative_evidence():
    check_stopping({'status':'execution_error','error':'runtime failed'},[],[],64)
    with pytest.raises(ValueError):check_stopping({'status':'execution_error','error':None},[],[],64)


def test_metadata_links_exact_search_and_preserves_runtime_schema():
    decision=dict(algorithm_version='capability_witness_v1',prediction_is_physical_evidence=False,new_candidate_index=2)
    proposal=dict(method='surrogate_search',search_algorithm=decision)
    manifest=dict(schema='pulsemorph.agentic_generalization_trial.v1',producer={'kind':'baseline','id':'generalization-surrogate_search','version':'2'})
    check_metadata(proposal,manifest,decision)
    with pytest.raises(ValueError,match='mislabeled'):check_metadata(dict(proposal,method='llm_feedback'),manifest,decision)
    with pytest.raises(ValueError,match='linkage'):check_metadata(proposal,manifest,dict(decision,new_candidate_index=3))
    bad=dict(decision,prediction_is_physical_evidence=True)
    with pytest.raises(ValueError,match='prediction'):check_metadata(dict(proposal,search_algorithm=bad),manifest,bad)


def test_best_uses_worst_l2_then_identity_and_gate():
    rows=[row('z',utility=1,l2=3),row('b',utility=1,l2=2),row('a',utility=1,l2=2),dict(row('bad',utility=0),physical_gate_passed=False)]
    assert best(rows)['candidate_id']=='a'


def test_canonical_index_is_strict_despite_disclosed_legacy_zero():
    proposal=dict(candidate_id="high5_b__capability__07",trial_index=0,search_algorithm={"new_candidate_index":7})
    check_index(proposal,7)
    with pytest.raises(ValueError,match="canonical"):check_index(proposal,6)
    proposal["search_algorithm"]["new_candidate_index"]=0
    with pytest.raises(ValueError,match="canonical"):check_index(proposal,7)
