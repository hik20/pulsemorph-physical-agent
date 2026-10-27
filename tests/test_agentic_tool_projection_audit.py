"""Compatible-export audit tests; no resimulation or model calls."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/agentic'))
import audit_tool_selection_projection as audit
from project_tool_reference import project_reference
from pulsemorph.motion.contracts import load_reference,save_reference,_plain
from pulsemorph.motion.g1 import assemble_reference_inputs

TRIAL=ROOT/'artifacts/runs/agentic/tool_selection_development_v2/trials/calibration_wave'


def saved_original():
    return load_reference(TRIAL/'common_reference'),audit.read(TRIAL/'manifest.json')


def test_actual_saved_raw_rejects_and_projection_matches_recorded_first_feed(tmp_path):
    original,manifest=saved_original()
    with pytest.raises(ValueError,match='unsupported target metadata fields'):
        assemble_reference_inputs(original,0.)
    projected,disclosure=project_reference(original,manifest)
    save_reference(projected,tmp_path/'projected')
    loaded=load_reference(tmp_path/'projected')
    assert audit.projection_relation(original,loaded,manifest)
    with np.load(TRIAL/'policy_inputs.npz',allow_pickle=False) as feeds,np.load(TRIAL/'reference_cache.npz',allow_pickle=False) as cache:
        anchor=audit.read(TRIAL/'motion_reference_draft.json')['body_names'].index('torso_link')
        heading=audit._legacy._state.compute_yaw_offset_np(feeds['call_0001__current_anchor_rot'][0],cache['committed__body_rot'][0,anchor])
        expected=assemble_reference_inputs(loaded,0.,heading_offset_xyzw=heading)
        for key,value in expected.items():
            tolerance=2e-7 if key.endswith('anchor_rot') else 0
            audit.equal(value,feeds['call_0001__'+key],'actual first physics feed '+key,tolerance)
    assert disclosure['original_strict_g1_admission'] is False


def test_original_roundtrip_and_full200_projection_feed_parity(tmp_path):
    original,manifest=saved_original();projected,_=project_reference(original,manifest)
    save_reference(projected,tmp_path/'projection')
    proposal=audit.read(TRIAL/'proposal.json')
    result=audit.common_parity(TRIAL,proposal,tmp_path/'raw_roundtrip',tmp_path/'projection')
    assert result['decisions']==200 and result['raw_array_bytes_unchanged'] is True
    assert result['original_strict_g1_admission'] is False
    assert result['compatible_projection_admitted'] is True


def test_unknown_field_retained_and_strict_adapter_still_rejects():
    original,manifest=saved_original();projected,_=project_reference(original,manifest)
    raw=replace(original,metadata={**_plain(original.metadata),'unknown_target_semantics':'forbidden'})
    candidate=replace(projected,metadata={**_plain(projected.metadata),'unknown_target_semantics':'forbidden'})
    with pytest.raises(ValueError,match='unsupported target metadata fields'):
        audit.projection_relation(raw,candidate,manifest)


def test_projection_must_not_drop_nested_producer_or_change_arrays():
    original,manifest=saved_original();projected,_=project_reference(original,manifest)
    metadata=_plain(projected.metadata);metadata['producer'].pop('control_tool')
    with pytest.raises(ValueError,match='only two'):
        audit.projection_relation(original,replace(projected,metadata=metadata),manifest)
    changed=projected.joint_position_rad.copy();changed[0,0]+=.00001
    with pytest.raises(ValueError,match='projection array bytes'):
        audit.projection_relation(original,replace(projected,joint_position_rad=changed),manifest)


def test_all_original_numeric_history_control_and_selector_checks_preserved():
    import ast
    first=ast.parse((ROOT/'experiments/agentic/audit_tool_selection_v2.py').read_text(encoding='utf8'))
    second=ast.parse((ROOT/'experiments/agentic/audit_tool_selection_projection.py').read_text(encoding='utf8'))
    a={n.name:ast.dump(n,include_attributes=False) for n in first.body if isinstance(n,ast.FunctionDef)}
    b={n.name:ast.dump(n,include_attributes=False) for n in second.body if isinstance(n,ast.FunctionDef)}
    for name in ('check_waveform','check_model_receipt','check_control_provenance','check_quarantined_predecessor',
        'check_development_scope','calibration_measurements','check_calibration_view','check_observation','check_record','check_summary_tools'):
        assert a[name]==b[name],name
