"""Posthoc reference-curve domain tests; no physics or complete-run analysis."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_agentic_reference import fixture

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'experiments/agentic/diagnose_tool_reference_projection.py'
spec=importlib.util.spec_from_file_location('tool_reference_diagnostics_projection_test',PATH)
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


def reference():
    _,old=fixture()
    times=np.arange(208)*.02
    return SimpleNamespace(time_offsets_s=times,joint_position_rad=np.zeros((208,29)),
        joint_velocity_rad_s=np.zeros((208,29)),metadata={'body':{'joint_names':old.joint_names}})


def test_executed_domain_excludes_later_lookahead_extrema():
    ref=reference();ref.joint_position_rad[201:,0]=100
    report=diag.reference_diagnostics(ref)
    executed=report['executed_domain']['all29'];support=report['full_reference_support']['all29']
    assert executed['knot_count']==201 and executed['actual_knot_domain_s']==[0.,4.]
    assert support['knot_count']==208 and support['actual_knot_domain_s']==[0.,4.14]
    assert executed['global_max_abs_velocity_rad_s']==0
    assert support['global_max_abs_velocity_rad_s']>0
    assert report['executed_domain']['arms14']['joint_count']==14
    assert executed['joint_count']==29 and executed['units']['acceleration']=='rad/s^2'


def test_existing_endpoint_required_never_interpolates_clipped_curve():
    ref=reference()
    for name in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s'):
        setattr(ref,name,np.delete(getattr(ref,name),200,axis=0))
    with pytest.raises(ValueError,match='existing4s knot; no clipping'):
        diag.reference_diagnostics(ref)


def test_reference_support_endpoint_distinguished_from_execution():
    ref=reference();ref.time_offsets_s[-1]=4.15
    with pytest.raises(ValueError,match='lookahead support'):
        diag.reference_diagnostics(ref)


def test_interior_speed_is_not_just_supplied_knot_dq():
    ref=reference();ref.joint_position_rad[1:,15]=1.
    report=diag.reference_diagnostics(ref)['executed_domain']
    assert report['all29']['sampled_knot_max_abs_velocity_rad_s']==0
    assert report['all29']['global_max_abs_velocity_rad_s']==pytest.approx(75.)
    assert report['arms14']['global_max_abs_velocity_rad_s']==pytest.approx(75.)
    assert report['arms14']['global_max_abs_velocity_joint_name'].startswith('left_shoulder_')
    assert report['all29']['global_max_abs_velocity_time_s']==pytest.approx(.01)


@pytest.mark.parametrize('start,end',[(-.01,4.),(0.,3.99),(0.,4.000001)])
def test_outside_exact_existing_execution_domain_rejected(start,end):
    ref=reference();ref.time_offsets_s[0]=start;ref.time_offsets_s[200]=end
    with pytest.raises(ValueError):diag.executed_knots(ref)


def test_no_realized_state_or_pd_hermite_assumption():
    text=PATH.read_text(encoding='utf8')
    assert 'qpos.npy' not in text and 'qvel.npy' not in text and 'applied_joint_targets.npy' not in text
    tree=ast.parse(text)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='hermite_derivative_extrema']
    assert len(calls)==1
    assert [ast.unparse(arg) for arg in calls[0].args]==['t','q','dq']
    assert 'scores_or_gates_changed=False' in text and 'training_admission=False' in text


def test_unaccepted_audit_rejected_before_output(tmp_path):
    pilot=tmp_path/'pilot';pilot.mkdir()
    for name in ('comparison.json','protocol.json'):(pilot/name).write_text('{}')
    audit=tmp_path/'audit.json';audit.write_text('{"schema":"wrong","audit_passed":false}')
    output=tmp_path/'analysis'
    with pytest.raises(ValueError,match='passed final matching'):
        diag.diagnose(pilot,tmp_path/'projection',audit,output)
    assert not output.exists()


def test_old_v1_final_audit_cannot_authorize_v2_analysis(tmp_path):
    pilot=tmp_path/'pilot';pilot.mkdir()
    for name in ('comparison.json','protocol.json'):(pilot/name).write_text('{}')
    audit=tmp_path/'audit.json';audit.write_text('{"schema":"pulsemorph.agentic_tool_selection_final_audit.v1","audit_passed":true}')
    output=tmp_path/'analysis'
    with pytest.raises(ValueError,match='passed final matching'):
        diag.diagnose(pilot,tmp_path/'projection',audit,output)
    assert not output.exists()


def test_v2_dependency_capture_explicit_and_v1_wrapper_untouched():
    text=PATH.read_text(encoding='utf8')
    for filename in ('audit_tool_selection_v2.py','tool_selection_driver_v2.py','tool_selection_tasks_v2.py','tool_selection_batch_v2.py'):
        assert filename in text
    assert 'pulsemorph.agentic_tool_selection_protocol.v2' in text
    assert 'pulsemorph.agentic_tool_selection_projection_audit.v1' in text
    old=ROOT/'experiments/agentic/diagnose_tool_reference.py'
    assert diag.sha(old)=='f29f82574ca37067e94f8fbd53e3df2156521abec604c0456a215ae6382cb649'


def test_projected_saved_arrays_exact_and_changed_payload_rejected(tmp_path):
    from pulsemorph.motion.contracts import load_reference,save_reference
    from dataclasses import replace
    source=ROOT/'artifacts/runs/agentic/tool_selection_development_v2/trials/calibration_wave/common_reference'
    original=load_reference(source)
    # This test exercises array identity only; no acceptance of forged metadata.
    save_reference(original,tmp_path/'same')
    assert diag.same_reference_arrays(source,tmp_path/'same')
    changed=original.joint_velocity_rad_s.copy();changed[1,1]+=.001
    save_reference(replace(original,joint_velocity_rad_s=changed),tmp_path/'changed')
    with pytest.raises(ValueError,match='payload changed'):
        diag.same_reference_arrays(source,tmp_path/'changed')


def test_projection_specific_admission_and_reference_paths_explicit():
    text=PATH.read_text(encoding='utf8')
    for field in ('original_strict_g1_admission=False','compatible_projection_required=True',
        'original_reference_directory','projection_reference_directory','matching original full inventory',
        'matching projection full inventory','projection_index_sha256'):
        assert field in text
