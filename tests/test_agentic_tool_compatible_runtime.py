"""Real package strict admission tests; no ONNX session, mj_step or download."""
import ast
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pulsemorph.agentic.common_player import materialize_reference
from pulsemorph.motion.contracts import save_reference, load_reference
from pulsemorph.motion.g1 import profile_from_yaml, assemble_reference_inputs

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'experiments/agentic/execute_tool_selection_compatible.py'
spec=importlib.util.spec_from_file_location('compatible_runtime_test',PATH)
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
TOOLS=('global_clock_v1','local_clock_1p4_2p8_v1')


def package(tmp_path,tool):
    profile=profile_from_yaml()
    t=np.arange(151)/30
    raw=dict(fps=30,dof_pos=np.sin(t[:,None]*np.arange(1,30)[None,:]/10)*.1,
        rigid_body_pos=np.zeros((len(t),len(profile['body_names']),3)),
        rigid_body_rot=np.broadcast_to([0.,0.,0.,1.],(len(t),len(profile['body_names']),4)).copy())
    p=dict(control_tool=tool,method='llm_feedback',parameters=dict(left_gain_a=.8,right_gain_a=1.2,
        left_gain_b=1.2,right_gain_b=.8,timing_shift_s=-.3))
    generator=runtime.build_generator(p,raw,joint_names=profile['joint_names'],body_names=profile['body_names'],
        source_start_s=0.,control_steps=200,section_windows_s=[[.4,2],[2,3.6]],source_provenance={'sha256':'a'*64})
    ref=materialize_reference(generator,profile=profile,candidate_id='test',observation_id='test',task_id='test',
        runtime_session_id='test',source_path='fixture.motion',source_sha256='a'*64,audio_observed_through_sample=88200,
        audio_duration_s=4.,protocol_sha256='b'*64,producer_kind='agentic')
    producer=runtime.proposal_producer(p);producer['control_tool']=tool
    ref=runtime.compatible_metadata(ref,producer)
    save_reference(ref,tmp_path/'common')
    return load_reference(tmp_path/'common'),p


@pytest.mark.parametrize('tool',TOOLS)
def test_built_saved_loaded_package_passes_real_strict_g1(tmp_path,tool):
    loaded,p=package(tmp_path,tool)
    assert set(loaded.metadata)=={'schema_version','reference_id','producer','body','controller','frame',
        'context','validity','observation','provenance'}
    assert loaded.metadata['producer']['control_tool']==tool
    assert loaded.metadata['producer']['model']=='gpt-6.1-sol'
    assert runtime.strict_g1_preflight(loaded)==2
    for offset in [0.,3.98]:
        actual=assemble_reference_inputs(loaded,offset)
        assert actual['mimic_future_dof_pos'].shape==(1,4,29)
        assert actual['mimic_future_dof_vel'].shape==(1,4,29)
        assert actual['mimic_future_anchor_rot'].shape==(1,4,4)


@pytest.mark.parametrize('bad', [{'control_tool':TOOLS[0]}, {'experiment_kind':'tool_selection_v1'}])
def test_legacy_extra_top_fields_rejected_before_mock_controller(tmp_path,bad):
    loaded,_=package(tmp_path,TOOLS[0])
    loaded=replace(loaded,metadata={**loaded.metadata,**bad})
    loaded.validate()  # This weaker check is exactly why the old runtime missed it.
    calls=[]
    def controller():calls.append('controller')
    with pytest.raises(ValueError,match='unsupported target metadata'):
        runtime.strict_g1_preflight(loaded)
        controller()
    assert calls==[]


def test_strict_preflight_rejects_profile_mismatch(tmp_path):
    loaded,_=package(tmp_path,TOOLS[1])
    metadata={**loaded.metadata,'controller':{**loaded.metadata['controller'],'profile_sha256':'c'*64}}
    with pytest.raises(ValueError,match='identity'):
        runtime.strict_g1_preflight(replace(loaded,metadata=metadata))


def test_runtime_preflight_actually_precedes_controller_and_session_construction():
    tree=ast.parse(PATH.read_text(encoding='utf8'))
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_trial')
    calls=[n for n in ast.walk(execute) if isinstance(n,ast.Call)]
    line=lambda text:next(n.lineno for n in calls if ast.unparse(n.func)==text)
    assert line('load_reference') < line('strict_g1_preflight') < line('CommonReferencePlayer') < line('upstream.run')
    assert line('strict_g1_preflight') < line('real_session')
    text=PATH.read_text(encoding='utf8')
    assert "'strict_g1_admission_passed': False" in text
    assert "manifest['strict_g1_admission_passed'] = True" in text
    assert "manifest['strict_g1_preflight_count'] = strict_g1_preflight(loaded)" in text
    assert 'sources = [Path(__file__)' in text


@pytest.mark.parametrize('tool',TOOLS)
@pytest.mark.parametrize('method',['calibration','surrogate_search','llm_feedback','llm_feedback_off'])
def test_producer_v2_keeps_method_and_tool_context(tool,method):
    p={'method':method,'control_tool':tool}
    producer=runtime.proposal_producer(p)
    assert producer['version']=='2'
    assert producer['id']=='tool-selection-compatible-'+method
    assert producer['kind']==('agentic' if method.startswith('llm_') else 'baseline')
    assert producer.get('model')==('gpt-6.1-sol' if method.startswith('llm_') else None)


def test_existing_frozen_runtime_is_not_imported_or_overwritten():
    text=PATH.read_text(encoding='utf8')
    tree=ast.parse(text)
    imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('execute_tool_selection' in n for n in imports)
    assert 'compatibility_replay' in text and 'new_llm_calls=0' in text
