"""Two fresh-reset compatibility replays; zero model calls, no optimization.

Freeze protocol before execution. Original 100-trial comparison stays immutable.
This helper is intentionally not a retry/resume driver: output must be new.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from musical_tasks import ROOT, read, sha256, write_new_json
from execute_tool_selection_compatible import execute_trial
from evaluate_generalization import _audit
from audit_multifeature import common_parity, inventory

RUNTIME='experiments/agentic/execute_tool_selection_compatible.py'
OLD_RUNTIME='experiments/agentic/execute_tool_selection.py'
CASES=(('wave_a','global_clock_v1'),('high5_b','local_clock_1p4_2p8_v1'))


def require(value,message):
    if not value:raise ValueError(message)


def ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha256(path)}


def array_archive_parity(first,second):
    with np.load(first,allow_pickle=False) as a,np.load(second,allow_pickle=False) as b:
        require(set(a.files)==set(b.files),'archive key mismatch')
        for key in a.files:
            require(a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes(),
                    'archive dtype/shape/bytes mismatch: '+key)
    return True


def trace_parity(old,new):
    for name in ('qpos.npy','qvel.npy','applied_joint_targets.npy','state_times.npy'):
        require((old/name).read_bytes()==(new/name).read_bytes(),'trace byte mismatch: '+name)
    for name in ('policy_inputs.npz','policy_outputs.npz','reference_cache.npz','common_reference/arrays.npz'):
        array_archive_parity(old/name,new/name)
    return {'npy_byte_parity':True,'npz_dtype_shape_payload_byte_parity':True,
        'common_all_four_arrays_identical':True,'metadata_identity_expected_to_change':True}


def input_pin_parity(old_manifest,new_manifest):
    a=old_manifest['preexecution_input_hashes'];b=new_manifest['preexecution_input_hashes']
    require(set(a)-{OLD_RUNTIME}==set(b)-{RUNTIME},'unexpected preexecution pin path difference')
    require(OLD_RUNTIME in a and RUNTIME not in a and RUNTIME in b and OLD_RUNTIME not in b,
            'declared executor switch missing')
    for key in set(a)-{OLD_RUNTIME}:
        require(a[key]==b[key],'changed shared preexecution pin: '+key)
    require(b[RUNTIME]==sha256(ROOT/RUNTIME),'compatible runtime pin mismatch')
    for key in ('onnx_sha256','motion_sha256','body_profile_sha256','controller_profile_sha256',
                'checkpoint_sha256','physics_contract','versions'):
        require(old_manifest[key]==new_manifest[key],'controller/body/source/environment mismatch: '+key)
    return {'only_declared_executor_switch':True,'removed':{OLD_RUNTIME:a[OLD_RUNTIME]},'added':{RUNTIME:b[RUNTIME]}}


def initialize(pilot,output):
    pilot,output=Path(pilot).resolve(),Path(output).resolve()
    allowed=(ROOT/'artifacts/runs/agentic').resolve()
    require(allowed in output.parents and output!=pilot and pilot not in output.parents,'new separate agentic output required')
    comparison=read(pilot/'comparison.json');original_protocol=read(pilot/'protocol.json')
    require(comparison['comparison_valid'] is True and comparison['physical_trials']==100,'completed original100 comparison required')
    require(original_protocol['schema']=='pulsemorph.agentic_tool_selection_protocol.v2','original v2 protocol required')
    require(comparison['protocol_sha256']==sha256(pilot/'protocol.json'),'original comparison protocol changed')
    selected=[]
    for task,tool in CASES:
        group=next(t for t in comparison['tasks'] if t['task_id']==task)
        winner=next(m for m in group['methods'] if m['method']=='llm_feedback')['selected']
        require(winner['control_tool']==tool and winner['intent_passed'] is True,'required selected LLM tool/success missing')
        identity=winner['candidate_id'];proposal_path=pilot/f'proposals/{identity}.json'
        evaluation_path=pilot/f'evaluations/{identity}.json';trial=pilot/'trials'/identity
        proposal=read(proposal_path);evaluation=read(evaluation_path);manifest=read(trial/'manifest.json')
        require(evaluation==winner,'selected evaluation differs from comparison')
        require(proposal['method']=='llm_feedback' and proposal['control_tool']==tool and proposal['task_id']==task,'selected proposal mismatch')
        require(sha256(trial/'proposal.json')==sha256(proposal_path),'original executed proposal differs')
        require(evaluation['execution_manifest_sha256']==sha256(trial/'manifest.json'),'original selected execution changed')
        selected.append({'task_id':task,'control_tool':tool,'original_candidate_id':identity,
            'proposal':ref(proposal_path),'evaluation':ref(evaluation_path),'manifest':ref(trial/'manifest.json')})
    # Capture complete original evidence; no files in pilot are written by this helper.
    before=inventory(pilot)
    deps=set(original_protocol['execution_code_sha256'])-{OLD_RUNTIME}
    deps.update((RUNTIME,'experiments/agentic/run_tool_compatibility_smoke.py',
        'experiments/agentic/audit_multifeature.py'))
    pins={name:sha256(ROOT/name) for name in sorted(deps)}
    for name,digest in original_protocol['execution_code_sha256'].items():
        require(sha256(ROOT/name)==digest,'original frozen execution code changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    for name,digest in pins.items():
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target);require(sha256(target)==digest,'helper source changed during snapshot')
    protocol={'schema':'pulsemorph.agentic_tool_compatibility_smoke_protocol.v1',
        'original_comparison':ref(pilot/'comparison.json'),'original_protocol':ref(pilot/'protocol.json'),
        'selected_cases':selected,'execution_code_sha256':pins,
        'max_physical_trials':2,'control_steps_per_trial':200,'physics_steps_per_trial':4000,
        'new_llm_calls':0,'no_optimization':True,'retry_allowed':False,'fresh_reset_each_trial':True,
        'selection_rule':'Existing selected LLM wave_a/global and high5_b/local; exact original method/tool/parameters/source/task.',
        'scope':'Compatibility replays only. No new original100 score, confirmation robustness, training admission or independent new-task claim.',
        'metadata_changes':['producer id/version updated; tool nested in producer only',
            'common unsupported top-level control_tool and experiment_kind absent',
            'reference_id/context namespaced to smoke protocol; source/body/controller/arrays identical'],
        'declared_preexecution_pin_switch':{'old_executor':OLD_RUNTIME,'compatible_executor':RUNTIME}}
    write_new_json(output/'original_inventory_before.json',before)
    write_new_json(output/'protocol.json',protocol)
    for item in selected:
        proposal=copy.deepcopy(read(item['proposal']['path']))
        candidate='compatibility_smoke__'+item['task_id']
        proposal.update(candidate_id=candidate,observation_id=candidate+':observation',
            runtime_session_id=candidate+':fresh-reset',protocol_sha256=sha256(output/'protocol.json'),
            execution_kind='compatibility_replay',replay_of=item['proposal'])
        # Old model artifacts remain protected by the original proposal link;
        # they are not fresh smoke observations or responses.
        proposal['observation_artifact']=None;proposal['response_artifact']=None
        write_new_json(output/f'proposals/{candidate}.json',proposal)
    return protocol,before


def run(pilot,output):
    pilot,output=Path(pilot).resolve(),Path(output).resolve()
    protocol,before=initialize(pilot,output)
    summary={'schema':'pulsemorph.agentic_tool_compatibility_smoke.v1','smoke_passed':False,
        'errors':[],'cases':[],'physical_trials':0,'control_steps':0,'physics_steps':0,'new_llm_calls':0,
        'max_physical_trials':2,'original_scope_not_rescored':True,'training_admission':False}
    try:
        for item in protocol['selected_cases']:
            candidate='compatibility_smoke__'+item['task_id'];new=output/'trials'/candidate
            proposal_path=output/f'proposals/{candidate}.json'
            manifest=execute_trial(proposal_path,new)
            require(manifest['status']=='success' and manifest.get('strict_g1_admission_passed') is True
                and manifest.get('strict_g1_preflight_count')==2,'compatible runtime failed strict admission/execution')
            require((manifest['actual_control_steps'],manifest['actual_physics_steps'])==(200,4000),'smoke execution count mismatch')
            old=pilot/'trials'/item['original_candidate_id'];old_manifest=read(old/'manifest.json')
            parity=trace_parity(old,new);pins=input_pin_parity(old_manifest,manifest)
            audit=_audit(new)  # standalone source/trace/current-FK audit, never old calibration normalization
            common=common_parity(new,read(proposal_path),output/'roundtrip_packages'/candidate)
            height=float(np.min(audit['sub'][:,2]));rmse=audit['mean_joint_rmse_rad']
            gate=height>=.6 and rmse<=.35
            require(gate,'new smoke physical gate failed')
            original=read(item['evaluation']['path'])
            summary['cases'].append({'candidate_id':candidate,'original_candidate_id':item['original_candidate_id'],
                'control_tool':item['control_tool'],'manifest':ref(new/'manifest.json'),
                'trace_parity':parity,'input_pin_parity':pins,'strict_g1_preflight_count':2,
                'strict_common_all200_parity':common,'fresh_anchor_audit':audit.get('fresh_anchor_audit'),
                'standalone_integrity_passed':True,'new_physical_gate_passed':gate,
                'min_root_height_m':height,'mean_joint_rmse_rad':rmse,
                'metrics_reused_by_trace_identity':{'original_evaluation':item['evaluation'],
                    'features':original['features'],'intent_passed':original['intent_passed'],
                    'utility':original['utility'],'not_new_metric_recompute':True}})
        summary['smoke_passed']=True
    except Exception as exc:
        summary['errors'].append(str(exc))
    finally:
        manifests=[read(p/'manifest.json') for p in (output/'trials').glob('*') if (p/'manifest.json').is_file()]
        summary.update(physical_trials=len(manifests),control_steps=sum(m.get('actual_control_steps',0) for m in manifests),
            physics_steps=sum(m.get('actual_physics_steps',0) for m in manifests))
        after=inventory(pilot)
        write_new_json(output/'original_inventory_after.json',after)
        summary['original_evidence_unchanged']=before==after
        summary['source_pins_unchanged']=all(sha256(ROOT/name)==digest for name,digest in protocol['execution_code_sha256'].items())
        if not summary['original_evidence_unchanged'] or not summary['source_pins_unchanged']:
            summary['smoke_passed']=False;summary['errors'].append('source or original evidence changed')
        write_new_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=run(args.pilot,args.output)
    print(json.dumps({key:result[key] for key in ('smoke_passed','errors','physical_trials','control_steps','physics_steps','new_llm_calls')}))
    raise SystemExit(0 if result['smoke_passed'] else 1)
