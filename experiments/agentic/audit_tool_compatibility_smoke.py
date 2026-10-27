"""Read-only recovery audit of existing compatibility traces; never physics."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np

from musical_tasks import ROOT,read,sha256 as _sha256,write_new_json
from run_tool_compatibility_smoke import trace_parity,ref,require,RUNTIME,OLD_RUNTIME
from evaluate_generalization import _audit
from audit_multifeature import common_parity,inventory


def sha256(path):
    return _sha256(Path(path))


def normalized_pins(pins):
    result={}
    for path,digest in pins.items():
        key=path.replace('\\','/')
        require(key not in result,'normalized input pin collision: '+key)
        result[key]=digest
    return result


def input_pin_parity(old_manifest,new_manifest):
    a=normalized_pins(old_manifest['preexecution_input_hashes']);b=normalized_pins(new_manifest['preexecution_input_hashes'])
    require(set(a)-{OLD_RUNTIME}==set(b)-{RUNTIME},'unexpected normalized preexecution pin difference')
    require(OLD_RUNTIME in a and RUNTIME not in a and RUNTIME in b and OLD_RUNTIME not in b,'declared executor switch missing')
    require(all(a[k]==b[k] for k in set(a)-{OLD_RUNTIME}),'changed normalized shared input pin')
    require(b[RUNTIME]==sha256(ROOT/RUNTIME),'compatible executor changed')
    for key in ('onnx_sha256','motion_sha256','body_profile_sha256','controller_profile_sha256','checkpoint_sha256','physics_contract','versions'):
        require(old_manifest[key]==new_manifest[key],'body/controller/source/environment changed: '+key)
    return dict(only_declared_executor_switch=True,windows_path_keys_normalized=True,
        removed={OLD_RUNTIME:a[OLD_RUNTIME]},added={RUNTIME:b[RUNTIME]})


def audit_case(pilot,smoke,item,output):
    identity='compatibility_smoke__'+item['task_id'];new=smoke/'trials'/identity;old=pilot/'trials'/item['original_candidate_id']
    manifest=read(new/'manifest.json');proposal=read(smoke/f'proposals/{identity}.json')
    require(manifest['status']=='success' and manifest.get('strict_g1_admission_passed') is True
        and manifest.get('strict_g1_preflight_count')==2,'strict-compatible execution incomplete')
    require((manifest['actual_control_steps'],manifest['actual_physics_steps'])==(200,4000),'runtime count mismatch')
    for name in ('proposal','evaluation','manifest'):
        link=item[name];require(sha256(link['path'])==link['sha256'],'original linked evidence changed: '+name)
    require(sha256(new/'proposal.json')==sha256(smoke/f'proposals/{identity}.json'),'executed replay proposal mismatch')
    original=read(item['proposal']['path'])
    for key in ('method','control_tool','parameters','source_motion','task_id'):
        require(proposal[key]==original[key],'replay changed original '+key)
    require(proposal['replay_of']==item['proposal'] and proposal['execution_kind']=='compatibility_replay','replay lineage missing')
    require(proposal['protocol_sha256']==sha256(smoke/'protocol.json'),'replay protocol mismatch')
    parity=trace_parity(old,new)
    pins=input_pin_parity(read(old/'manifest.json'),manifest)
    independent=_audit(new)
    common=common_parity(new,proposal,output/'roundtrip_packages'/identity)
    height=float(np.min(independent['sub'][:,2]));rmse=independent['mean_joint_rmse_rad']
    require(height>=.6 and rmse<=.35,'fresh replay physical gate failed')
    require(independent.get('fresh_anchor_audit') is not None,'fresh FK audit absent')
    metrics=read(item['evaluation']['path'])
    return dict(candidate_id=identity,original_candidate_id=item['original_candidate_id'],trace_parity=parity,
        input_pin_parity=pins,strict_common_all200_parity=common,fresh_anchor_audit=independent['fresh_anchor_audit'],
        standalone_integrity_passed=True,new_physical_gate_passed=True,min_root_height_m=height,mean_joint_rmse_rad=rmse,
        strict_g1_preflight_count=2,physical_trials=1,control_steps=200,physics_steps=4000,new_llm_calls=0,
        metrics_reused_by_trace_identity={'original_evaluation':item['evaluation'],'features':metrics['features'],
            'intent_passed':metrics['intent_passed'],'utility':metrics['utility'],'not_new_metric_recompute':True})


def audit(pilot,smoke,output):
    pilot,smoke,output=map(lambda p:Path(p).resolve(),(pilot,smoke,output))
    require(output!=smoke and smoke not in output.parents and output!=pilot and pilot not in output.parents,'separate audit output required')
    before=inventory(smoke);original_before=inventory(pilot);protocol=read(smoke/'protocol.json')
    original_inventory=read(smoke/'original_inventory_before.json')
    require(original_before==original_inventory,'original evidence changed since smoke freeze')
    require(read(smoke/'original_inventory_after.json')==original_inventory,'smoke recorded original mutation')
    require(sha256(protocol['original_comparison']['path'])==protocol['original_comparison']['sha256'],'original comparison changed')
    require(sha256(protocol['original_protocol']['path'])==protocol['original_protocol']['sha256'],'original protocol changed')
    for name,digest in protocol['execution_code_sha256'].items():
        require(sha256(ROOT/name)==digest and sha256(smoke/'source_snapshot'/name)==digest,'frozen smoke source changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    own=ROOT/'experiments/agentic/audit_tool_compatibility_smoke.py';own_digest=sha256(own)
    shutil.copyfile(own,output/'audit_source.py')
    result=dict(schema='pulsemorph.agentic_tool_compatibility_recovery_audit.v1',audit_passed=False,errors=[],cases=[],
        input_summary=ref(smoke/'summary.json'),smoke_protocol=ref(smoke/'protocol.json'),new_physical_trials=0,new_llm_calls=0,
        scope='Read-only verifier correction; failed original summary remains unchanged. No rerun or optimization.')
    try:
        for item in protocol['selected_cases']:
            identity='compatibility_smoke__'+item['task_id']
            if (smoke/'trials'/identity/'manifest.json').is_file():
                result['cases'].append(audit_case(pilot,smoke,item,output))
        require(len(result['cases'])==1 and result['cases'][0]['candidate_id']=='compatibility_smoke__wave_a',
            'recovery scope requires exactly the existing global wave_a replay')
        result['audit_passed']=True
    except Exception as exc:result['errors'].append(str(exc))
    finally:
        result.update(smoke_evidence_unchanged=inventory(smoke)==before,original_evidence_unchanged=inventory(pilot)==original_before,
            auditor_source_unchanged=sha256(own)==own_digest)
        if not all(result[k] for k in ('smoke_evidence_unchanged','original_evidence_unchanged','auditor_source_unchanged')):
            result['audit_passed']=False;result['errors'].append('evidence/source mutation during read-only audit')
        write_new_json(output/'audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True,type=Path);p.add_argument('--smoke',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();r=audit(args.pilot,args.smoke,args.output)
    print(json.dumps({k:r[k] for k in ('audit_passed','errors','new_physical_trials','new_llm_calls')}))
    raise SystemExit(0 if r['audit_passed'] else 1)
