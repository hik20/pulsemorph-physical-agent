"""Read-only final evidence audit. No simulation steps, model calls or corpus import."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))
from evaluate_multifeature import evaluate_multifeature, next_jacobian_candidate, _legacy
from pulsemorph.motion.contracts import load_reference, save_reference
from pulsemorph.motion.g1 import sample_reference, assemble_reference_inputs


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def inventory(directory):
    return {str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
def linked(pilot,item):
    require(isinstance(item,dict),'missing linked artifact')
    path=(pilot/item['path']).resolve()
    require(pilot in path.parents and path.is_file() and sha(path)==item['sha256'],'linked artifact hash/path mismatch')
    return path

def equal(a,b,label,atol=0):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=0,atol=atol),label+' mismatch')

def compact(row):
    keys=['candidate_id','parameters','integrity_passed','integrity_errors','physical_gate_passed','intent_passed',
          'features','feature_errors','per_dimension_pass','utility','utility_l2','mean_joint_rmse_rad','min_root_height_m','wall_seconds']
    result={k:row.get(k) for k in keys}
    if 'accent' in row:
        result['accent']={k:row['accent'][k] for k in ['available','peak_time_s','peak_rms_speed_rad_s','event_times_s','event_count','window_event_count']}
    return result


def common_parity(trial,proposal,output):
    reference=load_reference(trial/'common_reference')
    save_reference(reference,output)
    roundtrip=load_reference(output)
    for field in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw'):
        equal(getattr(reference,field),getattr(roundtrip,field),'save/load '+field)
    require(reference.metadata==roundtrip.metadata,'save/load metadata mismatch')
    context=reference.metadata['context']
    for key in ('observation_id','task_id'):
        require(context[key]==proposal[key],'common context '+key)
    require(context['decision_id']==proposal['candidate_id']+':offline','common decision identity')
    require(context['protocol_sha256']==proposal['protocol_sha256'],'common protocol identity')
    draft=read(trial/'motion_reference_draft.json');anchor=draft['body_names'].index('torso_link')
    with np.load(trial/'reference_cache.npz',allow_pickle=False) as archive:cache=dict(archive)
    times=np.arange(201)*.02
    requested=sample_reference(reference,times)
    for name,key in [('dof_pos','committed__dof_pos'),('dof_vel','committed__dof_vel')]:
        equal(requested[name].astype('float32'),cache[key],'committed '+name)
    equal(requested['anchor_rot'].astype('float32'),cache['committed__body_rot'][:,anchor],'committed torso',2e-7)
    with np.load(trial/'policy_inputs.npz',allow_pickle=False) as archive:feeds=dict(archive)
    heading=_legacy._state.compute_yaw_offset_np(feeds['call_0001__current_anchor_rot'][0],cache['committed__body_rot'][0,anchor])
    max_error=0.
    for i in range(200):
        offsets=i*.02+np.array([.02,.04,.08,.16])
        raw=sample_reference(reference,offsets)
        for name,key in [('dof_pos','forecast__dof_pos'),('dof_vel','forecast__dof_vel')]:
            equal(raw[name].astype('float32'),cache[key][i],'four-future cache '+name)
        equal(raw['anchor_rot'].astype('float32'),cache['forecast__body_rot'][i,:,anchor],'four-future cache torso',2e-7)
        inputs=assemble_reference_inputs(reference,i*.02,heading_offset_xyzw=heading)
        for key,value in inputs.items():
            actual=feeds[f'call_{i+1:04d}__'+key]
            tolerance=2e-7 if key.endswith('anchor_rot') else 0
            equal(value,actual,'runtime common '+key,tolerance)
            max_error=max(max_error,float(np.max(np.abs(value-actual))))
    return dict(save_load_passed=True,committed_samples=201,decisions=200,future_samples=800,
                runtime_reference_input_max_abs_error=max_error)


def proposal_provenance(pilot,proposal,protocol,task,recomputed):
    method=proposal['method'];index=proposal['trial_index'];candidate=proposal['candidate_id']
    parameters=[proposal['parameters'][name] for name in protocol['parameter_names']]
    bounds=np.asarray(protocol['bounds'])
    require(len(parameters)==5 and np.isfinite(parameters).all() and np.all(np.asarray(parameters)>=bounds[:,0]) and np.all(np.asarray(parameters)<=bounds[:,1]),'parameter domain mismatch')
    if method=='calibration':return dict(normalization='not applicable')
    require(candidate==f'{task["task_id"]}__{method}__{index}','candidate identity mismatch')
    require(proposal['task_id']==task['task_id'],'proposal task mismatch')
    if index==0:require(parameters==protocol['anchor'],'paired anchor parameters mismatch')
    observation_path=linked(pilot,proposal['observation_artifact']);obs=read(observation_path)
    require(obs['task_id']==task['task_id'] and obs['method']==method and obs['canonical_goal']==task['goal'],'observation identity/goal mismatch')
    revision=0 if method=='llm_feedback_off' else index
    require(obs['observation_id']==f'{task["task_id"]}__{method}__{revision}','observation revision identity mismatch')
    history=obs.get('prior_trial_results',[])
    require(len(history)==revision,'observation history length mismatch')
    for j,row in enumerate(history):
        previous=f'{task["task_id"]}__{method}__{j}'
        require(row==compact(recomputed[previous]),'observation history differs from independent outcome: '+previous)
    if method=='jacobian_search':
        rows=[recomputed[f'{task["task_id"]}__{method}__{j}'] for j in range(index)]
        equal(parameters,next_jacobian_candidate(rows,protocol,task['goal']),'deterministic optimizer parameters')
        return dict(normalization='not applicable',history_rows=revision)
    require(obs['protocol_sha256']==sha(pilot/'protocol.json') and obs['trial_index']==revision,'LLM observation protocol/revision mismatch')
    response_path=linked(pilot,proposal['response_artifact']);raw=read(response_path)
    require(Path(raw['only_observation_path']).resolve()==observation_path,'response observation path identity')
    values=raw['parameters']
    flat=isinstance(values,list) and len(values)==5 and all(type(v) in (int,float) for v in values)
    normalized=[values] if flat else values
    disclosure='wrapped_single_flat_vector' if flat else 'none'
    # Earlier nested off-stream responses required no repair and predate this
    # optional field. A flat response still requires the explicit repair label.
    require(proposal.get('response_parameter_layout_normalization','none')==disclosure,'normalization disclosure mismatch')
    require(len(normalized)==(8 if method=='llm_feedback_off' else 1),'response proposal count mismatch')
    equal(parameters,normalized[index if method=='llm_feedback_off' else 0],'raw-response parameters unchanged')
    return dict(normalization=disclosure,parameter_values_unchanged=True,history_rows=revision)


def audit(pilot,output):
    pilot,output=Path(pilot).resolve(),Path(output).resolve()
    require(output!=pilot and pilot not in output.parents,'audit output must be outside original pilot')
    output.mkdir(parents=True,exist_ok=False)
    source_files=[Path(__file__)]+[Path(__file__).with_name(name) for name in
        ('evaluate_multifeature.py','evaluate_trial.py','execute_multifeature.py','multifeature_driver.py','musical_tasks.py')]
    source_files += [ROOT/'src/pulsemorph/agentic'/name for name in
                     ('reference.py','multifeature_reference.py','common_player.py')]
    source_files+=sorted((ROOT/'src/pulsemorph/motion').glob('*.py'))
    before_sources={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    for path in source_files:
        dest=output/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    before=inventory(pilot)
    report=dict(schema='pulsemorph.agentic_multifeature_final_audit.v1',audit_passed=False,errors=[],
                scope='read-only evidence recomputation; no physics steps or model calls; no corpus import',
                limitations=['Audit success does not imply every goal succeeded, aesthetics, human preference, hardware safety or generalization.',
                             'UTC proposal and filesystem creation timestamps are local provenance evidence, not signed external attestations.'],
                source_hashes_before=before_sources,trials={},selections=[])
    try:
        protocol=read(pilot/'protocol.json');comparison=read(pilot/'comparison.json')
        require(comparison['protocol_sha256']==sha(pilot/'protocol.json'),'comparison protocol hash mismatch')
        require(len(protocol['tasks'])==2 and len(protocol['methods'])==3,'expected two tasks/three methods')
        ids=['calibration_0']+[f'{t["task_id"]}__{method}__{i}' for t in protocol['tasks'] for method in protocol['methods'] for i in range(8)]
        require({p.name for p in (pilot/'trials').iterdir() if p.is_dir()}==set(ids),'exact49 trial directory inventory mismatch')
        require(len(ids)==49,'trial budget mismatch')
        # Every off-stream commitment precedes the earliest actual candidate creation.
        candidate_start=min((pilot/'trials'/cid/'proposal.json').stat().st_ctime_ns for cid in ids[1:])
        candidate_start_utc=datetime.fromtimestamp(candidate_start/1e9).astimezone()
        off=[]
        for task in protocol['tasks']:
            for i in range(8):
                path=pilot/'proposals'/f'{task["task_id"]}__llm_feedback_off__{i}.json';p=read(path)
                require(datetime.fromisoformat(p['created_at_utc'])<candidate_start_utc,'feedback-off committed after candidate start')
                require(path.stat().st_ctime_ns<candidate_start,'feedback-off filesystem commitment after candidate start')
                off.append(p['candidate_id'])
        report['feedback_off_precommitment']=dict(candidates=off,earliest_candidate_creation_ns=candidate_start,
            ordering_basis='proposal UTC plus filesystem proposal creation vs first executed proposal-copy creation')
        recomputed={};controls=physics=0
        baseline_manifest=read(pilot/'trials/calibration_0/manifest.json')
        for name,digest in before_sources.items():
            if name in baseline_manifest['preexecution_input_hashes']:
                require(digest==baseline_manifest['preexecution_input_hashes'][name],'current audit dependency differs from trial-pinned source: '+name)
        for count,cid in enumerate(ids,1):
            trial=pilot/'trials'/cid;proposal=read(pilot/'proposals'/f'{cid}.json');manifest=read(trial/'manifest.json')
            require(sha(trial/'proposal.json')==sha(pilot/'proposals'/f'{cid}.json'),'copied proposal mismatch '+cid)
            require(proposal['candidate_id']==cid and proposal['protocol_sha256']==sha(pilot/'protocol.json'),'proposal identity/protocol mismatch')
            task=next((t for t in protocol['tasks'] if t['task_id']==proposal['task_id']),protocol['tasks'][0])
            result=evaluate_multifeature(trial,pilot/'trials/calibration_0',protocol,task['goal'])
            require(result['integrity_passed'],cid+': '+str(result['integrity_errors']))
            require(result.get('fresh_anchor_audit') is not None,'fresh FK audit missing '+cid)
            for key in ('preexecution_input_hashes','versions','onnx_sha256','motion_sha256','physics_contract'):
                require(manifest[key]==baseline_manifest[key],'source/controller/runtime differs '+cid+':'+key)
            result.update(candidate_id=cid,parameters=[proposal['parameters'][n] for n in protocol['parameter_names']],wall_seconds=manifest['wall_seconds'])
            result['provenance']=proposal_provenance(pilot,proposal,protocol,task,recomputed)
            result['common_parity']=common_parity(trial,proposal,output/'roundtrip_packages'/cid)
            recorded=read(pilot/'evaluations'/f'{cid}.json')
            for key in ('features','feature_errors','per_dimension_pass','intent_passed','physical_gate_passed','utility','utility_l2','mean_joint_rmse_rad','min_root_height_m'):
                require(recorded.get(key)==result.get(key),'stored evaluation mismatch '+cid+':'+key)
            recomputed[cid]=result
            report['trials'][cid]=result
            controls+=manifest['actual_control_steps'];physics+=manifest['actual_physics_steps']
            if count%10==0:print(f'audited {count}/49 trials',flush=True)
        require((controls,physics)==(9800,196000),'aggregate actual execution counts mismatch')
        require((comparison['physical_trials'],comparison['control_steps'],comparison['physics_steps'])==(49,controls,physics),'comparison execution count mismatch')
        report['actual_counts']=dict(physical_trials=49,control_steps=controls,physics_steps=physics)
        anchors=[cid for cid in ids[1:] if cid.endswith('__0')]
        for cid in anchors[1:]:
            for filename in ('qpos.npy','qvel.npy','applied_joint_targets.npy'):
                require((pilot/'trials'/cid/filename).read_bytes()==(pilot/'trials'/anchors[0]/filename).read_bytes(),'paired full-anchor byte inequality '+cid+':'+filename)
        report['paired_anchor_repeats']=dict(streams=anchors,qpos_qvel_pd_entire_trajectory_bitwise_equal=True)
        for task in protocol['tasks']:
            for method in protocol['methods']:
                rows=[recomputed[f'{task["task_id"]}__{method}__{i}'] for i in range(8)]
                eligible=[r for r in rows if r['integrity_passed'] and r['physical_gate_passed'] and r['utility'] is not None]
                selected=min(eligible,key=lambda r:(r['utility'],r['utility_l2'],r['candidate_id'])) if eligible else None
                stored_task=next(t for t in comparison['tasks'] if t['task_id']==task['task_id'])
                stored=next(m for m in stored_task['methods'] if m['method']==method)['selected']
                require((stored['candidate_id'] if stored else None)==(selected['candidate_id'] if selected else None),'common selector mismatch')
                successes=[i for i,r in enumerate(rows) if r['physical_gate_passed'] and all(r['per_dimension_pass'])]
                report['selections'].append(dict(task_id=task['task_id'],method=method,selected_candidate=selected['candidate_id'] if selected else None,
                    per_dimension_pass=selected['per_dimension_pass'] if selected else None,physical_gate_passed=selected['physical_gate_passed'] if selected else False,
                    full_goal_success=bool(selected and selected['intent_passed']),first_full_success_trial_index=successes[0] if successes else None,
                    first_success_scope='descriptive supplementary; all eight budgeted trials retained'))
        protected=read(pilot/'protected_hashes_before.json')
        require(bool(protected),'protected freeze inventory missing')
        report['protected_hashes']={name:sha(ROOT/name)==digest for name,digest in protected.items()}
        require(all(report['protected_hashes'].values()),'protected frozen artifact changed')
        report['all_selected_goals_successful']=all(row['full_goal_success'] for row in report['selections'])
        report['audit_passed']=True
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    finally:
        report['source_hashes_after']={str(p.relative_to(ROOT)):sha(p) for p in source_files}
        report['audit_sources_unchanged']=report['source_hashes_after']==before_sources
        report['original_evidence_unchanged']=inventory(pilot)==before
        if not report['audit_sources_unchanged'] or not report['original_evidence_unchanged']:
            report['audit_passed']=False;report['errors'].append('source or original evidence changed during audit')
        (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'audit_passed':report['audit_passed'],'errors':report['errors'],'output':str(output/'audit.json')}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=audit(args.pilot,args.output);sys.exit(0 if result['audit_passed'] else 1)
