"""Read saved persistent episodes; optional explicitly capped FK, never policy/steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from pulsemorph.agentic.app.phrase.assets import INPUTS, OUTPUTS

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inventory(path): return {p.relative_to(path).as_posix():sha(p) for p in path.rglob('*') if p.is_file()}
def require(value, message):
    if not value: raise ValueError(message)

def audit_arrays(q, dq, times, targets, feeds, outputs, calls, substeps):
    n=len(targets)
    require(n>0 and q.shape==(n+1,36) and dq.shape==(n+1,35) and times.shape==(n+1,) and targets.shape==(n,29),'State/target counts')
    require(all(np.isfinite(a).all() for a in (q,dq,times,targets,substeps)), 'Nonfinite saved data')
    require(np.allclose(times,np.arange(n+1)*.02,rtol=0,atol=1e-9),'Persistent time continuity')
    require(len(calls)==n+1 and calls[0]['role']=='warmup' and all(c['role']=='control' and c['status']=='completed' for c in calls[1:]),'One warmup then continuous control calls')
    require(substeps.shape==(n*20,6) and np.array_equal(substeps[:,0],np.arange(1,n*20+1)) and np.allclose(substeps[:,1],np.arange(1,n*20+1)*.001,rtol=0,atol=1e-9),'Substep clock/count continuity')
    require(set(feeds)=={f'call_{i:04d}__{k}' for i in range(n+1) for k in INPUTS} and set(outputs)=={f'call_{i:04d}__{k}' for i in range(n+1) for k in OUTPUTS},'Exact8feed4output inventory')
    for i in range(n+1):
        prefix=f'call_{i:04d}__'
        require(calls[i]['index']==i and calls[i]['status']=='completed' and abs(calls[i]['sim_time_s']-(max(0,i-1)*.02))<1e-9,'Call identity/time')
        for k in INPUTS: require(feeds[prefix+k].dtype==np.float32 and np.isfinite(feeds[prefix+k]).all(),'Feed dtype/finite')
        for k in OUTPUTS: require(outputs[prefix+k].shape==(1,29) and outputs[prefix+k].dtype==np.float32 and np.isfinite(outputs[prefix+k]).all(),'Output shape/dtype/finite')
        if i==0:
            require(all(np.count_nonzero(feeds[prefix+k])==0 for k in INPUTS),'Warmup dummy feed differs')
            continue
        require(np.array_equal(feeds[prefix+'current_dof_pos'],q[i-1,7:].astype('float32')[None]) and np.array_equal(feeds[prefix+'current_dof_vel'],dq[i-1,6:].astype('float32')[None]),'Actual current joint state')
        require(np.array_equal(feeds[prefix+'current_root_local_ang_vel'],dq[i-1,3:6].astype('float32')[None]),'Actual angular velocity')
        previous=np.zeros((1,1,29),np.float32) if i==1 else outputs[f'call_{i-1:04d}__joint_pos_targets'][:,None,:]
        require(np.array_equal(feeds[prefix+'historical_processed_actions'],previous),'Processed history reset/mismatch')
        require(np.array_equal(targets[i-1],outputs[prefix+'joint_pos_targets'][0]),'PD target/output mismatch')
    return n

def audit(directory, *, max_fk=0):
    directory=Path(directory).resolve();before=inventory(directory)
    backend=json.loads((directory/'backend.json').read_bytes());episode=json.loads((directory/'episode.json').read_bytes())
    for p,h in backend['source_pins'].items():require(sha(p)==h,'Pinned source changed')
    require(backend['initialization_count']==episode['initialization_count']==backend['mjdata_instances']==backend['onnx_sessions']==backend['warmup_calls']==1,'Persistent instance/initialization counts')
    load=lambda name:np.load(directory/(name+'.npy'),allow_pickle=False)
    q,dq,t,targets=map(load,('qpos','qvel','state_times','applied_joint_targets'))
    with np.load(directory/'policy_inputs.npz',allow_pickle=False) as z: feeds=dict(z)
    with np.load(directory/'policy_outputs.npz',allow_pickle=False) as z: outputs=dict(z)
    n=audit_arrays(q,dq,t,targets,feeds,outputs,backend['calls'],load('substep_telemetry'))
    require(backend['online_calls']==episode['control_steps']==n and backend['physics_steps']==n*20 and backend['forward_calls']==n+1,'Backend scope counts')
    integration=load('mj_integration_states');require(integration.ndim==2 and len(integration)==n+1 and np.isfinite(integration).all(),'Integration-state continuity records')
    from pulsemorph.motion.contracts import load_reference
    from pulsemorph.motion.g1 import assemble_reference_inputs
    cursor=0
    for j,action in enumerate(episode['actions']):
        require(action['status']=='completed' and abs(action['start_sim_time_s']-cursor*.02)<1e-9,'Action boundary/reset mismatch')
        ref=load_reference(directory/f'reference_{j:02d}')
        require(ref.metadata['producer']['id']=='agentic-embodied-joint-tool' and ref.metadata['context']['runtime_session_id']==episode['session_id'],'Original causal reference identity')
        for local in range(action['planned_control_steps']):
            strict=assemble_reference_inputs(ref,local*.02);prefix=f'call_{cursor+1:04d}__'
            for key in ('mimic_future_dof_pos','mimic_future_dof_vel','mimic_future_anchor_rot'):require(np.allclose(strict[key],feeds[prefix+key],rtol=0,atol=2e-6),'Strict action-local future feed/alignment')
            cursor+=1
    require(cursor==n,'Action coverage')
    fk=0
    if max_fk:
        require(type(max_fk) is int and max_fk>=n+1,'Explicit FK cap must cover all saved states')
        sys.path.insert(0,str(ROOT/'research/vendor/ProtoMotions'))
        import mujoco
        from deployment import test_tracker_mujoco as upstream
        from pulsemorph.motion.g1 import profile_from_yaml
        from pulsemorph.agentic.app.phrase.assets import MODEL
        profile=profile_from_yaml(ROOT/(MODEL[:-5]+'.yaml'))
        import xml.etree.ElementTree as ET
        doc=ET.fromstring(upstream._patch_mjcf_xml(Path(profile['body_path'])))
        compiler=doc.find('compiler')
        for key in ('meshdir','texturedir'):
            if compiler is not None and compiler.get(key):compiler.set(key,str((Path(profile['body_path']).parent/compiler.get(key)).resolve()))
        patched=ET.tostring(doc,encoding='unicode')
        require((directory/'render_model.xml').read_text(encoding='utf8')==patched,'Actual patched body XML differs')
        model=mujoco.MjModel.from_xml_string(patched)
        data=mujoco.MjData(model);anchor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
        for i in range(n+1):
            data.qpos[:]=q[i];data.qvel[:]=dq[i];data.time=t[i];mujoco.mj_forward(model,data);fk+=1
            if i<n:require(np.allclose(data.xquat[anchor][[1,2,3,0]][None],feeds[f'call_{i+1:04d}__current_anchor_rot'],rtol=0,atol=1e-6),'Fresh current torso FK')
            if i==n:
                for name,pose in episode['final_observation']['body_poses'].items():
                    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
                    require(body>=0 and np.allclose(data.xpos[body],pose['position_world_m'],rtol=0,atol=1e-9),'Final actual body position FK')
    require(inventory(directory)==before,'Raw episode changed during audit')
    for p,h in backend['source_pins'].items():require(sha(p)==h,'Pinned source changed during audit')
    return {'schema':'pulsemorph.embodied_saved_audit.v1','passed':True,'online_controls':n,'policy_calls':n+1,'physics_steps_recorded':n*20,'fresh_FK':fk,'new_NN':0,'new_physics_steps':0,'input_inventory':before,'limitations':['Integration states recorded, not a demonstrated resume API.','Substeps contain scalar telemetry, not independently replayed full joint states.','Wrist Jacobians not independently verified by this audit.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--max-fk',type=int,default=0);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.max_fk<0:raise ValueError('Nonnegative explicit FK cap required')
    result=audit(args.directory,max_fk=args.max_fk)
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
