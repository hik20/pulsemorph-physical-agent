"""Independent saved hybrid control audit, optional capped FK, no NN/steps.

Raw graph targets remain distinct from applied hybrid targets. The legacy array
validator receives an explicitly transformed COPY solely to audit actual PD and
history continuity; this does not assert graph-PD identity on hybrid arm joints.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('_legacy_episode_array_checks',Path(__file__).with_name('audit_episode_v1.py'))
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def inventory(path):
    rows={}
    for p in path.rglob('*'):
        require(not p.is_symlink() and not(getattr(p.lstat(),'st_file_attributes',0)&1024),'Linked input forbidden')
        if p.is_file():rows[p.relative_to(path).as_posix()]=sha(p)
    return rows

def check_hybrid_controls(rows,raw_outputs,applied,feeds,config):
    n=len(applied);arms=config['arm_indices'];nonarms=config['nonarm_indices']
    require(type(arms) is list and len(arms)==14 and all(type(i) is int for i in arms) and len(set(arms))==14 and all(0<=i<29 for i in arms),'Exact arm map')
    require(nonarms==[i for i in range(29) if i not in arms],'Exact nonarm complement')
    expected_shapes={k:(n,29) for k in ('q_ref','dq_ref','qfrc_bias_joint','kp','kd','raw_graph_PD','applied_PD')}
    expected_shapes.update(qpos_at_control=(n,36),qvel_at_control=(n,35))
    require(set(rows)==set(expected_shapes) and all(rows[k].shape==shape and np.isfinite(rows[k]).all() for k,shape in expected_shapes.items()),'Hybrid array shape/finite')
    require(np.all(rows['kp']>0) and np.all(rows['kd']>=0),'Positive kp/nonnegative kd')
    for i in range(n):
        graph=raw_outputs[f'call_{i+1:04d}__joint_pos_targets'][0]
        require(graph.dtype==np.float32 and np.array_equal(rows['raw_graph_PD'][i],graph),'Raw graph output changed')
        require(np.array_equal(rows['applied_PD'][i],applied[i]) and applied.dtype==np.float32,'Actual applied target binding')
        require(np.array_equal(applied[i,nonarms],graph[nonarms]),'Nonarm graph PD changed')
        expected=(rows['q_ref'][i,arms]+rows['kd'][i,arms]/rows['kp'][i,arms]*rows['dq_ref'][i,arms]+rows['qfrc_bias_joint'][i,arms]/rows['kp'][i,arms]).astype(np.float32)
        require(np.array_equal(applied[i,arms],expected),'Independent arm PD formula differs')
        previous=np.zeros((1,1,29),np.float32) if i==0 else applied[i-1][None,None,:]
        require(np.array_equal(feeds[f'call_{i+1:04d}__historical_processed_actions'],previous),'Applied hybrid history differs')
    return True

def check_force_peaks(telemetry,config,n):
    require(type(telemetry) is list and len(telemetry)==n,'Force telemetry count')
    model=config['compiled_model'];joint_ids=np.asarray(model['actuator_trnid'])[:,0].astype(int)
    limits=np.asarray(model['jnt_actfrcrange']);enabled=np.asarray(model['jnt_actfrclimited'])
    alimits=np.asarray(model['actuator_forcerange']);aenabled=np.asarray(model['actuator_forcelimited'])
    for i,row in enumerate(telemetry):
        require(row['control_index']==i and row['steps_completed_in_chunk_control']==20 and row['joint_caps_observed_compliant'] is True,'Completed force observation scope')
        for key,indices,ranges,flags in [('peak_abs_joint_qfrc_actuator',joint_ids,limits,enabled),('peak_abs_individual_actuator_force',np.arange(29),alimits,aenabled)]:
            peaks=np.asarray(row[key]);require(peaks.shape==(29,) and np.isfinite(peaks).all() and np.all(peaks>=0),'Finite nonnegative force peaks')
            for k,j in enumerate(indices):
                if flags[j]:
                    lo,hi=ranges[j];require(abs(lo+hi)<1e-12 and hi>=0,'Absolute-only peak cannot certify asymmetric caps')
                    require(peaks[k]<=hi+1e-5,'Recorded force peak exceeds preserved cap')
    return True

def audit(directory,*,max_fk=0):
    directory=Path(directory).resolve();before=inventory(directory)
    backend=json.loads((directory/'backend.json').read_bytes());episode=json.loads((directory/'episode.json').read_bytes());config=json.loads((directory/'hybrid_config.json').read_bytes())
    require(backend['schema']=='pulsemorph.embodied_hybrid_backend.v1' and config['schema']=='pulsemorph.embodied_hybrid_arm_config.v1','New hybrid schemas required')
    for name in ('hybrid_config','hybrid_control','hybrid_force_telemetry'):
        suffix='.npz' if name=='hybrid_control' else '.json';require(sha(directory/(name+suffix))==backend[name+'_sha256'],'Hybrid metadata hash binding')
    require(type(backend['source_pins']) is dict and bool(backend['source_pins']),'Nonempty source pins')
    for p,h in backend['source_pins'].items():require(sha(p)==h,'Pinned source changed')
    require(backend['initialization_count']==episode['initialization_count']==backend['mjdata_instances']==backend['onnx_sessions']==backend['warmup_calls']==1,'Persistent instance/initialization counts')
    load=lambda name:np.load(directory/(name+'.npy'),allow_pickle=False)
    q,v,t,applied=map(load,('qpos','qvel','state_times','applied_joint_targets'))
    with np.load(directory/'policy_inputs.npz',allow_pickle=False) as z:feeds=dict(z)
    with np.load(directory/'policy_outputs.npz',allow_pickle=False) as z:outputs=dict(z)
    with np.load(directory/'hybrid_control.npz',allow_pickle=False) as z:rows=dict(z)
    check_hybrid_controls(rows,outputs,applied,feeds,config)
    copied={k:a.copy() for k,a in outputs.items()}
    for i,value in enumerate(applied):copied[f'call_{i+1:04d}__joint_pos_targets']=value[None].copy()
    n=legacy.audit_arrays(q,v,t,applied,feeds,copied,backend['calls'],load('substep_telemetry'))
    require(np.array_equal(rows['qpos_at_control'],q[:-1]) and np.array_equal(rows['qvel_at_control'],v[:-1]),'Boundary actual q/v differs')
    require(backend['online_calls']==episode['control_steps']==n and backend['physics_steps']==n*20 and backend['forward_calls']==n+1,'Hybrid execution counts')
    integ=load('mj_integration_states');require(integ.ndim==2 and len(integ)==n+1 and np.isfinite(integ).all(),'Saved integration state rows')
    telemetry=json.loads((directory/'hybrid_force_telemetry.json').read_bytes());check_force_peaks(telemetry,config,n)
    gate=config.get('limited_physical_gate')
    require(type(gate) is dict and gate.get('min_root_height_m')==.6 and gate.get('min_root_up_z')==.7,'Declared limited physical gate differs')
    quaternions=q[:,3:7]
    require(np.allclose(np.linalg.norm(quaternions,axis=1),1,rtol=0,atol=1e-6),'Actual root quaternion normalization')
    sampled_up=1-2*(quaternions[:,1]**2+quaternions[:,2]**2)
    require(np.min(sampled_up)>=.7-1e-9 and backend.get('min_root_up_z',-1)>=.7,'Recorded/sample root-up limited screen')
    from pulsemorph.motion.contracts import load_reference
    from pulsemorph.motion.g1 import assemble_reference_inputs,sample_reference,profile_from_yaml
    from pulsemorph.agentic.app.phrase.assets import MODEL
    profile=profile_from_yaml(ROOT/(MODEL[:-5]+'.yaml'));joint_names=list(profile['joint_names'])
    arms=[i for i,name in enumerate(joint_names) if name.startswith(('left_','right_')) and any(p in name for p in ('shoulder_','elbow_','wrist_'))]
    require(config['arm_indices']==arms and config['arm_joint_names']==[joint_names[i] for i in arms],'Actual named arm ordering')
    cursor=0
    for j,action in enumerate(episode['actions']):
        require(action['status']=='completed' and abs(action['start_sim_time_s']-cursor*.02)<1e-9,'Action boundary time')
        ref=load_reference(directory/f'reference_{j:02d}')
        require(ref.metadata['producer']['id']=='agentic-embodied-joint-tool' and ref.metadata['context']['runtime_session_id']==episode['session_id'],'Causal reference producer/session')
        for local in range(action['planned_control_steps']):
            strict=assemble_reference_inputs(ref,local*.02);prefix=f'call_{cursor+1:04d}__'
            for key in ('mimic_future_dof_pos','mimic_future_dof_vel','mimic_future_anchor_rot'):require(np.allclose(strict[key],feeds[prefix+key],rtol=0,atol=2e-6),'Action-local strict common feed')
            sample=sample_reference(ref,np.array([(local+1)*.02]))
            require(np.array_equal(rows['q_ref'][cursor],sample['dof_pos'][0]) and np.array_equal(rows['dq_ref'][cursor],sample['dof_vel'][0]),'Hybrid first+20ms q/dq reference differs')
            cursor+=1
    require(cursor==n,'Action coverage')
    fk=0;bias_max_error=0.
    if max_fk:
        require(type(max_fk) is int and n+1<=max_fk<=501,'Explicit saved-state FK cap required')
        sys.path.insert(0,str(ROOT/'research/vendor/ProtoMotions'))
        import mujoco,yaml
        from deployment import test_tracker_mujoco as upstream
        cfg=yaml.safe_load((ROOT/(MODEL[:-5]+'.yaml')).read_text())
        doc=ET.fromstring(upstream._patch_mjcf_xml(Path(profile['body_path'])));compiler=doc.find('compiler')
        for key in ('meshdir','texturedir'):
            if compiler is not None and compiler.get(key):compiler.set(key,str((Path(profile['body_path']).parent/compiler.get(key)).resolve()))
        xml=ET.tostring(doc,encoding='unicode');require((directory/'render_model.xml').read_text()==xml,'Pinned body patch differs')
        model=mujoco.MjModel.from_xml_string(xml);model.opt.timestep=.001;model.jnt_stiffness[:]=0;model.dof_damping[:]=0;model.dof_frictionloss[:]=0
        kp=np.asarray(cfg['control']['stiffness']);kd=np.asarray(cfg['control']['damping'])
        for i in range(29):model.actuator_gainprm[i,0]=kp[i];model.actuator_biastype[i]=1;model.actuator_biasprm[i,:3]=[0,-kp[i],-kd[i]];model.actuator_ctrllimited[i]=0
        for name,values in config['compiled_model'].items():require(np.array_equal(np.asarray(getattr(model,name)),np.asarray(values)),'Preserved compiled gains/transmission/forcecaps differ')
        require(np.array_equal(rows['kp'],np.tile(kp,(n,1))) and np.array_equal(rows['kd'],np.tile(kd,(n,1))),'Recorded PD gains differ')
        addresses=[int(model.jnt_dofadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]) for name in joint_names]
        require(addresses==list(range(6,35)),'Actual joint DOF mapping')
        data=mujoco.MjData(model);anchor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
        for i in range(n+1):
            data.qpos[:]=q[i];data.qvel[:]=v[i];data.time=t[i];mujoco.mj_forward(model,data);fk+=1
            if i<n:
                require(np.allclose(data.xquat[anchor][[1,2,3,0]][None],feeds[f'call_{i+1:04d}__current_anchor_rot'],rtol=0,atol=1e-6),'Fresh current torso FK')
                error=float(np.max(np.abs(data.qfrc_bias[addresses]-rows['qfrc_bias_joint'][i])));bias_max_error=max(bias_max_error,error)
                require(error<=1e-9,'Fresh actual same-boundary bias differs')
    require(inventory(directory)==before,'Raw hybrid files changed')
    for p,h in backend['source_pins'].items():require(sha(p)==h,'Pinned source changed during audit')
    return {'schema':'pulsemorph.embodied_hybrid_saved_audit.v1','passed':True,'online_controls':n,'policy_calls':n+1,'recorded_physics_steps':n*20,'fresh_FK':fk,'compiled_model_reconstruction_verified':bool(fk),'bias_max_abs_error':bias_max_error if fk else None,'new_NN':0,'new_physics_steps':0,'input_inventory':before,'qualification':'Original raw graph outputs remain separate. Arm PD formula and actual-applied history audited; no graph-PD identity claim for arms.','limits':['Force peaks are recorded per-control maxima, not independent full-substep torque replay.','Optional FK checks actual state bias, not contact force compensation or general balance.','No resume API or arbitrary reaching capability demonstrated.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-fk',type=int,default=0);a=p.parse_args();require(a.max_fk>=0,'Nonnegative FK authority')
    result=audit(a.directory,max_fk=a.max_fk)
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
