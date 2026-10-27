"""App phrase worker: one bounded physical trial and saved-state audit.

Numerical controller is pinned upstream.run, unmodified. No pickle/history read.
Legacy adapter/gate code is derived from frozen adapter_program_v1.py; only
new cache/job plumbing and independent saved-state audit are added.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback,site,importlib.metadata
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions')]
import numpy as np
from scipy.spatial.transform import Rotation
from pulsemorph.motion.g1 import sample_reference,assemble_reference_inputs,profile_from_yaml
from pulsemorph.motion.contracts import load_reference
from pulsemorph.agentic.app.phrase.assets import PhraseAssets,sha,write_new,MODEL,PUBLIC_PYTHON,INPUTS,OUTPUTS

class PhrasePlayer:
    """Only source frame0 is a full carrier; future body_rot is auxiliary layout.

    Non-torso future rotations repeat authentic first-source frame0 solely to satisfy
    the unmodified legacy 33-body layout. They are derived static auxiliaries,
    not composed body rotations/FK/contact/root targets, and are never exported
    as motion evidence. Exact-eight-feed guard prevents their consumption.
    """
    total_frames=400
    control_dt=.02

    def __init__(self,reference,initial_carrier,anchor_index):
        reference.validate()
        if reference.time_offsets_s[-1]!=8.14 or reference.metadata['controller']['control_dt_s']!=.02 or reference.metadata['producer']['id']!='phrase-program-reference' or reference.metadata['producer']['version']!='1':
            raise ValueError('exact8.14s G1 phrase required')
        self.reference=reference
        self.initial={k:np.asarray(v).copy() for k,v in initial_carrier.items()}
        self.anchor=anchor_index
        if self.initial['body_pos'].shape!=(33,3) or self.initial['body_rot'].shape!=(33,4) or not all(np.isfinite(v).all() for v in self.initial.values()):
            raise ValueError('actual Wave initial body carrier required')
        first=sample_reference(reference,np.array([0.]))
        if not np.array_equal(first['dof_pos'][0].astype('float32'),self.initial['dof_pos']) or not np.array_equal(first['anchor_rot'][0].astype('float32'),self.initial['body_rot'][anchor_index]):
            raise ValueError('phrase initial q/torso differs from authentic first-source carrier')

    def get_state_at_frame(self,index):
        if type(index) is not int or not 0<=index<=400:raise ValueError('frame outside committed0..400')
        if index==0:return {k:v.copy() for k,v in self.initial.items()}
        q=sample_reference(self.reference,np.array([index*.02]))
        return {'dof_pos':q['dof_pos'][0].astype('float32'),'dof_vel':q['dof_vel'][0].astype('float32')}

    def get_future_references(self,index,steps):
        if type(index) is not int or not 0<=index<400 or list(steps)!=[1,2,4,8]:
            raise ValueError('only400control exact1/2/4/8 horizon supported')
        # Preserve legacy generator operation order: (index+offset)*control_dt.
        s=sample_reference(self.reference,np.array([(index+i)*.02 for i in steps]))
        rotations=np.broadcast_to(self.initial['body_rot'],(4,33,4)).copy()
        rotations[:,self.anchor]=s['anchor_rot'].astype('float32')
        return {'dof_pos':s['dof_pos'].astype('float32'),'dof_vel':s['dof_vel'].astype('float32'),'body_rot':rotations}


def same(actual,expected,label):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or actual.tobytes()!=expected.tobytes():
        raise ValueError('exact prefix bridge mismatch: '+label)


class PrefixGate:
    """Checks every warmup/online prefix row before permitting next work."""
    def __init__(self,inputs,outputs,qpos,qvel,times,targets):
        self.inputs,self.outputs=inputs,outputs
        self.qpos,self.qvel,self.times,self.targets=qpos,qvel,times,targets
        self.inputs_checked=self.outputs_checked=self.states_checked=self.targets_checked=0
        self.passed=False

    def feed(self,call,feed):
        if call>18:
            if not self.passed:raise ValueError('prefix boundary not passed before decision18')
            return
        if tuple(feed)!=INPUTS:raise ValueError('exact eight-feed interface required')
        prefix=f'call_{call:04d}__'
        for k,v in feed.items():same(v,self.inputs[prefix+k],'input '+str(call)+' '+k)
        self.inputs_checked+=1

    def output(self,call,names,values):
        if tuple(names)!=OUTPUTS or len(values)!=4:raise ValueError('exact four-output interface required')
        if call<=18:
            for n,v in zip(names,values):same(v,self.outputs[f'call_{call:04d}__'+n],'output '+str(call)+' '+n)
            self.outputs_checked+=1

    def state(self,index,qpos,qvel,time):
        if index<=18:
            same(qpos,self.qpos[index],'qpos '+str(index));same(qvel,self.qvel[index],'qvel '+str(index))
            if float(time)!=float(self.times[index]):raise ValueError('prefix time differs')
            self.states_checked+=1
        if index==18:
            if (self.inputs_checked,self.outputs_checked,self.states_checked,self.targets_checked)!=(19,19,19,18):
                raise ValueError('incomplete mandatory prefix boundary')
            self.passed=True

    def target(self,index,target):
        if index<=17:
            same(target,self.targets[index],'appliedPD '+str(index));self.targets_checked+=1


def interface_guard(session):
    ins=[(n.name,n.shape,n.type) for n in session.get_inputs()]
    shapes=[['batch_size',4],['batch_size',29],['batch_size',29],['batch_size',3],['batch_size',1,29],['batch_size',4,4],['batch_size',4,29],['batch_size',4,29]]
    if ins!=[(n,s,'tensor(float)') for n,s in zip(INPUTS,shapes)] or [(n.name,n.shape,n.type) for n in session.get_outputs()]!=[(n,['batch_size',29],'tensor(float)') for n in OUTPUTS]:
        raise ValueError('graph consumes unsupported feed/carrier or output layout')


def call_budget(index,prefix_passed):
    if type(index) is not int or not 0<=index<401:raise RuntimeError('401 total calls exhausted')
    if index>=19 and not prefix_passed:raise RuntimeError('mandatory prefix gate absent')


def step_budget(completed):
    if type(completed) is not int or not 0<=completed<8000:raise RuntimeError('8000step budget exhausted')


def environment():
    import mujoco,onnxruntime,numpy.core._multiarray_umath as nb
    import onnxruntime.capi.onnxruntime_pybind11_state as ob
    import mujoco._functions as mf,mujoco._structs as ms
    if Path(sys.executable).resolve()!=PUBLIC_PYTHON.resolve() or not sys.flags.isolated or site.ENABLE_USER_SITE is not False:raise RuntimeError('exact public isolated Python required')
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','mujoco','onnxruntime','torch')}
    if versions!={'numpy':'1.26.4','scipy':'1.16.3','mujoco':'3.3.7','onnxruntime':'1.31.0','torch':'2.7.1'}:raise RuntimeError('phrase numerical versions changed')
    modules={}
    for m in (np,mujoco,onnxruntime,nb,ob,mf,ms):
        path=Path(m.__file__).resolve()
        if not path.is_relative_to(PUBLIC_PYTHON.resolve().parents[1]):raise RuntimeError('module outside isolated runtime')
        modules[str(path)]=sha(path)
    for path in Path(mujoco.__file__).parent.glob('mujoco*.dll'):modules[str(path.resolve())]=sha(path)
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Python3.11 required')
    return {'versions':versions,'python':sys.version,'python_sha256':sha(sys.executable),'isolated':True,'user_site':False,'module_pins':modules}


def verify(pins):
    for p,h in pins.items():
        if sha(p)!=h:raise RuntimeError('phrase trial input changed: '+p)


def make_gate(assets):
    g=assets.arrays('golden_prefix')
    return PrefixGate({k[8:]:v for k,v in g.items() if k.startswith('inputs__')},{k[9:]:v for k,v in g.items() if k.startswith('outputs__')},g['qpos'],g['qvel'],g['time'],g['PD'])


def preflight(assets,reference):
    player=PhrasePlayer(reference,assets.arrays('initial_carrier'),16)
    for i in range(400):assemble_reference_inputs(reference,i*.02)
    from deployment.tracker_inputs import init_tracker_alignment,align_future_motion_refs,build_tracker_onnx_inputs
    import yaml
    cfg=yaml.safe_load((ROOT/MODEL).with_suffix('.yaml').read_text());gold=assets.arrays('golden_prefix');carrier=player.initial
    expected_reset=np.r_[carrier['body_pos'][0],carrier['body_rot'][0][[3,0,1,2]],carrier['dof_pos']].astype('float64')
    same(expected_reset,gold['qpos'][0],'authentic reset qpos')
    if np.any(gold['qvel'][0]):raise RuntimeError('authentic reset qvel is not zero')
    alignment=init_tracker_alignment(robot_anchor_pos=carrier['body_pos'][16],robot_anchor_rot=gold['inputs__call_0001__current_anchor_rot'][0],motion_anchor_pos=carrier['body_pos'][16],motion_anchor_rot=carrier['body_rot'][16])
    for i in range(18):
        prefix=f'inputs__call_{i+1:04d}__'
        feed=build_tracker_onnx_inputs(onnx_input_names=list(INPUTS),onnx_name_to_key=cfg['_runtime']['onnx_name_to_in_key'],dof_pos=gold[prefix+'current_dof_pos'][0],dof_vel=gold[prefix+'current_dof_vel'][0],anchor_rot=gold[prefix+'current_anchor_rot'][0],root_local_ang_vel=gold[prefix+'current_root_local_ang_vel'][0],future_refs=align_future_motion_refs(player.get_future_references(i,[1,2,4,8]),alignment),anchor_body_index=16,num_dofs=29,prev_actions=gold[prefix+'historical_processed_actions'][0,0])
        for name in INPUTS:same(feed[name],gold[prefix+name],'static18feed '+name)
    return player


def audit_saved(directory,assets,result):
    """All400 saved controls/FK/reference/history/PD/time before metrics."""
    import mujoco,yaml,xml.etree.ElementTree as ET
    from deployment.state_utils import compute_yaw_offset_np,apply_heading_offset_np
    before={p.relative_to(directory).as_posix():sha(p) for p in directory.rglob('*') if p.is_file() and p.name not in ('worker.log','result.json')}
    q=np.load(directory/'qpos.npy',allow_pickle=False);v=np.load(directory/'qvel.npy',allow_pickle=False);t=np.load(directory/'state_times.npy',allow_pickle=False);pd=np.load(directory/'applied_joint_targets.npy',allow_pickle=False)
    ref=load_reference(directory/'common_reference')
    with np.load(directory/'policy_inputs.npz',allow_pickle=False) as z:ins={k:z[k].copy() for k in z.files}
    with np.load(directory/'policy_outputs.npz',allow_pickle=False) as z:outs={k:z[k].copy() for k in z.files}
    calls=json.loads((directory/'policy_calls.json').read_text())
    if q.shape!=(401,36) or v.shape!=(401,35) or t.shape!=(401,) or pd.shape!=(400,29) or len(ins)!=3208 or len(outs)!=1604 or len(calls)!=401:raise RuntimeError('saved phrase count/layout mismatch')
    if len(list((directory/'attempts').glob('call_*.json')))!=401:raise RuntimeError('durable attempt budget mismatch')
    if not all(np.isfinite(a).all() for a in (q,v,t,pd)) or not np.allclose(t,np.arange(401)*.02,rtol=0,atol=1e-10):raise RuntimeError('saved state finite/time failure')
    shapes=((1,4),(1,29),(1,29),(1,3),(1,1,29),(1,4,4),(1,4,29),(1,4,29))
    gate=make_gate(assets);gate.state(0,q[0],v[0],t[0])
    for c in range(401):
        row=calls[c]
        if row['call_index']!=c or row['control_index']!=(None if c==0 else c-1) or row['role']!=('warmup' if c==0 else 'control') or row.get('error') or tuple(row['output_names'])!=OUTPUTS:raise RuntimeError('recorded call layout mismatch')
        if json.loads((directory/'attempts'/f'call_{c:04d}.json').read_text())!={'index':c,'status':'attempted_before_session_run'}:raise RuntimeError('attempt binding mismatch')
        for name,shape in [*zip(INPUTS,shapes),*[(n,(1,29)) for n in OUTPUTS]]:
            a=(ins if name in INPUTS else outs)[f'call_{c:04d}__'+name]
            if a.dtype!=np.float32 or a.shape!=shape or not np.isfinite(a).all():raise RuntimeError('saved feed/output invalid')
        if c<=18:
            gate.feed(c,{n:ins[f'call_{c:04d}__'+n] for n in INPUTS});gate.output(c,OUTPUTS,[outs[f'call_{c:04d}__'+n] for n in OUTPUTS])
            if c:gate.target(c-1,pd[c-1]);gate.state(c,q[c],v[c],t[c])
    if not gate.passed:raise RuntimeError('saved exact prefix failed')
    profile=profile_from_yaml();cfg=yaml.safe_load(Path(profile['yaml_path']).read_text())
    if tuple(cfg['_runtime']['onnx_in_names'])!=INPUTS or tuple(cfg['_runtime']['onnx_out_names'])!=OUTPUTS or cfg['control']['pd_target_max_accel'] is not None or cfg['control']['action_ema_alpha']!=1:raise RuntimeError('controller semantics changed')
    doc=ET.parse(profile['body_path']).getroot();compiler=doc.find('compiler');compiler.set('meshdir',str((Path(profile['body_path']).parent/compiler.get('meshdir')).resolve()))
    if not any(g.get('name')=='floor' for g in doc.find('worldbody').findall('geom')):ET.SubElement(doc.find('worldbody'),'geom',name='floor',type='plane',size='0 0 .05')
    model=mujoco.MjModel.from_xml_string(ET.tostring(doc,encoding='unicode'));data=mujoco.MjData(model);names=profile['joint_names']
    qa=[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names];va=[int(model.jnt_dofadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names]
    if qa!=list(range(7,36)) or va!=list(range(6,35)):raise RuntimeError('actual joint-address map mismatch')
    torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link');rot=[]
    for i in range(400):
        data.qpos[:]=q[i];data.qvel[:]=v[i];mujoco.mj_kinematics(model,data);rot.append(data.xquat[torso][[1,2,3,0]].copy().astype('float32'))
    heading=compute_yaw_offset_np(rot[0],sample_reference(ref,np.array([0.]))['anchor_rot'][0].astype('float32'));torsoerr=0.
    for i in range(400):
        a=sample_reference(ref,np.array([(i+s)*.02 for s in (1,2,4,8)]));key=f'call_{i+1:04d}__'
        expected={'current_dof_pos':q[i,qa].astype('float32')[None],'current_dof_vel':v[i,va].astype('float32')[None],'current_root_local_ang_vel':v[i,3:6].astype('float32')[None],'historical_processed_actions':np.zeros((1,1,29),np.float32) if i==0 else outs[f'call_{i:04d}__joint_pos_targets'][:,None,:],'mimic_future_dof_pos':a['dof_pos'].astype('float32')[None],'mimic_future_dof_vel':a['dof_vel'].astype('float32')[None],'mimic_future_anchor_rot':apply_heading_offset_np(heading,a['anchor_rot'].astype('float32'))[None]}
        for n,x in expected.items():same(x,ins[key+n],'all400 '+n)
        err=check_current_torso(rot[i],ins[key+'current_anchor_rot']);torsoerr=max(torsoerr,err)
        same(pd[i],outs[key+'joint_pos_targets'][0].astype('float64'),'graph PD assignment')
        assemble_reference_inputs(ref,i*.02)
    with np.load(directory/'substep_telemetry.npz',allow_pickle=False) as z:sub=z['values'];columns=tuple(z['columns'])
    if sub.shape!=(8000,7) or columns!=('step','time_s','root_height_m','contacts','max_abs_actuator_force','all_finite','contiguous') or not np.isfinite(sub).all() or not np.array_equal(sub[:,0],np.arange(1,8001)) or not np.all(sub[:,5:]==1) or not np.allclose(sub[:,1],np.arange(1,8001)*.001,rtol=0,atol=1e-10):raise RuntimeError('scalar substep integrity failed')
    same(sub[19::20,1],t[1:],'scalar time');same(sub[19::20,2],q[1:,2],'scalar height')
    if result['counts']!={'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1} or result['warnings_before']!=result['warnings_after']:raise RuntimeError('physical counters/warnings failed')
    refq=sample_reference(ref,np.arange(1,401)*.02)['dof_pos'].astype('float32');rmse=float(np.mean(np.sqrt(np.mean((q[1:,7:]-refq)**2,axis=1))));height=float(sub[:,2].min())
    refs=assets.references();rms=[]
    for source,(lo,hi) in zip(('wave','high5'),((1.2,3.2),(5.2,7.2))):
        mask=(t>=lo-1e-10)&(t<=hi+1e-10)
        if mask.sum()!=101:raise RuntimeError('RMS window state count differs')
        for side in ('left','right'):
            group=[j for j,n in enumerate(names) if n.startswith(side+'_') and any(s in n for s in ('shoulder_','elbow_','wrist_'))]
            if len(group)!=7:raise RuntimeError('arm group differs')
            rms.append(float(np.sqrt(np.mean((q[mask,7:][:,group]-refs[source].joint_position_rad[0,group])**2))))
    after={p.relative_to(directory).as_posix():sha(p) for p in directory.rglob('*') if p.is_file() and p.name not in ('worker.log','result.json')}
    if before!=after:raise RuntimeError('saved trial changed during audit')
    return {'schema':'pulsemorph.app_phrase_audit.v1','integrity_passed':True,'fresh_mj_kinematics_calls':400,'new_physics_steps':0,'new_onnx_calls':0,'current_torso_max_rotation_error_rad':torsoerr,'inventory':after,'metrics':{'arm_rms_rad':rms,'min_root_height_m':height,'mean_joint_rmse_rad':rmse,'physical_gate_passed':height>=.6 and rmse<=.35,'actual_physics_steps':8000,'online_onnx_call_count':400},'qualification':'400control saved-state/FK/feed/PD/history audit; scalar substeps, no full-substep-vector dynamics replay. RMS centers Wave0/High50 separately.'}


def execute(request_path):
    request_path=Path(request_path).resolve();directory=request_path.parent
    request=json.loads(request_path.read_text(encoding='utf8'))
    artifact_root=Path(request['artifact_root']).resolve()
    if request['schema']!='pulsemorph.app_phrase_request.v1' or not directory.is_relative_to(artifact_root) or directory==artifact_root or request_path.name!='request.json':raise RuntimeError('invalid private trial path/schema')
    write_new(directory/'worker_started.json',{'status':'started_once','request_sha256':sha(request_path)})
    counts={'calls_attempted':0,'calls_completed':0,'steps':0,'forward':0,'initial':0};states=[];vel=[];times=[];targets=[];calls=[];inputs=[];outputs=[];sub=[];restore=[];warnings_before=None;warnings_after=None;failure=None;started=time.monotonic()
    result={'schema':'pulsemorph.app_phrase_physical_trial.v1','status':'failed','program':request['program'],'id':request['id'],'index':request['index'],'job_id':request['job_id'],'music_observation':request['music_observation'],'source_pins':request['source_pins'],'new_llm_calls':0,'request_sha256':sha(request_path)}
    try:
        env=environment()
        if env!=request['environment']:raise RuntimeError('loaded numerical environment differs from request pins')
        result['versions']=env['versions'];result['environment']=env;verify(request['source_pins']);assets=PhraseAssets(ROOT)
        from deployment import test_tracker_mujoco as upstream
        import mujoco,onnxruntime as ort
        spec=importlib.util.spec_from_file_location('app_reset_only',ROOT/'experiments/agentic/app/execute_app.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        reference=load_reference(directory/'common_reference');player=preflight(assets,reference);gate=make_gate(assets)
        real_session,real_step,real_initial=ort.InferenceSession,mujoco.mj_step,upstream.set_initial_pose
        def budget():
            if failure is not None:raise RuntimeError('latched inference failure')
            if time.monotonic()-started>300:raise RuntimeError('wall budget exceeded')
        class Session:
            def __init__(self,*a,**kw):
                verify(request['source_pins']);options=ort.SessionOptions();options.intra_op_num_threads,options.inter_op_num_threads=2,1;kw['sess_options']=options
                self.session=real_session(*a,**kw);interface_guard(self.session)
                if self.session.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('CPU provider required')
                options=self.session.get_session_options()
                if (options.intra_op_num_threads,options.inter_op_num_threads)!=(2,1):raise RuntimeError('session thread drift')
                verify(request['source_pins'])
            def __getattr__(self,k):return getattr(self.session,k)
            def run(self,names,feed,*a,**kw):
                nonlocal failure
                budget();i=counts['calls_attempted'];call_budget(i,gate.passed)
                row={'call_index':i,'role':'warmup' if i==0 else 'control','control_index':None if i==0 else i-1,'output_names':list(names)};calls.append(row);inputs.append({n:v.copy() for n,v in feed.items()})
                try:
                    if tuple(feed)!=INPUTS or tuple(names)!=OUTPUTS or any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in feed.values()):raise RuntimeError('unsupported/nonfinite feed')
                    gate.feed(i,feed);write_new(directory/'attempts'/f'call_{i:04d}.json',{'index':i,'status':'attempted_before_session_run'});counts['calls_attempted']+=1
                    values=self.session.run(names,feed,*a,**kw);counts['calls_completed']+=1;outputs.append(dict(zip(names,[v.copy() for v in values])))
                    if len(values)!=4 or any(a.shape!=(1,29) or a.dtype!=np.float32 or not np.isfinite(a).all() for a in values):raise RuntimeError('invalid model output')
                    gate.output(i,names,values);return values
                except Exception as e:failure=str(e);row['error']=str(e);raise
        def initial(model,data,p):
            nonlocal warnings_before,warnings_after
            budget();counts['initial']+=1
            if counts['initial']!=1 or counts['steps']:raise RuntimeError('unexpected reset')
            real_initial(model,data,p)
            original.apply_initial_perturbation(model,data,{'joint_position_offset_rad':[0.]*29,'joint_velocity_offset_rad_s':[0.]*29,'root_linear_velocity_offset_world_m_s':[0.]*3,'root_yaw_offset_rad':0.},mujoco)
            warnings_before=[int(w.number) for w in data.warning];warnings_after=warnings_before.copy()
            states.append(data.qpos.copy());vel.append(data.qvel.copy());times.append(float(data.time));gate.state(0,data.qpos,data.qvel,data.time)
        def step(model,data):
            nonlocal warnings_after
            budget();step_budget(counts['steps']);before=float(data.time)
            if counts['steps']%20==0:targets.append(data.ctrl.copy());gate.target(counts['steps']//20,data.ctrl)
            real_step(model,data);counts['steps']+=1
            if counts['steps']%20==0:mujoco.mj_forward(model,data);counts['forward']+=1
            now=float(data.time);finite=bool(all(np.isfinite(a).all() for a in (data.qpos,data.qvel,data.qacc,data.ctrl,data.actuator_force)));contiguous=bool(np.isclose(now-before,.001,rtol=0,atol=1e-10));warnings_after=[int(w.number) for w in data.warning]
            sub.append([counts['steps'],now,float(data.qpos[2]),int(data.ncon),float(np.max(np.abs(data.actuator_force))),finite,contiguous])
            if not finite or not contiguous or warnings_after!=warnings_before:raise RuntimeError('physics finite/time/warning failure')
            if counts['steps']%20==0:states.append(data.qpos.copy());vel.append(data.qvel.copy());times.append(now);gate.state(counts['steps']//20,data.qpos,data.qvel,now)
        (directory/'attempts').mkdir()
        for owner,name,new in ((ort,'InferenceSession',Session),(mujoco,'mj_step',step),(upstream,'set_initial_pose',initial),(upstream,'MotionPlayer',lambda *a,**kw:(budget() or player))):restore.append((owner,name,getattr(owner,name)));setattr(owner,name,new)
        upstream.run(str(ROOT/MODEL),'safe_phrase_common_no_pickle',realtime=False,num_loops=1,render=False,cache_motion=False)
        if failure or counts!={'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1} or not gate.passed:raise RuntimeError('trial counts/prefix failed')
        result['prefix_gate']={'passed':True,'inputs_checked':gate.inputs_checked,'outputs_checked':gate.outputs_checked,'states_checked':gate.states_checked,'targets_checked':gate.targets_checked};result['status']='physical_completed_awaiting_audit'
    except Exception as e:result.update(error=str(e),error_type=type(e).__name__)
    finally:
        for owner,name,old in reversed(restore):setattr(owner,name,old)
        for name,value in (('qpos',states),('qvel',vel),('state_times',times),('applied_joint_targets',targets)):np.save(directory/(name+'.npy'),np.asarray(value))
        for name,records in (('policy_inputs',inputs),('policy_outputs',outputs)):np.savez_compressed(directory/(name+'.npz'),**{f'call_{i:04d}__{n}':a for i,row in enumerate(records) for n,a in row.items()})
        np.savez_compressed(directory/'substep_telemetry.npz',values=np.asarray(sub),columns=np.asarray(['step','time_s','root_height_m','contacts','max_abs_actuator_force','all_finite','contiguous']))
        write_new(directory/'policy_calls.json',calls);result.update(counts=counts,warnings_before=warnings_before,warnings_after=warnings_after,wall_seconds=time.monotonic()-started)
    if result['status']=='physical_completed_awaiting_audit':
        try:
            verify(request['source_pins']);audit=audit_saved(directory,assets,result);write_new(directory/'audit.json',audit);verify(request['source_pins']);assets.verify()
            result.update(status='completed',integrity_passed=True,audit_sha256=sha(directory/'audit.json'),metrics=audit['metrics'])
        except Exception as e:result.update(status='failed',error='saved-state audit failed: '+str(e),integrity_passed=False)
    write_new(directory/'result.json',result)
    return 0 if result['status']=='completed' else 1


def check_current_torso(rotation,recorded):
    """Exact NN-input bytes are mandatory; SO3 error is diagnostic only."""
    same(rotation[None],recorded,'current_anchor_rot exact FK input')
    return float((Rotation.from_quat(rotation).inv()*Rotation.from_quat(recorded[0])).magnitude())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--request',type=Path);group.add_argument('--environment',action='store_true');args=parser.parse_args()
    if args.environment:print(json.dumps(environment(),sort_keys=True))
    else:raise SystemExit(execute(args.request))
