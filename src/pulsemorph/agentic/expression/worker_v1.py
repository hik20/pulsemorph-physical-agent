"""Expression-only worker; copied controller closures, explicitly new provenance."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback,site,importlib.metadata,re
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions')]
import numpy as np
from scipy.spatial.transform import Rotation
from pulsemorph.motion.g1 import sample_reference,assemble_reference_inputs,profile_from_yaml
from pulsemorph.motion.contracts import load_reference
from pulsemorph.agentic.app.phrase.assets import PhraseAssets,sha,write_new,MODEL,PUBLIC_PYTHON,INPUTS,OUTPUTS
from pulsemorph.agentic.app.phrase import worker as frozen
from pulsemorph.agentic.app.phrase.worker import same,PrefixGate,interface_guard,call_budget,step_budget,environment,verify,make_gate,check_current_torso
from pulsemorph.agentic.expression.player_v1 import ExpressionPlayer,validate_expression_reference
from pulsemorph.agentic.phrase_expression_v1 import parse_expression
from pulsemorph.agentic.phrase_program import _arm_indices
from pulsemorph.agentic.expression.runtime_v1 import expected_implementation_pins

def load_request(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink() or bool(getattr(part.lstat(),'st_file_attributes',0)&1024):raise ValueError('Linked expression request forbidden')
    if not path.is_file() or path.stat().st_size>128*1024:raise ValueError('Bounded ordinary expression request required')
    with path.open('rb') as f:raw=f.read(128*1024+1)
    if len(raw)>128*1024:raise ValueError('Expression request exceeds bound')
    return json.loads(raw)

def validate_request(path,request):
    if Path(path).stat().st_size>128*1024 or type(request) is not dict:raise ValueError('Bounded expression request required')
    fields={'schema','artifact_root','id','index','job_id','expression','source_pins','environment','music_observation'}
    if set(request)!=fields or request['schema']!='pulsemorph.expression_physical_request.v1' or type(request['index']) is not int or request['index']< -1 or type(request['job_id']) is not str or re.fullmatch(r'[A-Za-z0-9_-]{1,100}',request['job_id']) is None:raise ValueError('Exact expression request schema/job/index required')
    expected_id=request['job_id']+('__baseline' if request['index']==-1 else '__trial_'+str(request['index']))
    if type(request['id']) is not str or request['id']!=expected_id:raise ValueError('Expression request ID assignment differs')
    parsed=parse_expression(request['expression'])
    if parsed['order']!=['wave','high5']:raise ValueError('Only Wave→High5 expression physical admission')
    if request.get('index')==-1 and any(parsed[k]!=1 for k in parsed if k.endswith('_gain')):raise ValueError('Unit expression baseline required')
    verify(request['source_pins'])
    reference=load_reference(Path(path).parent/'common_reference');validate_expression_reference(reference,parsed)
    if reference.metadata['reference_id']!=request['id'] or reference.metadata['context']['task_id']!=request['job_id'] or reference.metadata['context']['runtime_session_id']!=request['id']:raise ValueError('Expression request/reference context mismatch')

def validate_pin_assignment(request,assets,env,directory):
    expected=expected_implementation_pins(ROOT,assets,PUBLIC_PYTHON,env)
    music=request['music_observation'];expected[music['path']]=music['sha256']
    for name in ('common_reference/manifest.json','common_reference/arrays.npz','compile_report.json'):expected[str(directory/name)]=sha(directory/name)
    if request['source_pins']!=expected:raise RuntimeError('Expression exact source assignment mismatch')
    # Reuse original complete PCM validation; music remains fixed provenance.
    from pulsemorph.agentic.app.phrase.runtime import PhraseRuntime
    if PhraseRuntime._music(music)!=music:raise RuntimeError('Expression music observation mismatch')

def audit_saved(directory,assets,result):
    audit=frozen.audit_saved(directory,assets,result)
    paths=[directory/'qpos.npy',directory/'state_times.npy']
    for path in paths:
        if sha(path)!=audit['inventory'][path.name]:raise RuntimeError('Diagnostic saved state changed')
    q=np.load(paths[0],allow_pickle=False);t=np.load(paths[1],allow_pickle=False);groups=_arm_indices(profile_from_yaml()['joint_names']);diagnostics=[]
    for phase,(lo,hi) in enumerate(((1.2,3.2),(5.2,7.2))):
        keep=(t>=lo-1e-10)&(t<=hi+1e-10)
        if keep.sum()!=101:raise RuntimeError('Expression diagnostic window differs')
        for side,indices in zip(('left','right'),groups):
            values=q[keep][:,7+indices];mean=values.mean(axis=0);source=assets.references()[('wave','high5')[phase]];q0=source.joint_position_rad[0,indices]
            diagnostics.append({'phase':phase,'side':side,'window_s':[lo,hi],'sample_count':101,'centered_temporal_rms_rad':float(np.sqrt(np.mean((values-mean)**2))),'mean_pose_offset_rms_rad':float(np.sqrt(np.mean((mean-q0)**2))),'per_joint_peak_to_peak_rad':np.ptp(values,axis=0).tolist()})
    for path in paths:
        if sha(path)!=audit['inventory'][path.name]:raise RuntimeError('Diagnostic saved state changed')
    json.dumps(diagnostics,allow_nan=False)
    return {**audit,'schema':'pulsemorph.expression_saved_audit.v1','numerical_audit_source_sha256':sha(frozen.__file__),'expression_diagnostics':diagnostics,'diagnostic_new_FK_calls':0,'qualification':audit['qualification']+' Expression producer/request validated without relabel; appended temporal/mean-pose diagnostics do not replace original own-q0 RMS/physical screens.'}

def preflight(assets,reference,expression):
    player=ExpressionPlayer(reference,assets.arrays('initial_carrier'),16,expression)
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


def execute(request_path):
    request_path=Path(request_path).absolute();directory=request_path.parent
    request=load_request(request_path)
    artifact_root=Path(request['artifact_root']).resolve()
    if request['schema']!='pulsemorph.expression_physical_request.v1' or not directory.is_relative_to(artifact_root) or directory==artifact_root or request_path.name!='request.json':raise RuntimeError('invalid private trial path/schema')
    validate_request(request_path,request)
    write_new(directory/'worker_started.json',{'status':'started_once','request_sha256':sha(request_path)})
    counts={'calls_attempted':0,'calls_completed':0,'steps':0,'forward':0,'initial':0};states=[];vel=[];times=[];targets=[];calls=[];inputs=[];outputs=[];sub=[];restore=[];warnings_before=None;warnings_after=None;failure=None;started=time.monotonic()
    result={'schema':'pulsemorph.expression_physical_trial.v1','status':'failed','expression':parse_expression(request['expression']),'id':request['id'],'index':request['index'],'job_id':request['job_id'],'music_observation':request['music_observation'],'source_pins':request['source_pins'],'new_llm_calls':0,'request_sha256':sha(request_path)}
    try:
        env=environment()
        if env!=request['environment']:raise RuntimeError('loaded numerical environment differs from request pins')
        result['versions']=env['versions'];result['environment']=env;verify(request['source_pins']);assets=PhraseAssets(ROOT);validate_pin_assignment(request,assets,env,directory)
        from deployment import test_tracker_mujoco as upstream
        import mujoco,onnxruntime as ort
        spec=importlib.util.spec_from_file_location('app_reset_only',ROOT/'experiments/agentic/app/execute_app.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        reference=load_reference(directory/'common_reference');validate_expression_reference(reference,request['expression']);player=preflight(assets,reference,request['expression']);gate=make_gate(assets)
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
            result.update(status='completed',integrity_passed=True,audit_sha256=sha(directory/'audit.json'),metrics=audit['metrics'],expression_diagnostics=audit['expression_diagnostics'],audit_numerical_source_sha256=audit['numerical_audit_source_sha256'])
        except Exception as e:result.update(status='failed',error='saved-state audit failed: '+str(e),integrity_passed=False)
    write_new(directory/'result.json',result)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--request',type=Path);group.add_argument('--environment',action='store_true');args=parser.parse_args()
    if args.environment:print(json.dumps(environment(),sort_keys=True))
    else:raise SystemExit(execute(args.request))
