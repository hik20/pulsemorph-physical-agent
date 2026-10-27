"""One prospective 8s phrase trial; preparation is the default separate action.

Numerical controller/PD/physics loop is the pinned upstream.run, unmodified.
Instrumentation follows execute_app.py: real_step then boundary mj_forward.
No execution is authorized by the existence of this file.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import shutil
import site
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
UPSTREAM=ROOT/'research/vendor/ProtoMotions'
PUBLIC=Path('C:/Users/halke/.cache/pulsemorph-agentic/envs/app-public-win-cp311-v1/Scripts/python.exe')
PLAN=ROOT/'artifacts/runs/agentic/phrase_program_execution_plan_v1'
REFERENCE_PLAN=ROOT/'artifacts/runs/agentic/phrase_program_reference_v2'
OUT=Path('C:/Users/halke/.cache/pulsemorph-agentic/experiments/phrase_program_execution_v1')
BASE=ROOT/'artifacts/runs/agentic/app/jobs/20261027T023422_8efe4bc4/trials/baseline'
MODEL=UPSTREAM/'data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx'
sys.path[:0]=[str(ROOT/'src'),str(UPSTREAM),str(HERE)]
import numpy as np
from adapter_program_v1 import PhrasePlayer,PrefixGate,interface_guard,call_budget,step_budget,INPUTS,OUTPUTS


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def verify(pins):
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('frozen input changed: '+p)


def environment():
    import mujoco,onnxruntime,numpy.core._multiarray_umath as npbinary
    import onnxruntime.capi.onnxruntime_pybind11_state as ortbinary
    if Path(sys.executable).resolve()!=PUBLIC.resolve() or not sys.flags.isolated or site.ENABLE_USER_SITE is not False:
        raise ValueError('public production -I/no-user-site environment required')
    prefix=PUBLIC.resolve().parents[1];origins={}
    for m in (np,mujoco,onnxruntime,npbinary,ortbinary):
        path=Path(m.__file__).resolve()
        if not path.is_relative_to(prefix):raise ValueError('module outside public environment')
        origins[str(path)]=sha(path)
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','mujoco','onnxruntime','torch')}
    if versions!={'numpy':'1.26.4','scipy':'1.16.3','mujoco':'3.3.7','onnxruntime':'1.31.0','torch':'2.7.1'}:
        raise ValueError('runtime versions drift')
    return {'python':str(PUBLIC),'python_sha256':sha(PUBLIC),'isolated':True,'user_site':False,'versions':versions,'module_pins':origins}


def preflight():
    """Pure reference/array checks. No model session, model run, FK or stepping."""
    from pulsemorph.motion.contracts import load_reference
    from pulsemorph.motion.g1 import profile_from_yaml,assemble_reference_inputs
    from deployment.tracker_inputs import init_tracker_alignment,align_future_motion_refs,build_tracker_onnx_inputs
    profile=profile_from_yaml();reference=load_reference(REFERENCE_PLAN/'common_reference')
    program=reference.metadata['provenance'].get('phrase_program')
    if not program or tuple(program['order'])!=('wave','high5'):
        raise ValueError('this concrete execution protocol only admits the frozen Wave-first program')
    import yaml
    name_map=yaml.safe_load(Path(profile['yaml_path']).read_text())['_runtime']['onnx_name_to_in_key']
    with np.load(BASE/'reference_cache.npz',allow_pickle=False) as cache:
        carrier={k:cache['committed__'+k][0].copy() for k in ('dof_pos','dof_vel','body_pos','body_rot','body_vel','body_ang_vel')}
    player=PhrasePlayer(reference,carrier,profile['anchor_body_index'])
    expected_reset=np.r_[carrier['body_pos'][0],carrier['body_rot'][0][[3,0,1,2]],carrier['dof_pos']].astype('float64')
    if expected_reset.tobytes()!=np.load(BASE/'qpos.npy',allow_pickle=False)[0].tobytes() or np.any(np.load(BASE/'qvel.npy',allow_pickle=False)[0]):
        raise ValueError('authentic first-source frame0 reset does not reproduce archived initial qpos/zero qvel')
    future=[player.get_future_references(i,[1,2,4,8]) for i in range(400)]
    for i in range(400):assemble_reference_inputs(reference,i*.02)
    # Actual archived current/history inputs held fixed: verify new adapter and
    # unmodified heading/assembly yield identical eight feeds for all18 rows.
    with np.load(BASE/'policy_inputs.npz',allow_pickle=False) as inputs:
        first=inputs['call_0001__current_anchor_rot'][0]
        alignment=init_tracker_alignment(robot_anchor_pos=carrier['body_pos'][profile['anchor_body_index']],robot_anchor_rot=first,motion_anchor_pos=carrier['body_pos'][profile['anchor_body_index']],motion_anchor_rot=carrier['body_rot'][profile['anchor_body_index']])
        for i in range(18):
            prefix=f'call_{i+1:04d}__'
            feed=build_tracker_onnx_inputs(onnx_input_names=list(INPUTS),onnx_name_to_key=name_map,dof_pos=inputs[prefix+'current_dof_pos'][0],dof_vel=inputs[prefix+'current_dof_vel'][0],anchor_rot=inputs[prefix+'current_anchor_rot'][0],root_local_ang_vel=inputs[prefix+'current_root_local_ang_vel'][0],future_refs=align_future_motion_refs(future[i],alignment),anchor_body_index=profile['anchor_body_index'],num_dofs=29,prev_actions=inputs[prefix+'historical_processed_actions'][0,0])
            if set(feed)!=set(INPUTS) or any(feed[n].tobytes()!=inputs[prefix+n].tobytes() for n in INPUTS):
                raise ValueError('actual legacy assembly prefix mismatch at '+str(i))
    return player,{'strict_future_queries':400,'actual_cached_state_legacy_assembly_prefix_rows':18,'all_eight_feed_bytes_equal':True,'source_reset_qpos_byte_equal':True,'archived_initial_qvel_zero':True,'auxiliary_rotations':'static authentic first-source0 non-torso rotations; derived layout only, never graph targets','root_target':'none'}


def prepare():
    PLAN.mkdir(parents=True,exist_ok=False)
    report=json.loads((REFERENCE_PLAN/'report.json').read_text())
    if report['status']!='reference_only_passed_awaiting_design_review' or not report['source_pins']:raise ValueError('reference preparation not passed/pinned')
    pins={str(ROOT/p):h for p,h in report['source_pins'].items()};verify(pins)
    env=environment();_,checks=preflight()
    extras=[Path(__file__),HERE/'adapter_program_v1.py',REFERENCE_PLAN/'report.json',REFERENCE_PLAN/'prefix_actual_assembly_parity.json',REFERENCE_PLAN/'common_reference/manifest.json',REFERENCE_PLAN/'common_reference/arrays.npz',MODEL,ROOT/'experiments/agentic/app/execute_app.py',ROOT/'design/agentic/PHRASE_PROGRAM_V1.md']
    extras += [UPSTREAM/'deployment'/n for n in ('test_tracker_mujoco.py','state_utils.py','tracker_inputs.py','motion_utils.py')]
    extras += sorted((UPSTREAM/'protomotions/data/assets/mesh/G1').glob('*.stl'))
    extras += [BASE/p for p in ('qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy','policy_calls.json','policy_inputs.npz','policy_outputs.npz','reference_cache.npz','manifest.json','preexecution_inputs.json')]
    for p in extras:pins[str(p)]=sha(p)
    pins.update(env['module_pins']);pins[str(PUBLIC)]=sha(PUBLIC)
    tree=ast.parse((UPSTREAM/'deployment/test_tracker_mujoco.py').read_text())
    numeric={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('run','set_initial_pose','load_mujoco_model','read_robot_state')}
    protocol={'schema':'pulsemorph.phrase_program_execution_protocol.v1','status':'PREPARED_NO_EXECUTION_AUTHORITY','pins':pins,'environment':env,'output':str(OUT),'reference_checks':checks,'original_reference_pins':len(report['source_pins']),
        'scientific_schedule':'Frozen model-proposed Wave-to-High5 program; source0..3.5, C1joint/C0torso bridge3.5..4.5, second(t-4) thereafter; exact compiled reference authoritative','duration_s':8,'maximum_trials':1,'maximum_online_calls':400,'maximum_warmups':1,'maximum_physics_steps':8000,'maximum_wall_seconds':300,'minimum_free_bytes':256*1024**2,'audio':True,'llm':False,'audio_scope':'Offline eight-second PCM summary informed the frozen program; execution makes no model/audio-analysis calls or beat-timing claim',
        'prefix_gate':{'online_decisions':[0,17],'warmup':True,'states':[0,18],'fields':'all eight inputs,four outputs,appliedPD,qpos,qvel,time','tolerance':0,'stop':'every mismatched row stops; after control17 physics require full gate before decision18'},
        'numeric_reuse':{'upstream_AST':numeric,'method':'Call pinned upstream.run unmodified with patched MotionPlayer/session/initial/step instrumentation. PD/action processing, robot-state read, heading, input assembly, real_step and reset are upstream; no alternate controller loop.','wrapper_basis':'execute_app.py real_step→control-boundary mj_forward; app zero perturbation path reused unchanged.','new_scope':'8s common-only player, exclusive output,401call/8000step budgets, exact prefix guard, explicit graph interface, typed-only target cache, no fabricated root reference metric'},
        'physical_screen':{'min_root_height_m':.6,'max_mean_joint_rmse_rad':.35,'integrity':'all substeps finite/time-contiguous/no new MuJoCo warnings; exactly1reset/400control-boundary-forward/8000steps/401completedcalls'},
        'limitations':['Torso C0 only; no angular velocity continuity or physical superiority claim','No reference root/body/contact composition, fullbody auxiliary rotation rows are unconsumed','One trial tool-composition feasibility, not skill learning/generalization/aesthetic or recovery proof'],
        'stop_rule':'Any pin/preflight/initial/feed/output/state/PD/finite/time/warning/budget/runtime failure stops and preserves this one trial. No retry,resume,replacement or tolerance/environment change.',
        'execution_approval_required':{'approved':True,'protocol_sha256':'exact frozen protocol digest','executor_sha256':sha(__file__)},'new_physics_steps':0,'new_onnx_calls':0}
    write(PLAN/'protocol.json',protocol)
    # Explicit source diff is review aid, not an assertion the large wrapper ASTs match.
    import difflib
    with (PLAN/'wrapper_diff.patch').open('x') as f:f.writelines(difflib.unified_diff((ROOT/'experiments/agentic/app/execute_app.py').read_text().splitlines(True),Path(__file__).read_text().splitlines(True),fromfile='unchanged_execute_app.py',tofile='new_execute_program_v1.py'))
    write(PLAN/'numerics_review.json',{'upstream_functions_used_without_copy_or_change':numeric,'new_wrapper_diff_sha256':sha(PLAN/'wrapper_diff.patch'),'source_file_sha256':sha(UPSTREAM/'deployment/test_tracker_mujoco.py'),'note':'Original upstream numeric loop is called directly; wrapper AST is different because setup/cache/gates are new. No claim of whole-wrapper AST equality. Static reviewed mechanics must be confirmed by exact in-loop prefix.'})


def approval_check(path):
    protocol=json.loads((PLAN/'protocol.json').read_text());approval=json.loads(Path(path).read_text())
    if approval.get('approved') is not True or approval.get('protocol_sha256')!=sha(PLAN/'protocol.json') or approval.get('executor_sha256')!=sha(__file__):raise ValueError('exact root approval required')
    verify(protocol['pins'])
    if environment()!=protocol['environment']:raise ValueError('environment changed')
    return protocol


def launch(approval):
    protocol=approval_check(approval)
    if OUT.exists():raise FileExistsError('single trial output already consumed')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(OUT.parent).free<protocol['minimum_free_bytes']:raise ValueError('insufficient disk')
    OUT.mkdir();write(OUT/'launch_attempt.json',{'trial_attempt':1,'approval_sha256':sha(approval),'protocol_sha256':sha(PLAN/'protocol.json'),'status':'attempted_before_worker_launch'})
    with (OUT/'worker.log').open('xb') as log:
        proc=subprocess.Popen([str(PUBLIC),'-I',str(Path(__file__).resolve()),'--worker',str(Path(approval).resolve())],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:code=proc.wait(timeout=protocol['maximum_wall_seconds'])
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait();write(OUT/'watchdog_timeout.json',{'status':'failed','reason':'wall cap300s','no_retry':True});return 1
    if not (OUT/'result.json').exists():
        write(OUT/'setup_failure.json',{'status':'failed','worker_exit':code,'durable_call_attempt_records':len(list((OUT/'attempts').glob('call_*.json'))),'reason':'Worker failed before publishing result; preserve log/attempts, no retry. Completed calls unknown unless independently evidenced.'})
    return code


def worker(approval):
    protocol=approval_check(approval)
    if not (OUT/'launch_attempt.json').is_file():raise ValueError('no prior exclusive launch')
    write(OUT/'worker_started.json',{'status':'started_once','new_llm_calls':0})
    import mujoco,onnxruntime as ort
    from deployment import test_tracker_mujoco as upstream
    app_spec=importlib.util.spec_from_file_location('original_app_executor',ROOT/'experiments/agentic/app/execute_app.py')
    app=importlib.util.module_from_spec(app_spec);app_spec.loader.exec_module(app)
    player,checks=preflight()
    with np.load(BASE/'policy_inputs.npz',allow_pickle=False) as z:baseline_inputs={n:z[n].copy() for n in z.files}
    with np.load(BASE/'policy_outputs.npz',allow_pickle=False) as z:baseline_outputs={n:z[n].copy() for n in z.files}
    gate=PrefixGate(baseline_inputs,baseline_outputs,*[np.load(BASE/n,allow_pickle=False) for n in ('qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy')])
    real_session,real_step,real_initial=ort.InferenceSession,mujoco.mj_step,upstream.set_initial_pose
    counts={'calls_attempted':0,'calls_completed':0,'steps':0,'forward':0,'initial':0};states=[];velocities=[];times=[];targets=[];calls=[];ins=[];outs=[];sub=[];warnings_initial=None;failure=None;restorations=[];started=time.monotonic()
    result={'schema':'pulsemorph.phrase_program_physical_trial.v1','status':'failed','new_llm_calls':0,'audio':True,'protocol_sha256':sha(PLAN/'protocol.json'),'source_pins':protocol['pins'],'reference_checks':checks}
    def budget():
        if failure is not None:raise RuntimeError('latched prior failure: '+failure)
        if time.monotonic()-started>300:raise RuntimeError('wall budget exceeded')
    class Session:
        def __init__(self,*a,**kw):
            verify(protocol['pins'])
            options=ort.SessionOptions();options.intra_op_num_threads,options.inter_op_num_threads=2,1;kw['sess_options']=options
            self.session=real_session(*a,**kw);interface_guard(self.session)
            if self.session.get_providers()!=['CPUExecutionProvider']:raise ValueError('CPU only')
            actual=self.session.get_session_options()
            if (actual.intra_op_num_threads,actual.inter_op_num_threads)!=(2,1):raise ValueError('session threads changed')
            verify(protocol['pins'])
        def __getattr__(self,k):return getattr(self.session,k)
        def run(self,names,feed,*a,**kw):
            nonlocal failure
            budget();index=counts['calls_attempted']
            call_budget(index,gate.passed)
            row={'call_index':index,'role':'warmup' if index==0 else 'control','control_index':None if index==0 else index-1,'output_names':list(names)}
            calls.append(row);ins.append({n:v.copy() for n,v in feed.items()})
            try:
                if tuple(feed)!=INPUTS or tuple(names)!=OUTPUTS or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in feed.values()):raise ValueError('exact finite float32 graph feeds required')
                gate.feed(index,feed)
                write(OUT/'attempts'/f'call_{index:04d}.json',{'index':index,'status':'attempted_before_session_run'})
                counts['calls_attempted']+=1
                values=self.session.run(names,feed,*a,**kw);counts['calls_completed']+=1;outs.append(dict(zip(names,[v.copy() for v in values])))
                if len(values)!=4 or any(v.shape!=(1,29) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in values):raise ValueError('nonfinite or unsupported output')
                gate.output(index,names,values)
                return values
            except Exception as e:failure=repr(e);row['error']=failure;raise
    def initial(model,data,motion_player):
        nonlocal warnings_initial
        budget();counts['initial']+=1
        if counts['initial']!=1 or counts['steps']:raise RuntimeError('unexpected reset')
        real_initial(model,data,motion_player)
        zero={'joint_position_offset_rad':[0.]*29,'joint_velocity_offset_rad_s':[0.]*29,'root_linear_velocity_offset_world_m_s':[0.]*3,'root_yaw_offset_rad':0.}
        app.apply_initial_perturbation(model,data,zero,mujoco)
        warnings_initial=[int(w.number) for w in data.warning]
        states.append(data.qpos.copy());velocities.append(data.qvel.copy());times.append(float(data.time))
        gate.state(0,data.qpos,data.qvel,data.time)
    def step(model,data):
        budget()
        step_budget(counts['steps'])
        before=float(data.time)
        if counts['steps']%20==0:
            targets.append(data.ctrl.copy());gate.target(counts['steps']//20,data.ctrl)
        real_step(model,data);counts['steps']+=1
        if counts['steps']%20==0:
            mujoco.mj_forward(model,data);counts['forward']+=1
        now=float(data.time)
        finite=bool(all(np.isfinite(v).all() for v in (data.qpos,data.qvel,data.qacc,data.ctrl,data.actuator_force)))
        contiguous=bool(np.isclose(now-before,.001,rtol=0,atol=1e-10))
        sub.append([counts['steps'],now,float(data.qpos[2]),int(data.ncon),float(np.max(np.abs(data.actuator_force))),finite,contiguous])
        if not finite or not contiguous or [int(w.number) for w in data.warning]!=warnings_initial:
            np.savez_compressed(OUT/'failed_substep_state.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy(),qacc=data.qacc.copy(),ctrl=data.ctrl.copy(),time=np.asarray([now]))
            raise RuntimeError('finite/time/warning physics failure')
        if counts['steps']%20==0:
            states.append(data.qpos.copy());velocities.append(data.qvel.copy());times.append(now)
            gate.state(counts['steps']//20,data.qpos,data.qvel,now)
    try:
        (OUT/'attempts').mkdir()
        verify(protocol['pins'])
        for owner,name,value in ((ort,'InferenceSession',Session),(mujoco,'mj_step',step),(upstream,'set_initial_pose',initial),(upstream,'MotionPlayer',lambda *a,**kw:(budget() or player))):
            restorations.append((owner,name,getattr(owner,name)));setattr(owner,name,value)
        upstream.run(str(MODEL),'phrase_common_reference_no_pickle',realtime=False,num_loops=1,render=False,cache_motion=False)
        if failure is not None or counts!={'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1} or not gate.passed:raise RuntimeError('trial counts/prefix failed')
        verify(protocol['pins'])
        q=np.asarray(states);ref=np.stack([player.get_state_at_frame(i)['dof_pos'] for i in range(401)])
        rmse=float(np.mean(np.sqrt(np.mean((q[1:,7:]-ref[1:])**2,axis=1))))
        height=float(min(x[2] for x in sub))
        result.update(status='completed',mean_joint_rmse_rad=rmse,min_root_height_m=height,physical_gate_passed=height>=.6 and rmse<=.35)
    except Exception as e:result['error']=repr(e);result['traceback']=traceback.format_exc()
    finally:
        for owner,name,original in reversed(restorations):setattr(owner,name,original)
        for name,array in (('qpos',states),('qvel',velocities),('state_times',times),('applied_joint_targets',targets)):
            np.save(OUT/(name+'.npy'),np.asarray(array))
        for name,records in (('policy_inputs',ins),('policy_outputs',outs)):
            np.savez_compressed(OUT/(name+'.npz'),**{f'call_{i:04d}__{n}':v for i,row in enumerate(records) for n,v in row.items()})
        np.savez_compressed(OUT/'substep_telemetry.npz',values=np.asarray(sub),columns=np.asarray(['step','time_s','root_height_m','contacts','max_abs_actuator_force','all_finite','contiguous']))
        from pulsemorph.motion.contracts import save_reference
        save_reference(player.reference,OUT/'common_reference')
        write(OUT/'policy_calls.json',calls)
        result.update(counts=counts,prefix_gate={'passed':gate.passed,'inputs_checked':gate.inputs_checked,'outputs_checked':gate.outputs_checked,'states_checked':gate.states_checked,'targets_checked':gate.targets_checked},wall_seconds=time.monotonic()-started)
        write(OUT/'result.json',result)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--execute',type=Path);g.add_argument('--worker',type=Path)
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.execute:raise SystemExit(launch(a.execute))
    else:raise SystemExit(worker(a.worker))
