"""One live MuJoCo instance + one fixed ONNX actor. Construction is execution."""
from pathlib import Path
import hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
from .runtime_v1 import canonical

class PortableTrackerBackend:
 @classmethod
 def start(cls,root,*,authority,config):
  """Root must approve a frozen authority before calling; never called at import."""
  return cls(root,authority=authority,config=config)
 def __init__(self,root,*,authority,config):
  if type(authority) is not dict or set(authority)!={'schema','session_id','max_control_steps','source_pins'} or authority['schema']!='pulsemorph.embodied_execution_authority.v1' or type(authority['max_control_steps']) is not int or not 1<=authority['max_control_steps']<=500:raise ValueError('Explicit finite root execution authority required')
  self.config=config;self.vendor=config['vendor_root'];self.root=Path(root).resolve();self.authority=authority;self.source_pins=dict(authority['source_pins']);self._verify()
  sys.path.insert(0,str(self.vendor))
  import mujoco,onnxruntime as ort,yaml
  from deployment import test_tracker_mujoco as upstream
  from deployment.tracker_inputs import get_tracker_input_requirements,init_tracker_alignment
  from .app.interface_v5 import MODEL_SHA,BODY_SHA,CONTROLLER_SHA,INPUTS,OUTPUTS
  MODEL='data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx'
  from .app.profile_v5 import verify_vendor,verify_profile
  identity=verify_vendor(config);verify_profile(config)
  required={str((self.vendor/row['path']).resolve()):row['sha256'] for row in identity['files']}
  for p in (Path(__file__),config['asset_identity'],config['profile_dir']/'manifest.json',config['profile_dir']/'standing.npz',config['runtime_python']):
   required[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
  if any(self.source_pins.get(p)!=h for p,h in required.items()):raise RuntimeError('Authority omits or changes portable assets/profile/source/runtime')
  from .app.interface_v5 import interface_guard
  from .app.environment_v5 import environment
  from pulsemorph.motion.g1 import profile_from_yaml
  self.mj=mujoco;self.upstream=upstream;self.INPUTS=INPUTS;self.OUTPUTS=OUTPUTS;self.environment=environment(config)
  if any(self.source_pins.get(p)!=h for p,h in self.environment['module_pins'].items()):raise RuntimeError('Authority omits or changes loaded numerical modules')
  profile=profile_from_yaml(self.vendor/(MODEL[:-5]+'.yaml'));self.profile=profile
  if Path(profile['body_path']).resolve()!=(self.vendor/'protomotions/data/assets/mjcf/g1_holo_compat.xml').resolve():raise RuntimeError('Body resolved outside configured vendor')
  for name,digest in ((MODEL,MODEL_SHA),(MODEL[:-5]+'.yaml',CONTROLLER_SHA),('protomotions/data/assets/mjcf/g1_holo_compat.xml',BODY_SHA)):
   if self.source_pins.get(str((self.vendor/name).resolve()))!=digest:raise RuntimeError('Authority must bind exact model/body/controller')
  cfg=yaml.safe_load((self.vendor/(MODEL[:-5]+'.yaml')).read_text());self.cfg=cfg;self.joint_names=list(profile['joint_names'])
  if (cfg['timing']['control_dt'],cfg['timing']['physics_dt'],cfg['timing']['decimation'])!=(.02,.001,20) or cfg['control']['pd_target_max_accel'] is not None or cfg['control']['action_ema_alpha']!=1:raise RuntimeError('Controller timing/filter semantics differ')
  requirements=get_tracker_input_requirements(cfg['_runtime']['onnx_name_to_in_key'])
  if requirements.needs_reference_positions or requirements.needs_spatial_start:raise RuntimeError('This actor requires unsupported spatial carriers')
  from pulsemorph.agentic.phrase_program import _joint_limits
  ranges=_joint_limits(profile);self.joint_ranges=[ranges[n] for n in self.joint_names]
  # Same MJCF configuration without upstream temporary writes into vendor folders.
  xml=ET.fromstring(upstream._patch_mjcf_xml(Path(profile['body_path'])))
  compiler=xml.find('compiler')
  for key in ('meshdir','texturedir'):
   if compiler is not None and compiler.get(key):compiler.set(key,str((Path(profile['body_path']).parent/compiler.get(key)).resolve()))
  self.patched_xml=ET.tostring(xml,encoding='unicode')
  self.model=mujoco.MjModel.from_xml_string(self.patched_xml);self.data=mujoco.MjData(self.model)
  if (self.model.nq,self.model.nv,self.model.nu)!=(36,35,29):raise RuntimeError('Exact G1 dimensions required')
  self.model.opt.timestep=.001;self.model.jnt_stiffness[:]=0;self.model.dof_damping[:]=0;self.model.dof_frictionloss[:]=0
  stiffness=cfg['control']['stiffness'];damping=cfg['control']['damping']
  for i,(kp,kd) in enumerate(zip(stiffness,damping)):
   self.model.actuator_gainprm[i,0]=kp;self.model.actuator_biastype[i]=1;self.model.actuator_biasprm[i,:3]=[0,-kp,-kd];self.model.actuator_ctrllimited[i]=0
  self.addresses=[int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in self.joint_names]
  if self.addresses!=list(range(6,35)):raise RuntimeError('Joint address map differs')
  with np.load(config['profile_dir']/'standing.npz',allow_pickle=False) as z:
   self.data.qpos[:]=z['qpos'];self.data.qvel[:]=z['qvel']
  self.initialization={'kind':'owned_symmetric_stand_v1; no source motion or historical profile','profile_manifest_sha256':hashlib.sha256((config['profile_dir']/'manifest.json').read_bytes()).hexdigest()}
  self.initialization_count=1;self.forward_count=0;mujoco.mj_forward(self.model,self.data);self.forward_count+=1
  self.warning_initial=[int(w.number) for w in self.data.warning];self.min_height=float(self.data.qpos[2]);self.control_index=0;self.step_count=0;self.calls=[];self.inputs=[];self.outputs=[];self.substeps=[];self.states=[];self.velocities=[];self.times=[];self.targets=[];self.integration_states=[];self.stopped=False;self.jacobian_calls=0
  self.prev_targets=np.zeros(29,np.float32);self.anchor_index=cfg['robot']['anchor_body_index'];self.root_index=cfg['robot']['root_body_index']
  state=upstream.read_robot_state(self.data,self.anchor_index,self.root_index);self.anchor_target=state['body_rot'][self.anchor_index].copy()
  self.alignment=init_tracker_alignment(state['body_pos'][self.anchor_index],self.anchor_target,state['body_pos'][self.anchor_index],self.anchor_target)
  options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
  self.session=ort.InferenceSession(str(self.vendor/MODEL),sess_options=options,providers=['CPUExecutionProvider']);interface_guard(self.session)
  if self.session.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('CPU actor only')
  self._verify();dummy={i.name:np.zeros([1 if d is None or isinstance(d,str) else d for d in i.shape],np.float32) for i in self.session.get_inputs()}
  self._infer(dummy,'warmup');self._record_state();self._cached_observation=None
 def _verify(self):
  for p,h in self.source_pins.items():
   path=Path(p)
   if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise RuntimeError('Embodied execution source pin changed')
 @property
 def sim_time_s(self):return float(self.data.time)
 def _infer(self,feed,role):
  if self.stopped or len(self.calls)>=self.authority['max_control_steps']+1:raise RuntimeError('Inference authority exhausted/stopped')
  if tuple(feed)!=self.INPUTS or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in feed.values()):raise RuntimeError('Actual8feed interface mismatch')
  self.inputs.append({k:v.copy() for k,v in feed.items()});self.calls.append({'index':len(self.calls),'role':role,'sim_time_s':self.sim_time_s,'status':'attempted'})
  try:
   out=self.session.run(list(self.OUTPUTS),feed)
   if len(out)!=4 or any(v.shape!=(1,29) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in out):raise RuntimeError('Actor output interface mismatch')
   self.outputs.append(dict(zip(self.OUTPUTS,[x.copy() for x in out])));self.calls[-1]['status']='completed';return out
  except Exception:self.stopped=True;raise
 def _integration(self):
  kind=self.mj.mjtState.mjSTATE_INTEGRATION;array=np.empty(self.mj.mj_stateSize(self.model,kind));self.mj.mj_getState(self.model,self.data,array,kind);return array
 def _record_state(self):
  self.states.append(self.data.qpos.copy());self.velocities.append(self.data.qvel.copy());self.times.append(self.sim_time_s);self.integration_states.append(self._integration())
 def advance(self,future_q,future_dq):
  if self.stopped or self.control_index>=self.authority['max_control_steps']:raise RuntimeError('Episode stopped or control budget exhausted')
  from deployment.tracker_inputs import build_tracker_onnx_inputs,align_future_motion_refs
  self._cached_observation=None
  state=self.upstream.read_robot_state(self.data,self.anchor_index,self.root_index)
  future={'dof_pos':np.asarray(future_q,np.float32),'dof_vel':np.asarray(future_dq,np.float32),'body_rot':np.tile(self.anchor_target,(4,len(self.cfg['robot']['body_names']),1))}
  future=align_future_motion_refs(future,self.alignment)
  feed=build_tracker_onnx_inputs(onnx_input_names=list(self.INPUTS),onnx_name_to_key=self.cfg['_runtime']['onnx_name_to_in_key'],dof_pos=state['dof_pos'],dof_vel=state['dof_vel'],anchor_rot=state['body_rot'][self.anchor_index],root_local_ang_vel=state['root_local_ang_vel'],future_refs=future,anchor_body_index=self.anchor_index,num_dofs=29,prev_actions=self.prev_targets)
  out=self._infer(feed,'control');pd=out[1][0].copy();self.prev_targets=pd.copy();self.targets.append(pd.copy());self.data.ctrl[:]=pd
  for _ in range(20):
   self.mj.mj_step(self.model,self.data);self.step_count+=1;self.min_height=min(self.min_height,float(self.data.qpos[2]));finite=bool(all(np.isfinite(v).all() for v in (self.data.qpos,self.data.qvel,self.data.qacc,self.data.ctrl,self.data.actuator_force)));warnings=[int(w.number) for w in self.data.warning];self.substeps.append([self.step_count,self.sim_time_s,float(self.data.qpos[2]),self.data.ncon,finite,warnings==self.warning_initial])
   if not finite or self.data.qpos[2]<.6 or warnings!=self.warning_initial:
    self.stopped=True
    if finite:self.mj.mj_forward(self.model,self.data);self.forward_count+=1
    raise RuntimeError('Actual physical finite/rootheight/warning stop')
  self.mj.mj_forward(self.model,self.data);self.forward_count+=1;self.control_index+=1;self._record_state();self._cached_observation=None
 def observe(self):
  if self._cached_observation is None:
   bodies={};jacobians={}
   for i in range(1,self.model.nbody):
    name=self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_BODY,i)
    if name:bodies[name]={'position_world_m':self.data.xpos[i].tolist(),'orientation_xyzw':self.data.xquat[i][[1,2,3,0]].tolist()}
   for name in ('left_rubber_hand','right_rubber_hand'):
    i=self.mj.mj_name2id(self.model,self.mj.mjtObj.mjOBJ_BODY,name)
    if i<0:raise RuntimeError('Actual wrist body missing')
    jp=np.zeros((3,self.model.nv));jr=np.zeros((3,self.model.nv));self.mj.mj_jacBody(self.model,self.data,jp,jr,i);self.jacobian_calls+=1;jacobians[name]={n:jp[:,a].tolist() for n,a in zip(self.joint_names,self.addresses)}
   contacts=[]
   for i in range(self.data.ncon):
    c=self.data.contact[i];contacts.append({'body1':self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[c.geom1])),'body2':self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[c.geom2])),'position_world_m':c.pos.tolist(),'distance_m':float(c.dist)})
   self._cached_observation={'sim_time_s':self.sim_time_s,'qpos':self.data.qpos.tolist(),'qvel':self.data.qvel.tolist(),'body_poses':bodies,'contacts':contacts,'processed_history_sha256':hashlib.sha256(self.prev_targets.tobytes()).hexdigest(),'physical_flags':{'finite':bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()),'root_height_m':float(self.data.qpos[2]),'min_root_height_m':self.min_height,'warnings_unchanged':[int(w.number) for w in self.data.warning]==self.warning_initial},'wrist_jacobians_world_m_per_rad':jacobians}
  import copy
  return copy.deepcopy(self._cached_observation)
 def closure_sha256(self):
  h=hashlib.sha256(self._integration().tobytes());h.update(self.prev_targets.tobytes());h.update(self.anchor_target.tobytes())
  for name in ('heading_offset','odom_start_xy','odom_start_heading_inv','robot_anchor_start_pos','motion_anchor_start_pos'):h.update(np.asarray(getattr(self.alignment,name)).tobytes())
  for value in (self.model.actuator_gainprm,self.model.actuator_biasprm,self.model.jnt_stiffness,self.model.dof_damping,self.model.dof_frictionloss):h.update(np.asarray(value).tobytes())
  h.update(canonical({'control_index':self.control_index,'physics_steps':self.step_count,'initializations':self.initialization_count,'session_id':self.authority['session_id'],'timestep':self.model.opt.timestep,'integrator':int(self.model.opt.integrator),'solver':int(self.model.opt.solver),'source_pins':self.source_pins,'actor_has_no_hidden_recurrent_inputs':True}));return h.hexdigest()
 def export(self,directory):
  directory=Path(directory)
  (directory/'render_model.xml').write_text(self.patched_xml,encoding='utf8')
  for name,value in [('qpos',self.states),('qvel',self.velocities),('state_times',self.times),('applied_joint_targets',self.targets),('mj_integration_states',self.integration_states)]:np.save(directory/(name+'.npy'),np.asarray(value))
  for name,rows in [('policy_inputs',self.inputs),('policy_outputs',self.outputs)]:np.savez_compressed(directory/(name+'.npz'),**{f'call_{i:04d}__'+k:v for i,row in enumerate(rows) for k,v in row.items()})
  np.save(directory/'substep_telemetry.npy',np.asarray(self.substeps))
  (directory/'backend.json').write_bytes(canonical({'schema':'pulsemorph.embodied_backend.v1','initialization':self.initialization,'initialization_count':self.initialization_count,'mjdata_instances':1,'onnx_sessions':1,'physics_steps':self.step_count,'online_calls':self.control_index,'warmup_calls':1,'calls':self.calls,'forward_calls':self.forward_count,'wrist_jacobian_calls':self.jacobian_calls,'source_pins':self.source_pins,'environment':self.environment,'controller':'fixed pretrained tracker; arbitrary reference/balance capability unproven until physical pilot','native_filesystem_monitor':False}))
