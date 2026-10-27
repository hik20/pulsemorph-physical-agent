"""Versioned classical arm servo + fixed pretrained legs/trunk tracker.

MuJoCo force law: https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model
qfrc_bias includes gravity/Coriolis/centrifugal terms, not contact constraint force.
No source file, joint limit, actuator force limit or model weight is altered.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .backend_v1 import TrackerBackend
from .runtime_v1 import canonical
MODE='classical_arm_PD_velocity_and_sampled_bias_plus_pretrained_legs_trunk_v1'

def root_up_z(qpos):
 q=np.asarray(qpos,dtype=float)
 if q.shape!=(36,):raise ValueError('G1 qpos36 required')
 return float(1-2*(q[4]*q[4]+q[5]*q[5]))

def applied_targets(raw_graph_pd,q_ref,dq_ref,qfrc_bias_joint,kp,kd,arm_indices):
 """Pure formula with explicit float32 application/history boundary; no clamp."""
 raw=np.asarray(raw_graph_pd)
 if raw.shape!=(29,) or raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Exact rawgraph float32 PD required')
 values=[np.asarray(v,dtype=float) for v in (q_ref,dq_ref,qfrc_bias_joint,kp,kd)]
 if any(v.shape!=(29,) or not np.isfinite(v).all() for v in values):raise ValueError('Finite29joint servo arrays required')
 q,dq,bias,kp,kd=values;indices=np.asarray(arm_indices)
 if indices.shape!=(14,) or indices.dtype.kind not in 'iu' or len(set(indices.tolist()))!=14 or np.any(indices<0) or np.any(indices>=29):raise ValueError('Exact14distinctarm indices required')
 if np.any(kp<=0) or np.any(kd<0):raise ValueError('Positivekp/nonnegativekd required')
 result=raw.copy();computed=q[indices]+kd[indices]/kp[indices]*dq[indices]+bias[indices]/kp[indices]
 if not np.isfinite(computed).all():raise ValueError('Nonfinite hybrid command')
 result[indices]=computed.astype(np.float32)
 if not np.isfinite(result).all():raise ValueError('Float32 hybrid command overflow')
 return result

class HybridArmBackend(TrackerBackend):
 def __init__(self,root,*,authority):
  source=str(Path(__file__).resolve());digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  if authority.get('source_pins',{}).get(source)!=digest:raise ValueError('Authority must bind new hybrid control implementation')
  super().__init__(root,authority=authority)
  self.arm_indices=[i for i,n in enumerate(self.joint_names) if n.startswith(('left_','right_')) and any(part in n for part in ('shoulder_','elbow_','wrist_'))]
  if len(self.arm_indices)!=14:raise RuntimeError('Exact14namedarm joints required')
  self.nonarm_indices=[i for i in range(29) if i not in self.arm_indices]
  self.kp=self.model.actuator_gainprm[:,0].copy();self.kd=-self.model.actuator_biasprm[:,2].copy();self._validate_scalar_PD()
  self.hybrid_config={'schema':'pulsemorph.embodied_hybrid_arm_config.v1','mode':MODE,'arm_indices':self.arm_indices,'arm_joint_names':[self.joint_names[i] for i in self.arm_indices],'nonarm_indices':self.nonarm_indices,'formula':'ctrl_arm=q_ref+(kd/kp)*dq_ref+qfrc_bias_at_current_joint_dof/kp','qref_sample':'first future target at +20ms; fromactualLLMauthored common reference','bias_sample':'actual same-current-state20msboundary qfrc_bias, held over20physics substeps; gravity+Coriolis+centrifugal, excludes contactconstraintforce','application_dtype':'float64 formula then explicitfloat32 assignment/history; no numericalclipping','history':'previous APPLIEDjointtargets, notrawgraphPD/actions','limited_physical_gate':{'min_root_height_m':.6,'min_root_up_z':.7,'qualification':'Finite/rootheight/upright/warning/aggregateforce screening only, notfullbalance or hardware safety'},'qualifications':['Arms use classical local servo, not a newly learned policy.','Legs/trunk use the unchanged pretrained actor, with changed applied arm history/distribution.','No hardware safety, arbitrary reachability or exact dynamic decoupling claim.','ctrlrange is not torque limit; individual actuator and aggregate joint force limits are recorded separately and unchanged.'],'compiled_model':self._model_contract()}
  self.min_root_up_z=root_up_z(self.data.qpos);self.hybrid_rows=[];self.hybrid_force_rows=[]
 def _validate_scalar_PD(self):
  m=self.model;mj=self.mj
  if np.any(self.kp<=0) or np.any(self.kd<0) or not np.isfinite(self.kp).all() or not np.isfinite(self.kd).all():raise RuntimeError('Invalid existing PD gains')
  if not np.array_equal(m.actuator_gaintype,np.zeros(29,dtype=int)) or not np.array_equal(m.actuator_biastype,np.ones(29,dtype=int)) or not np.array_equal(m.actuator_dyntype,np.zeros(29,dtype=int)):raise RuntimeError('Expected fixedgain affinebias statelessPD actuators')
  if np.any(m.actuator_biasprm[:,0]) or not np.array_equal(m.actuator_biasprm[:,1],-self.kp):raise RuntimeError('Unexpected PD bias/offset')
  if not np.array_equal(m.actuator_gear,np.tile([1.,0.,0.,0.,0.,0.],(29,1))):raise RuntimeError('Hybrid requires unchangedunitgear transmission')
  if not np.all(m.actuator_trntype==int(mj.mjtTrn.mjTRN_JOINT)):raise RuntimeError('Direct jointtransmission required')
  for i,name in enumerate(self.joint_names):
   j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name)
   if int(m.actuator_trnid[i,0])!=j or int(m.jnt_dofadr[j])!=self.addresses[i] or int(m.jnt_qposadr[j])!=7+i:raise RuntimeError('Actuator/jointDOF/order mismatch')
  if not np.allclose(self.data.actuator_length,self.data.qpos[7:],rtol=0,atol=1e-10):raise RuntimeError('Nonzero scalar actuator length offset unsupported')
  if np.any(m.body_gravcomp):raise RuntimeError('Existing passive gravity compensation woulddoublecountbias')
 def _model_contract(self):
  m=self.model
  return {name:np.asarray(getattr(m,name)).tolist() for name in ('actuator_gainprm','actuator_biasprm','actuator_gear','actuator_trntype','actuator_trnid','actuator_gaintype','actuator_biastype','actuator_dyntype','actuator_ctrllimited','actuator_ctrlrange','actuator_forcelimited','actuator_forcerange','jnt_actfrclimited','jnt_actfrcrange','jnt_limited','jnt_range')}
 def _check_hybrid(self):
  if self._model_contract()!=self.hybrid_config['compiled_model']:raise RuntimeError('Hybrid model/gains/limits changed')
 def advance(self,future_q,future_dq):
  if self.stopped or self.control_index>=self.authority['max_control_steps']:raise RuntimeError('Hybrid episode stopped or controlbudget exhausted')
  self._check_hybrid();self._cached_observation=None
  from deployment.tracker_inputs import build_tracker_onnx_inputs,align_future_motion_refs
  state=self.upstream.read_robot_state(self.data,self.anchor_index,self.root_index)
  future={'dof_pos':np.asarray(future_q,np.float32),'dof_vel':np.asarray(future_dq,np.float32),'body_rot':np.tile(self.anchor_target,(4,len(self.cfg['robot']['body_names']),1))};future=align_future_motion_refs(future,self.alignment)
  feed=build_tracker_onnx_inputs(onnx_input_names=list(self.INPUTS),onnx_name_to_key=self.cfg['_runtime']['onnx_name_to_in_key'],dof_pos=state['dof_pos'],dof_vel=state['dof_vel'],anchor_rot=state['body_rot'][self.anchor_index],root_local_ang_vel=state['root_local_ang_vel'],future_refs=future,anchor_body_index=self.anchor_index,num_dofs=29,prev_actions=self.prev_targets)
  out=self._infer(feed,'control');raw=out[1][0].copy();q=np.asarray(future_q,dtype=float)[0].copy();dq=np.asarray(future_dq,dtype=float)[0].copy();bias=self.data.qfrc_bias[self.addresses].copy();applied=applied_targets(raw,q,dq,bias,self.kp,self.kd,self.arm_indices)
  if not np.array_equal(applied[self.nonarm_indices],raw[self.nonarm_indices]):raise RuntimeError('Nonarm actor control altered')
  self.hybrid_rows.append({'control_index':self.control_index,'sim_time_s':self.sim_time_s,'q_ref':q.tolist(),'dq_ref':dq.tolist(),'qfrc_bias_joint':bias.tolist(),'kp':self.kp.tolist(),'kd':self.kd.tolist(),'raw_graph_PD':raw.tolist(),'applied_PD':applied.tolist(),'qpos_at_control':self.data.qpos.tolist(),'qvel_at_control':self.data.qvel.tolist()})
  self.prev_targets=applied.copy();self.targets.append(applied.copy());self.data.ctrl[:]=applied
  force_peak=np.zeros(29);actuator_peak=np.zeros(29);cap_violation=False
  for _ in range(20):
   self.mj.mj_step(self.model,self.data);self.step_count+=1;self.min_height=min(self.min_height,float(self.data.qpos[2]));up_z=root_up_z(self.data.qpos);self.min_root_up_z=min(self.min_root_up_z,up_z);finite=bool(all(np.isfinite(v).all() for v in (self.data.qpos,self.data.qvel,self.data.qacc,self.data.ctrl,self.data.actuator_force,self.data.qfrc_actuator)));warnings=[int(w.number) for w in self.data.warning];self.substeps.append([self.step_count,self.sim_time_s,float(self.data.qpos[2]),self.data.ncon,finite,warnings==self.warning_initial])
   generalized=self.data.qfrc_actuator[self.addresses];force_peak=np.maximum(force_peak,np.abs(generalized));actuator_peak=np.maximum(actuator_peak,np.abs(self.data.actuator_force))
   for i,name in enumerate(self.joint_names):
    j=int(self.model.actuator_trnid[i,0]);lo,hi=self.model.jnt_actfrcrange[j]
    if self.model.jnt_actfrclimited[j] and not lo-1e-5<=generalized[i]<=hi+1e-5:cap_violation=True
   stop_reason=('nonfinite_state' if not finite else 'root_height_below_0p6' if self.data.qpos[2]<.6 else 'root_up_z_below_0p7' if up_z<.7 else 'mujoco_warning_change' if warnings!=self.warning_initial else 'aggregate_joint_force_cap_violation' if cap_violation else None)
   if stop_reason is not None:
    self.hybrid_force_rows.append({'control_index':self.control_index,'steps_completed_in_chunk_control':self.step_count%20 or 20,'peak_abs_joint_qfrc_actuator':force_peak.tolist(),'peak_abs_individual_actuator_force':actuator_peak.tolist(),'joint_caps_observed_compliant':not cap_violation,'stop_reason':stop_reason,'min_root_up_z':self.min_root_up_z});self.stopped=True
    if finite:self.mj.mj_forward(self.model,self.data);self.forward_count+=1
    raise RuntimeError('Actual physical finite/rootheight/warning stop')
  self.hybrid_force_rows.append({'control_index':self.control_index,'steps_completed_in_chunk_control':20,'peak_abs_joint_qfrc_actuator':force_peak.tolist(),'peak_abs_individual_actuator_force':actuator_peak.tolist(),'joint_caps_observed_compliant':True,'stop_reason':None,'min_root_up_z':self.min_root_up_z})
  self.mj.mj_forward(self.model,self.data);self.forward_count+=1;self.control_index+=1;self._record_state();self._cached_observation=None
 def closure_sha256(self):
  parent=super().closure_sha256();config=getattr(self,'hybrid_config',None);rows=getattr(self,'hybrid_rows',[]);forces=getattr(self,'hybrid_force_rows',[])
  return hashlib.sha256(parent.encode()+canonical({'mode':MODE,'config':config,'rows':rows,'force_observations':forces,'min_root_up_z':getattr(self,'min_root_up_z',None)})).hexdigest()
 def export(self,directory):
  super().export(directory);directory=Path(directory)
  rows=self.hybrid_rows;keys=('q_ref','dq_ref','qfrc_bias_joint','kp','kd','raw_graph_PD','applied_PD','qpos_at_control','qvel_at_control')
  np.savez_compressed(directory/'hybrid_control.npz',**{key:np.asarray([row[key] for row in rows]) for key in keys})
  (directory/'hybrid_config.json').write_bytes(canonical(self.hybrid_config));(directory/'hybrid_force_telemetry.json').write_bytes(canonical(self.hybrid_force_rows))
  meta=json.loads((directory/'backend.json').read_bytes());meta.update(schema='pulsemorph.embodied_hybrid_backend.v1',controller=MODE,arm_control='Classical direct localPD+velocityfeedforward+sampledbiascompensation, notlearnedarmchoreography',raw_graph_outputs_preserved=True,min_root_up_z=self.min_root_up_z,limited_physical_gate=self.hybrid_config['limited_physical_gate'],processed_history='previousAPPLIEDPDtargets includingarmservo, float32',hybrid_config_sha256=hashlib.sha256((directory/'hybrid_config.json').read_bytes()).hexdigest(),hybrid_control_sha256=hashlib.sha256((directory/'hybrid_control.npz').read_bytes()).hexdigest(),hybrid_force_telemetry_sha256=hashlib.sha256((directory/'hybrid_force_telemetry.json').read_bytes()).hexdigest());(directory/'backend.json').write_bytes(canonical(meta))
