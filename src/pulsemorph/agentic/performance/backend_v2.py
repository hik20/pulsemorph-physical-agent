"""Direct classical all29-joint servo experiment; construction executes motor.
The fixed actor is diagnostic only. No learned leg balance, root constraint,
external forces, target projection or numerical clamp is introduced.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .backend_v1 import PerformanceBackend as TrackerPerformanceBackend, validate_future
from pulsemorph.agentic.embodied.hybrid_arm_backend_v1 import root_up_z
from pulsemorph.agentic.embodied.runtime_v1 import canonical
MODE='classical_all29_PD_velocity_sampled_bias_actor_diagnostic_only_v1'


def direct_targets(raw_graph_pd,q_ref,dq_ref,bias,kp,kd):
 raw=np.asarray(raw_graph_pd)
 if raw.shape!=(29,) or raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Finite float32 raw graph29 required')
 values=[np.asarray(x,dtype=float) for x in (q_ref,dq_ref,bias,kp,kd)]
 if any(x.shape!=(29,) or not np.isfinite(x).all() for x in values):raise ValueError('Finite29 servo arrays required')
 q,dq,b,kp,kd=values
 if np.any(kp<=0) or np.any(kd<0):raise ValueError('Positive kp/nonnegative kd required')
 with np.errstate(over='ignore',invalid='ignore'):
  result=(q+(kd/kp)*dq+b/kp).astype(np.float32)
 if not np.isfinite(result).all():raise ValueError('Servo command overflow')
 return result


class PerformanceBackend(TrackerPerformanceBackend):
 def __init__(self,root,*,authority):
  own=str(Path(__file__).resolve())
  if type(authority) is not dict or authority.get('source_pins',{}).get(own)!=hashlib.sha256(Path(own).read_bytes()).hexdigest():raise ValueError('Authority must pin direct-all29 backend')
  super().__init__(root,authority=authority)
  self.servo_indices=list(range(29));self.actor_controlled_indices=[]
  self.hybrid_config.update(schema='pulsemorph.performance_direct29_config.v1',mode=MODE,
   arm_indices=self.arm_indices,nonarm_indices=[],servo_indices=self.servo_indices,
   servo_joint_names=list(self.joint_names),actor_controlled_indices=[],
   formula='ctrl_ALL29=q_ref+(kd/kp)*dq_ref+qfrc_bias_at_current_joint_dof/kp',
   qualifications=['All29 joints use classical local servo; actor outputs diagnostic only.',
    'No pretrained leg balance actuation remains; stance stability unproven.',
    'Boundary dynamic bias held20substeps; contact force excluded.',
    'Existing individual and aggregate joint caps unchanged; no hardware safety claim.'])
  self.performance_contract.update(schema='pulsemorph.performance_backend.v2',
   actuation='ALL29 classical PD velocityfeedforward and sampled dynamic bias; actor diagnostic only',
   qualifications=list(self.hybrid_config['qualifications']))
 def advance(self,future_q,future_dq):
  future_q,future_dq=validate_future(future_q,future_dq,self.joint_ranges)
  if self.stopped or self.control_index>=self.authority['max_control_steps']:raise RuntimeError('Hybrid episode stopped or controlbudget exhausted')
  self._check_authority();self._check_hybrid();self._cached_observation=None
  from deployment.tracker_inputs import build_tracker_onnx_inputs,align_future_motion_refs
  state=self.upstream.read_robot_state(self.data,self.anchor_index,self.root_index)
  future={'dof_pos':np.asarray(future_q,np.float32),'dof_vel':np.asarray(future_dq,np.float32),'body_rot':np.tile(self.anchor_target,(4,len(self.cfg['robot']['body_names']),1))};future=align_future_motion_refs(future,self.alignment)
  feed=build_tracker_onnx_inputs(onnx_input_names=list(self.INPUTS),onnx_name_to_key=self.cfg['_runtime']['onnx_name_to_in_key'],dof_pos=state['dof_pos'],dof_vel=state['dof_vel'],anchor_rot=state['body_rot'][self.anchor_index],root_local_ang_vel=state['root_local_ang_vel'],future_refs=future,anchor_body_index=self.anchor_index,num_dofs=29,prev_actions=self.prev_targets)
  out=self._infer(feed,'control');raw=out[1][0].copy();q=np.asarray(future_q,dtype=float)[0].copy();dq=np.asarray(future_dq,dtype=float)[0].copy();bias=self.data.qfrc_bias[self.addresses].copy();applied=direct_targets(raw,q,dq,bias,self.kp,self.kd)
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

 def export(self,directory):
  super().export(directory)
  path=Path(directory)/'backend.json';data=json.loads(path.read_bytes())
  data.update(schema='pulsemorph.performance_backend.v2',controller=MODE,
   actor_role='Diagnostic raw outputs only; NO actor output actuates any joint',
   arm_control='All29 classical local PD+velocityfeedforward+sampled dynamic bias',
   servo_indices=self.servo_indices,actor_controlled_indices=[])
  path.write_bytes(canonical(data))
