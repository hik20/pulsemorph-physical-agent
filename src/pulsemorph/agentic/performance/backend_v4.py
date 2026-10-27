"""Experimental18 direct servos +11 actor joints + declared rootXY carrier.
Direct hips-pitch/knees may improve squat tracking but stance is unverified.
Actor ankles/hiproll/hipyaw/waist remain responsible for balance response.
No authored q/dq change, external force, constraint, cap relaxation or clamp.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .backend_v3 import PerformanceBackend as OuterBackend,CONFIG,tilt_request,applied_anchor
from .backend_v1 import validate_future
from .backend_v2 import direct_targets
from pulsemorph.agentic.embodied.hybrid_arm_backend_v1 import root_up_z
from pulsemorph.agentic.embodied.runtime_v1 import canonical
MODE='classical18_arms_hippitch_knee_actor11_rootXY_carrier_v1'


def mixed_targets(raw,q,dq,bias,kp,kd,indices):
    indices=np.asarray(indices)
    if indices.shape!=(18,) or indices.dtype.kind not in 'iu' or len(set(indices.tolist()))!=18 or np.any(indices<0) or np.any(indices>=29):
        raise ValueError('Exact18 distinct servo indices required')
    servo=direct_targets(raw,q,dq,bias,kp,kd)
    result=np.asarray(raw).copy();result[indices]=servo[indices]
    return result


class PerformanceBackend(OuterBackend):
    def __init__(self,root,*,authority):
        own=str(Path(__file__).resolve())
        if type(authority) is not dict or authority.get('source_pins',{}).get(own)!=hashlib.sha256(Path(own).read_bytes()).hexdigest():
            raise ValueError('Authority must pin mixed18 backend')
        super().__init__(root,authority=authority)
        extra=['left_hip_pitch_joint','right_hip_pitch_joint','left_knee_joint','right_knee_joint']
        if any(name not in self.joint_names for name in extra):raise RuntimeError('Expected exact G1 pitch/knee names')
        self.servo_indices=sorted(self.arm_indices+[self.joint_names.index(name) for name in extra])
        self.actor_controlled_indices=[i for i in range(29) if i not in self.servo_indices]
        if len(self.servo_indices)!=18 or len(self.actor_controlled_indices)!=11:raise RuntimeError('Exact18/11 roles required')
        self.hybrid_config.update(schema='pulsemorph.performance_mixed18_config.v1',mode=MODE,
            servo_indices=self.servo_indices,servo_joint_names=[self.joint_names[i] for i in self.servo_indices],
            actor_controlled_indices=self.actor_controlled_indices,
            actor_controlled_joint_names=[self.joint_names[i] for i in self.actor_controlled_indices],
            nonarm_indices=self.actor_controlled_indices,
            formula='ctrl_SERVO18=q_ref+(kd/kp)*dq_ref+qfrc_bias/kp; remaining11 rawactorPD unchanged')
        self.performance_contract.update(schema='pulsemorph.performance_backend.v4',
            actuation='18 directservo:14arms+2hippitch+2knees;11actor:ankles/hiproll/hipyaw/waist',
            qualifications=self.performance_contract['qualifications']+[
                'Mixed18 stance/squat tracking unproven until physical pilot; not guaranteed balance.'])
    def advance(self, future_q, future_dq):
        self._check_authority()
        q, dq = validate_future(future_q, future_dq, self.joint_ranges)
        if self.stopped or self.control_index >= self.authority['max_control_steps']:
            raise RuntimeError('Performance episode stopped or controlbudget exhausted')
        if self.outer_config != CONFIG:
            raise RuntimeError('Outer-loop configuration changed')
        nominal = self.anchor_target.copy()
        error = self.data.qpos[:2] - self.root_xy_start
        velocity = self.data.qvel[:2].copy()
        next_tilt, raw = tilt_request(error, velocity, self.outer_tilt, self.outer_config)
        carrier = applied_anchor(nominal, next_tilt)
        self.outer_rows.append({'control_index': self.control_index,
            'sim_time_s': self.sim_time_s, 'root_error_world_xy_m': error.tolist(),
            'root_velocity_world_xy_m_s': velocity.tolist(),
            'previous_roll_pitch_rad': self.outer_tilt.tolist(),
            'raw_roll_pitch_rad': raw.tolist(), 'applied_roll_pitch_rad': next_tilt.tolist(),
            'nominal_anchor_xyzw': nominal.tolist(), 'applied_actor_anchor_xyzw': carrier.tolist()})
        self.outer_tilt = next_tilt
        self.anchor_target = carrier
        try:
            return self._advance_mixed(q, dq)
        finally:
            self.anchor_target = nominal

    def _advance_mixed(self,future_q,future_dq):
        if self.stopped or self.control_index>=self.authority['max_control_steps']:raise RuntimeError('Hybrid episode stopped or controlbudget exhausted')
        self._check_hybrid();self._cached_observation=None
        from deployment.tracker_inputs import build_tracker_onnx_inputs,align_future_motion_refs
        state=self.upstream.read_robot_state(self.data,self.anchor_index,self.root_index)
        future={'dof_pos':np.asarray(future_q,np.float32),'dof_vel':np.asarray(future_dq,np.float32),'body_rot':np.tile(self.anchor_target,(4,len(self.cfg['robot']['body_names']),1))};future=align_future_motion_refs(future,self.alignment)
        feed=build_tracker_onnx_inputs(onnx_input_names=list(self.INPUTS),onnx_name_to_key=self.cfg['_runtime']['onnx_name_to_in_key'],dof_pos=state['dof_pos'],dof_vel=state['dof_vel'],anchor_rot=state['body_rot'][self.anchor_index],root_local_ang_vel=state['root_local_ang_vel'],future_refs=future,anchor_body_index=self.anchor_index,num_dofs=29,prev_actions=self.prev_targets)
        out=self._infer(feed,'control');raw=out[1][0].copy();q=np.asarray(future_q,dtype=float)[0].copy();dq=np.asarray(future_dq,dtype=float)[0].copy();bias=self.data.qfrc_bias[self.addresses].copy();applied=mixed_targets(raw,q,dq,bias,self.kp,self.kd,self.servo_indices)
        if not np.array_equal(applied[self.actor_controlled_indices],raw[self.actor_controlled_indices]):raise RuntimeError('Actor-controlled11 targets altered')
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
        p=Path(directory)/'backend.json';data=json.loads(p.read_bytes())
        data.update(schema='pulsemorph.performance_backend.v4',controller=MODE,
            servo_indices=self.servo_indices,actor_controlled_indices=self.actor_controlled_indices,
            arm_control='18 classical servos:14arms+hippitch/knees; remaining11 actor outputs unchanged')
        p.write_bytes(canonical(data))
