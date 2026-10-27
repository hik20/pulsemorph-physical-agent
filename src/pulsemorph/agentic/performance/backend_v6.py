"""Static vertical contact-wrench feedforward experiment; actual root stays free.
Contact forces here are predictions used ONLY to compute joint actuator commands.
No external force, actual bias mutation, constraint, teleport or cap relaxation.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from .backend_v5 import PerformanceBackend as StanceBackend,MAX_REFERENCE_FK
from .stance_ik_v1 import solve,body_tasks,StanceIKError
from .backend_v2 import direct_targets
from .backend_v1 import validate_future
from pulsemorph.agentic.embodied.hybrid_arm_backend_v1 import root_up_z
from pulsemorph.agentic.embodied.runtime_v1 import canonical


def vertical_support(jacobian_z,bias):
    j=np.asarray(jacobian_z,float);b=np.asarray(bias,float)
    if j.ndim!=2 or j.shape[1]!=35 or not 1<=j.shape[0]<=64 or b.shape!=(35,) or not np.isfinite(j).all() or not np.isfinite(b).all():raise ValueError('Finite1..64contactJacobians and35bias required')
    a=j[:,:6].T;forces,residual=nnls(a,b[:6],maxiter=100)
    tau=b[6:]-j[:,6:].T@forces
    if not np.isfinite(tau).all() or not np.isfinite(forces).all():raise RuntimeError('Nonfinite contact support solution')
    return tau,{'vertical_forces_N':forces.tolist(),'predicted_root_wrench':(a@forces).tolist(),'requested_bias_root_wrench':b[:6].tolist(),'root_wrench_residual':(a@forces-b[:6]).tolist(),'root_wrench_residual_norm':float(residual)}


class PerformanceBackend(StanceBackend):
    def __init__(self,root,*,authority):
        p=str(Path(__file__).resolve())
        if authority.get('source_pins',{}).get(p)!=hashlib.sha256(Path(p).read_bytes()).hexdigest():raise ValueError('Pin contactfeedforward backend')
        super().__init__(root,authority=authority)
        self.support_rows=[];self.contact_jacobian_calls=0

    def activate_stance(self,initial_requested_q):
        super().activate_stance(initial_requested_q)
        self.hybrid_config.update(mode='double_stance_IK_contact_feedforward29_v1',
            formula='ctrl29=projectedq+(kd/kp)*projecteddq+(actualbias_joint-JcontactZ_joint.T*fz)/kp')
        self.performance_contract.update(schema='pulsemorph.performance_backend.v6',
            actuation='29classicalservo with staticnonnegativeverticalcontactwrench feedforward; actor diagnostic')

    def _contact_feedforward(self):
        footids={int(a[1]) for a in self.stance['anchors']};rows=[];points=[]
        for i in range(self.data.ncon):
            c=self.data.contact[i];a=int(self.model.geom_bodyid[c.geom1]);b=int(self.model.geom_bodyid[c.geom2])
            foot=a if a in footids and b==0 else b if b in footids and a==0 else None
            if foot is None:continue
            if len(rows)>=64:raise RuntimeError('Morethan64footplanecontacts unsupported')
            jp=np.zeros((3,self.model.nv));jr=jp.copy()
            self.mj.mj_jac(self.model,self.data,jp,jr,c.pos,foot);self.contact_jacobian_calls+=1
            rows.append(jp[2]);points.append(np.asarray(c.pos).tolist())
        if not rows:raise RuntimeError('Noactualfootplanecontact for staticfeedforward')
        tau,report=vertical_support(rows,self.data.qfrc_bias.copy())
        report.update(control_index=self.control_index,contact_points_world_m=points,tau_ff_joint=tau.tolist(),
            actual_bias_joint=self.data.qfrc_bias[self.addresses].tolist(),
            qualification='Verticalonlystaticwrench approximation; no horizontal/friction/contactdynamic force guarantee; held20substeps')
        self.support_rows.append(report)
        return tau,report
    def advance(self,future_q,future_dq):
        if self.stance is None:return StanceBackend.advance(self,future_q,future_dq)
        self._check_authority()
        if self.stopped or self.control_index>=self.authority['max_control_steps']:raise RuntimeError('Stance stopped/budget')
        q,dq=validate_future(future_q,future_dq,self.joint_ranges)
        s=self.stance;delta=q[0]-np.asarray(s['requested_baseline'])
        desired=np.asarray(s['actual_baseline'])+delta
        com,z=body_tasks(delta,self.joint_names,s['com'],s['root_z'])
        args=dict(unknown_dofs=s['unknown_dofs'],leg_indices=s['leg_indices'],desired_q=desired,
            anchors=[(a,b,np.asarray(p),np.asarray(r)) for a,b,p,r in s['anchors']],
            com_xy=com,root_z=z,root_rotation=np.asarray(s['root_rotation']),joint_ranges=np.asarray(self.joint_ranges))
        remaining=MAX_REFERENCE_FK-self.reference_forward_count
        if remaining<=0:
            self.stopped=True;raise RuntimeError('Stance reference FK budget exhausted')
        args['max_iterations']=min(12,remaining)
        try:projected,report=solve(self.mj,self.model,self.reference_data,**args)
        except StanceIKError as exc:
            self.reference_forward_count+=exc.report['reference_FK'];self.stopped=True
            self.stance_rows.append({'control_index':self.control_index,'status':'IK_failed','report':exc.report})
            raise
        self.reference_forward_count+=report['reference_FK']
        velocity=(projected-np.asarray(s['previous_projected_q']))/.02
        if np.any(np.abs(velocity)>2):
            self.stopped=True;raise RuntimeError('Projected reference rate exceeds2rad/s; no clamp')
        self.stance_rows.append({'control_index':self.control_index,'authored_q':q[0].tolist(),
            'desired_joint_q':desired.tolist(),'projected_q':projected.tolist(),'projected_dq':velocity.tolist(),
            'requested_com_xy':com.tolist(),'requested_root_z':z,'report':report})
        s['previous_projected_q']=projected.tolist()
        return self._advance_contact(np.tile(projected,(4,1)),np.tile(velocity,(4,1)))

    def _advance_contact(self,future_q,future_dq):
        future_q,future_dq=validate_future(future_q,future_dq,self.joint_ranges)
        if self.stopped or self.control_index>=self.authority['max_control_steps']:raise RuntimeError('Hybrid episode stopped or controlbudget exhausted')
        self._check_authority();self._check_hybrid();self._cached_observation=None
        from deployment.tracker_inputs import build_tracker_onnx_inputs,align_future_motion_refs
        state=self.upstream.read_robot_state(self.data,self.anchor_index,self.root_index)
        future={'dof_pos':np.asarray(future_q,np.float32),'dof_vel':np.asarray(future_dq,np.float32),'body_rot':np.tile(self.anchor_target,(4,len(self.cfg['robot']['body_names']),1))};future=align_future_motion_refs(future,self.alignment)
        feed=build_tracker_onnx_inputs(onnx_input_names=list(self.INPUTS),onnx_name_to_key=self.cfg['_runtime']['onnx_name_to_in_key'],dof_pos=state['dof_pos'],dof_vel=state['dof_vel'],anchor_rot=state['body_rot'][self.anchor_index],root_local_ang_vel=state['root_local_ang_vel'],future_refs=future,anchor_body_index=self.anchor_index,num_dofs=29,prev_actions=self.prev_targets)
        out=self._infer(feed,'control');raw=out[1][0].copy();q=np.asarray(future_q,dtype=float)[0].copy();dq=np.asarray(future_dq,dtype=float)[0].copy();bias=self.data.qfrc_bias[self.addresses].copy();tau_ff,support=self._contact_feedforward();applied=direct_targets(raw,q,dq,tau_ff,self.kp,self.kd)
        self.hybrid_rows.append({'control_index':self.control_index,'sim_time_s':self.sim_time_s,'q_ref':q.tolist(),'dq_ref':dq.tolist(),'qfrc_bias_joint':bias.tolist(),'contact_support_tau_ff':tau_ff.tolist(),'kp':self.kp.tolist(),'kd':self.kd.tolist(),'raw_graph_PD':raw.tolist(),'applied_PD':applied.tolist(),'qpos_at_control':self.data.qpos.tolist(),'qvel_at_control':self.data.qvel.tolist()})
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
        parent=super().closure_sha256()
        return hashlib.sha256(parent.encode()+canonical({'support_rows':getattr(self,'support_rows',[]),'contact_jacobian_calls':getattr(self,'contact_jacobian_calls',0)})).hexdigest()

    def export(self,directory):
        super().export(directory);p=Path(directory)
        proof={'schema':'pulsemorph.performance_contact_support.v1','rows':self.support_rows,'contact_jacobian_calls':self.contact_jacobian_calls,'actual_external_forces_applied':False}
        (p/'contact_support.json').write_bytes(canonical(proof))
        meta=json.loads((p/'backend.json').read_bytes());meta.update(schema='pulsemorph.performance_backend.v6',contact_jacobian_calls=self.contact_jacobian_calls,
            contact_support_sha256=hashlib.sha256((p/'contact_support.json').read_bytes()).hexdigest())
        (p/'backend.json').write_bytes(canonical(meta))
