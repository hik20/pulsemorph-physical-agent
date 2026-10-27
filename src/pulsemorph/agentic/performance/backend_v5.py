"""Hybrid settle, then experimental double-stance IK + all29 classical servo.
Construction/activation/advance require separate explicit root execution authority.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from .backend_v3 import PerformanceBackend as SettlingBackend
from .backend_v2 import PerformanceBackend as DirectBackend
from .backend_v1 import validate_future
from .stance_ik_v1 import solve,body_tasks,StanceIKError
from pulsemorph.agentic.embodied.runtime_v1 import canonical
MAX_REFERENCE_FK=2400  # This candidate is a finite stance pilot, not full-show admission.

class PerformanceBackend(SettlingBackend):
    def __init__(self,root,*,authority):
        p=str(Path(__file__).resolve())
        if authority.get('source_pins',{}).get(p)!=hashlib.sha256(Path(p).read_bytes()).hexdigest():raise ValueError('Pin stance backend')
        super().__init__(root,authority=authority)
        self.stance=None;self.stance_rows=[];self.reference_forward_count=0
        self.stance_activation_control_index=None

    def activate_stance(self,initial_requested_q):
        self._check_authority()
        if self.stance is not None or self.stopped:raise RuntimeError('Stance activation once only on live body')
        requested=np.asarray(initial_requested_q,float)
        if requested.shape!=(29,) or not np.isfinite(requested).all():raise ValueError('Finite29 requested baseline')
        self.reference_data=copy.copy(self.data)
        if np.shares_memory(self.reference_data.qpos,self.data.qpos):raise RuntimeError('Reference copy alias')
        legs=[i for i,n in enumerate(self.joint_names) if any(x in n for x in ('hip_','knee_','ankle_'))]
        if len(legs)!=12:raise RuntimeError('Exact12legs required')
        self.stance_activation_control_index=self.control_index
        self.stance={'requested_baseline':requested.tolist(),'actual_baseline':self.data.qpos[7:].tolist(),
            'leg_indices':legs,'unknown_dofs':list(range(6))+[6+i for i in legs],
            'com':self.data.subtree_com[1].tolist(),'root_z':float(self.data.qpos[2]),
            'root_rotation':self.data.xmat[1].reshape(3,3).tolist(),'anchors':[],
            'previous_projected_q':self.data.qpos[7:].tolist(),
            'qualification':'Fixed ankle-roll BODY frames, not contact sole or friction guarantee; actual free root untouched.'}
        for side in ('left','right'):
            body=self.mj.mj_name2id(self.model,self.mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
            if body<0:raise RuntimeError('Foot body missing')
            self.stance['anchors'].append([side,body,self.data.xpos[body].tolist(),self.data.xmat[body].reshape(3,3).tolist()])
        self.hybrid_config.update(mode='double_stance_IK_classical29_v1',servo_indices=list(range(29)),actor_controlled_indices=[],
            formula='ALL29 projectedq+(kd/kp)*projecteddq+actualboundarybias/kp',
            qualifications=['Actor raw outputs diagnostic only after stance activation.','Reference-only IK; no actual freebase/root writes or external forces.'])
        self.performance_contract.update(schema='pulsemorph.performance_backend.v5',
            actuation='Hybrid settle then double-stance IK projected29 classical servo; actor diagnostic after activation')

    def advance(self,future_q,future_dq):
        if self.stance is None:return super().advance(future_q,future_dq)
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
        return DirectBackend.advance(self,np.tile(projected,(4,1)),np.tile(velocity,(4,1)))

    def closure_sha256(self):
        parent=super().closure_sha256();ref=None
        if hasattr(self,'reference_data'):
            kind=self.mj.mjtState.mjSTATE_INTEGRATION;a=np.empty(self.mj.mj_stateSize(self.model,kind))
            self.mj.mj_getState(self.model,self.reference_data,a,kind);ref=hashlib.sha256(a.tobytes()).hexdigest()
        return hashlib.sha256(parent.encode()+canonical({'stance':getattr(self,'stance',None),
            'reference_state':ref,'rows':getattr(self,'stance_rows',[]),'reference_FK':getattr(self,'reference_forward_count',0)})).hexdigest()

    def export(self,directory):
        super().export(directory);p=Path(directory)
        proof={'schema':'pulsemorph.performance_stance_IK.v1','stance':self.stance,'rows':self.stance_rows,
            'reference_FK':self.reference_forward_count,'actual_root_overwritten':False,
            'max_reference_FK':MAX_REFERENCE_FK,
            'projection':'Original authored jointq/dq remains logged; actualservo uses IKprojectedq and finite-differencevelocity, repeatedacross4futureframes; notidentical tooriginalcommonreference',
            'activation_control_index':self.stance_activation_control_index,
            'control_roles':{'before_activation':'14arms directservo;15actor;rootXYcarrier',
                             'from_activation':'29projectedjoint directservo;actor diagnostic only'},
            'actual_forward_calls':self.forward_count,
            'reference_forward_calls_separate_from_actual':True}
        (p/'stance_ik.json').write_bytes(canonical(proof))
        data=json.loads((p/'backend.json').read_bytes());data.update(schema='pulsemorph.performance_backend.v5',
            reference_forward_calls=self.reference_forward_count,stance_activation_control_index=self.stance_activation_control_index,
            stance_ik_sha256=hashlib.sha256((p/'stance_ik.json').read_bytes()).hexdigest())
        (p/'backend.json').write_bytes(canonical(data))
