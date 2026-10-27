"""Bounded reference-only double-stance numerical IK; no actual state writes."""
import numpy as np

TOLERANCES={'foot_position_m':.0005,'foot_orientation_rad':.003,
            'com_xy_m':.0005,'root_z_m':.0005,'root_orientation_rad':.003}

def residual_passed(residuals):
    return all(value <= (TOLERANCES['foot_position_m'] if key.endswith('_position_m') else
                         TOLERANCES['foot_orientation_rad'] if key in ('left_orientation_rad','right_orientation_rad') else
                         TOLERANCES[key]) for key,value in residuals.items())

class StanceIKError(RuntimeError):
    def __init__(self, report):
        super().__init__('Double-stance reference IK residual/limit failure')
        self.report=report

def rotation_error(target,current):
    relative=np.asarray(target)@np.asarray(current).T
    cosine=float(np.clip((np.trace(relative)-1)/2,-1,1));angle=float(np.arccos(cosine))
    skew=np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])
    if angle<1e-7:return .5*skew
    if np.pi-angle<1e-5:
        values,vectors=np.linalg.eigh((relative+relative.T)/2)
        axis=vectors[:,np.argmax(values)]
        if np.dot(axis,skew)<0:axis=-axis
        return axis*angle
    return skew*(angle/(2*np.sin(angle)))

def damped_step(jacobian,error,damping=.01):
    j=np.asarray(jacobian,float);e=np.asarray(error,float)
    if j.ndim!=2 or e.shape!=(j.shape[0],) or not np.isfinite(j).all() or not np.isfinite(e).all():raise ValueError('Finite Jacobian/error required')
    return j.T@np.linalg.solve(j@j.T+damping*damping*np.eye(len(e)),e)

def body_tasks(delta,names,baseline_com,baseline_z):
    d=dict(zip(names,np.asarray(delta,float)))
    mean=lambda part:sum(d[f'{side}_{part}_joint'] for side in ('left','right'))/2
    xy=np.array([.05*mean('hip_pitch'),.25*mean('hip_roll')+.10*d['waist_roll_joint']])
    xy=np.clip(xy,-.06,.06)
    z=baseline_z+float(np.clip(-.12*mean('knee'),-.10,.10))
    return np.asarray(baseline_com)[:2]+xy,z

def solve(mj,model,data,*,unknown_dofs,leg_indices,desired_q,anchors,com_xy,root_z,root_rotation,joint_ranges,max_iterations=12):
    if type(max_iterations)is not int or not 1<=max_iterations<=12:raise ValueError('IK iterations1..12')
    unknown=np.asarray(unknown_dofs,int);desired=np.asarray(desired_q,float)
    if unknown.shape!=(18,) or desired.shape!=(29,) or not np.isfinite(desired).all():raise ValueError('Exact18unknown/29targets')
    fixed=[i for i in range(29) if i not in leg_indices]
    data.qpos[7+np.asarray(fixed)]=desired[fixed]
    evaluations=0
    for iteration in range(max_iterations):
        mj.mj_forward(model,data);evaluations+=1
        rows=[];errors=[];res={}
        for label,body,target_pos,target_rot in anchors:
            jp=np.zeros((3,model.nv));jr=jp.copy();mj.mj_jacBody(model,data,jp,jr,body)
            ep=target_pos-data.xpos[body];er=rotation_error(target_rot,data.xmat[body].reshape(3,3))
            rows.extend([jp[:,unknown]*4,jr[:,unknown]*2]);errors.extend([ep*4,er*2])
            res[label+'_position_m']=float(np.linalg.norm(ep));res[label+'_orientation_rad']=float(np.linalg.norm(er))
        jc=np.zeros((3,model.nv));mj.mj_jacSubtreeCom(model,data,jc,1)
        ec=np.asarray(com_xy)-data.subtree_com[1,:2]
        ez=root_z-data.qpos[2];er=rotation_error(root_rotation,data.xmat[1].reshape(3,3))
        jp=np.zeros((3,model.nv));jr=jp.copy();mj.mj_jacBody(model,data,jp,jr,1)
        rows.extend([jc[:2,unknown],np.eye(model.nv)[2:3,unknown]*2,jr[:,unknown]])
        errors.extend([ec,np.array([ez*2]),er])
        res.update(com_xy_m=float(np.linalg.norm(ec)),root_z_m=abs(float(ez)),root_orientation_rad=float(np.linalg.norm(er)))
        q=data.qpos[7:];valid=bool(np.all(q>=joint_ranges[:,0]) and np.all(q<=joint_ranges[:,1]))
        passed=valid and residual_passed(res)
        if passed:return q.copy(),{'status':'converged','iterations':iteration+1,'reference_FK':evaluations,'residuals':res,'tolerances':dict(TOLERANCES)}
        if iteration==max_iterations-1:break
        step=damped_step(np.vstack(rows),np.concatenate(errors));step*=min(1.,.08/max(np.max(np.abs(step)),1e-12))
        velocity=np.zeros(model.nv);velocity[unknown]=step
        # Shrink an IK update until joint limits admit it; never clamp coordinates.
        original=data.qpos.copy();accepted=False
        for scale in (1.,.5,.25,.125,.0625):
            trial=original.copy();mj.mj_integratePos(model,trial,velocity,scale)
            if np.all(trial[7:]>=joint_ranges[:,0]) and np.all(trial[7:]<=joint_ranges[:,1]):
                data.qpos[:]=trial;accepted=True;break
        if not accepted:break
    raise StanceIKError({'status':'unavailable','reference_FK':evaluations,'residuals':res,'joint_limits_passed':valid})
