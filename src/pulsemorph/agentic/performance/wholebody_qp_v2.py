"""Small bounded equality/inequality constrained inverse-dynamics QP.
World foot wrench order is forceXYZ,momentXYZ at the ankle BODY origin.
No simulated force is applied by this module.
"""
import numpy as np
MIN_LOCAL_FZ_N = 15.0

class WBCInfeasible(RuntimeError):
    def __init__(self,report):
        super().__init__('Whole-body QP infeasible or bounded solver did not converge')
        self.report=report

def constrained_ls(a,target,equality,rhs,inequality,upper,*,max_iterations=80):
    from scipy.optimize import minimize,lsq_linear
    a=np.asarray(a,float);target=np.asarray(target,float);e=np.asarray(equality,float);r=np.asarray(rhs,float);g=np.asarray(inequality,float);u=np.asarray(upper,float)
    if a.ndim!=2:raise ValueError('Matrixobjective required')
    n=a.shape[1]
    if target.shape!=(a.shape[0],) or e.ndim!=2 or e.shape[1]!=n or r.shape!=(len(e),) or g.ndim!=2 or g.shape[1]!=n or u.shape!=(len(g),) or not all(np.isfinite(v).all() for v in (a,target,e,r,g,u)):
        raise ValueError('Finite compatible QP arrays required')
    if type(max_iterations)is not int or not 1<=max_iterations<=80:raise ValueError('QP iteration budget1..80')
    h=a.T@a+np.eye(n)*1e-9;c=a.T@target
    # SLSQP requires independent equality rows. Remove only redundant rows;
    # verify ALL original equalities afterward, including inconsistent ones.
    from scipy.linalg import qr
    if len(e):
        _,tri,piv=qr(e.T,pivoting=True,mode='economic')
        diagonal=np.abs(np.diag(tri));rank=int(np.sum(diagonal>max(np.max(diagonal,initial=0)*1e-10,1e-12)))
        selected=piv[:rank];er=e[selected];rr=r[selected]
    else:er=e;rr=r
    matrix=np.block([[h,er.T],[er,np.zeros((len(er),len(er)))]])
    x0=np.linalg.lstsq(matrix,np.r_[c,rr],rcond=1e-12)[0][:n]
    constraints=[]
    if len(er):constraints.append({'type':'eq','fun':lambda x:er@x-rr,'jac':lambda x:er})
    if len(g):constraints.append({'type':'ineq','fun':lambda x:u-g@x,'jac':lambda x:-g})
    result=minimize(lambda x:.5*x@h@x-c@x,x0,jac=lambda x:h@x-c,
        method='SLSQP',constraints=constraints,options={'maxiter':max_iterations,'ftol':1e-9,'disp':False})
    x=result.x
    eq_res=float(np.max(np.abs(e@x-r),initial=0));viol=g@x-u
    primal=float(max(np.max(viol,initial=0),0));active=np.flatnonzero(np.abs(viol)<=1e-5)
    k=np.vstack([e,g[active]]) if len(active) else e
    gradient=h@x-c
    if len(k):
        multipliers=lsq_linear(k.T,-gradient,bounds=(np.r_[np.full(len(e),-np.inf),np.zeros(len(active))],np.full(len(k),np.inf)),tol=1e-12,max_iter=80)
        stationarity=float(np.max(np.abs(gradient+k.T@multipliers.x),initial=0))
        minimum_dual=float(np.min(multipliers.x[len(e):],initial=0))
    else:stationarity=float(np.max(np.abs(gradient),initial=0));minimum_dual=0.
    report={'solver':'scipy_SLSQP_analytic_gradient_linear_Jacobian','status':'converged' if result.success else 'solver_not_converged',
        'solver_status':int(result.status),'iterations':int(result.nit),'active_inequalities':active.tolist(),
        'equality_residual':eq_res,'max_inequality_violation':primal,
        'stationarity_residual':stationarity,'minimum_active_dual':minimum_dual}
    if not result.success or not np.isfinite(x).all() or eq_res>1e-5 or primal>1e-6 or stationarity>1e-4 or minimum_dual< -1e-8:
        raise WBCInfeasible(report)
    return x,report

def wrench_inequalities(rotations,mu=.6):
    """Local positiveZ support, friction pyramid, CoP box and yaw moment.
    Contact plane z=-.035 relative ankle origin; x[-.05,.12],y±.025.
    Local moment at sole = ankle moment - (0,0,-.035) cross force.
    """
    rows=[]
    for foot,rotation in enumerate(rotations):
        rt=np.asarray(rotation,float).T
        if rt.shape!=(3,3) or not np.isfinite(rt).all():raise ValueError('Finitefootrotation')
        f=np.zeros((3,12));m=f.copy();f[:,foot*6:foot*6+3]=rt;m[:,foot*6+3:foot*6+6]=rt
        fx,fy,fz=f;mx,my,mz=m
        # Translate ankle wrench to a sole plane .035m below ankle.
        sole_mx=mx-.035*fy;sole_my=my+.035*fx
        rows.extend([-fz,fx-(mu/np.sqrt(2))*fz,-fx-(mu/np.sqrt(2))*fz,fy-(mu/np.sqrt(2))*fz,-fy-(mu/np.sqrt(2))*fz,
                     sole_mx-.025*fz,-sole_mx-.025*fz,
                     sole_my-.05*fz,-sole_my-.12*fz,
                     mz-.03*fz,-mz-.03*fz])
    return np.asarray(rows)

def solve_wholebody(mass,bias,jacobian,foot_acceleration,desired_acceleration,force_prior,rotations,torque_limits):
    m=np.asarray(mass,float);b=np.asarray(bias,float);j=np.asarray(jacobian,float)
    desired=np.asarray(desired_acceleration,float);foot=np.asarray(foot_acceleration,float);prior=np.asarray(force_prior,float);limits=np.asarray(torque_limits,float)
    if m.shape!=(35,35) or b.shape!=(35,) or j.shape!=(12,35) or desired.shape!=(35,) or foot.shape!=(12,) or prior.shape!=(12,) or limits.shape!=(29,2):raise ValueError('ExactG1WBCshapes')
    dynamics=np.c_[m,-j.T]
    e=np.vstack([dynamics[:6],np.c_[j,np.zeros((12,12))]])
    rhs=np.r_[-b[:6],foot]
    weights=np.r_[np.full(6,5.),np.ones(29)]
    objective=np.vstack([np.c_[np.diag(weights),np.zeros((35,12))],np.c_[np.zeros((12,35)),np.eye(12)*.01]])
    target=np.r_[desired*weights,prior*.01]
    friction=np.c_[np.zeros((22,35)),wrench_inequalities(rotations)]
    torque=dynamics[6:]
    friction_upper=np.zeros(22);friction_upper[[0,11]]=-MIN_LOCAL_FZ_N
    g=np.vstack([friction,torque,-torque]);upper=np.r_[friction_upper,limits[:,1]-b[6:],-limits[:,0]+b[6:]]
    x,report=constrained_ls(objective,target,e,rhs,g,upper)
    tau=torque@x+b[6:]
    report.update(qdd=x[:35].tolist(),world_foot_wrenches=x[35:].tolist(),torque=tau.tolist(),
                  minimum_local_normal_force_N=MIN_LOCAL_FZ_N,
                  local_normal_forces_N=[float(np.asarray(rotations[i])[:,2]@x[35+i*6:38+i*6]) for i in range(2)],
                  root_dynamics_residual=(dynamics[:6]@x+b[:6]).tolist(),
                  foot_acceleration_residual=(j@x[:35]-foot).tolist())
    return tau,report
