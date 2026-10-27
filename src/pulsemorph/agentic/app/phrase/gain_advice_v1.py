"""Conditional numerical reference advice, never control or execution authority."""
import copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicHermiteSpline
from ...phrase_program import parse_program,GAIN_FIELDS,compile_program,ProgramCompileError,_joint_limits
from pulsemorph.motion.g1 import profile_from_yaml

METHOD='affine_common_cubic_rational_extrema_v1'
BASE_GAINS=(1.,1.2,.8,.75)
GAIN_MARGIN=1e-7
class AdviceCancelled(RuntimeError):pass
class AdviceTimeout(RuntimeError):pass
def _encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def _sha(data):return hashlib.sha256(data).hexdigest()

def _conditional_bounds(a,b,knots,names,limits,check=lambda:None):
    """a+(gain-1)b cubics, descending local coefficients; no sampled extrema."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);knots=np.asarray(knots,dtype=float)
    if a.shape!=b.shape or a.shape!=(4,len(knots)-1,len(names)) or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(knots).all() or not np.all(np.diff(knots)>0):raise ValueError('Finite consistent piecewise cubics required')
    lower={'gain':.75,'constraint':'schema'};upper={'gain':1.25,'constraint':'schema'};evaluations=0
    for interval,dt in enumerate(np.diff(knots)):
        check()
        for j,name in enumerate(names):
            low,high=limits[name]
            if not math.isfinite(low) or not math.isfinite(high) or low>=high:raise ValueError('Finite ordered limits required')
            A=Polynomial(a[:,interval,j][::-1]);B=Polynomial(b[:,interval,j][::-1])
            # A zero-sensitivity interval may still make every gain infeasible.
            if not np.any(B.coef):
                points=[0.,float(dt)]+[float(z.real) for z in A.deriv().roots() if abs(z.imag)<1e-9 and 0<z.real<dt]
                if any(not low<=float(A(t))<=high for t in points):return {'status':'empty','reason':'fixed_reference_constraint','joint':name,'interval_start_s':float(knots[interval])}
                continue
            for bound,label in ((low,'minimum'),(high,'maximum')):
                N=Polynomial([bound])-A;critical=N.deriv()*B-N*B.deriv()
                roots=critical.roots();points=[0.,float(dt)]+[float(z.real) for z in roots if abs(z.imag)<1e-8 and 0<z.real<dt]
                # B zeros do not constrain gain if A is already admitted; reject
                # an inconsistent fixed value rather than infer a safe interval.
                zeros=[float(z.real) for z in B.roots() if abs(z.imag)<1e-8 and 0<=z.real<=dt]
                if any(float(A(t))<low-1e-9 or float(A(t))>high+1e-9 for t in zeros):return {'status':'empty','reason':'zero_sensitivity_constraint','joint':name}
                for t in points:
                    sensitivity=float(B(t))
                    if sensitivity==0:continue
                    gain=1+float(N(t))/sensitivity
                    if not math.isfinite(gain):raise ValueError('Nonfinite numerical gain bound')
                    evaluations+=1
                    item={'gain':gain,'joint':name,'side':label,'time_s':float(knots[interval]+t),'limit_rad':bound,'base_value_rad':float(A(t)),'sensitivity_rad_per_gain':sensitivity}
                    is_lower=(label=='minimum' and sensitivity>0) or (label=='maximum' and sensitivity<0)
                    if is_lower and gain>lower['gain']:lower=item
                    if not is_lower and gain<upper['gain']:upper=item
    return {'status':'available' if lower['gain']<=upper['gain'] else 'empty','lower':lower,'upper':upper,'critical_evaluations':evaluations}

def advise_gain_interval(assets,program,field,*,cancel_event=None,timeout_s=45):
    p=parse_program(program)
    if field not in GAIN_FIELDS:raise ValueError('Unknown bounded gain field')
    if type(timeout_s) not in (int,float) or not 0<timeout_s<=60 or not math.isfinite(timeout_s):raise ValueError('Advice timeout must be finite in (0,60]')
    deadline=time.monotonic()+timeout_s
    def check():
        if cancel_event is not None and cancel_event.is_set():raise AdviceCancelled('Conditional advice cancelled')
        if time.monotonic()>deadline:raise AdviceTimeout('Conditional advice timed out')
    check();assets.verify();check();references=assets.references();sources=assets.source_records();profile=profile_from_yaml()
    pins=assets.manifest['live_numeric_pins']
    for key,path in (('body_sha256',profile['body_path']),('yaml_sha256',profile['yaml_path'])):
        actual=_sha(Path(path).read_bytes())
        if actual!=profile[key] or actual not in pins.values():raise ValueError('Advice profile is not the admitted source profile')
    identity={'program':p,'field':field,'asset_manifest':assets.manifest,'sources':sources,'body_sha256':profile['body_sha256'],'controller_sha256':profile['yaml_sha256'],'method':METHOD,'advice_source_sha256':_sha(Path(__file__).read_bytes())}
    evidence={'schema':'pulsemorph.conditional_gain_advice.v1','field':field,'status':'unavailable','input_sha256':_sha(_encoded(identity)),'program_sha256':_sha(_encoded(p)),'conditioning':{'order':p['order'],'heading_policy':p['heading_policy'],'other_gains':{k:p[k] for k in GAIN_FIELDS if k!=field},'execution_s':8,'reference_horizon_s':8.14,'transition_s':[3.5,4.5]},'source_profile':{'asset_manifest_canonical_sha256':_sha(_encoded(assets.manifest)),'asset_manifest_raw_sha256':_sha((assets.directory/'manifest.json').read_bytes()),'body_sha256':profile['body_sha256'],'controller_sha256':profile['yaml_sha256'],'source_records_sha256':_sha(_encoded(sources))},'method':METHOD,'basis_attempts':[],'new_model_calls':0,'new_onnx_calls':0,'new_FK_calls':0,'new_physics_steps':0,'execution_authority':False,'qualification':'Conditional numerical reference advice only; not interval-arithmetic certified, physical attainability, goal rewriting or permission to execute. Original compiler validates every proposal. No control is selected or clamped. Cancellation checked between compiler calls and intervals.'}
    admitted=[]
    for gain in BASE_GAINS:
        check();basis=dict(p,**{field:gain});attempt={'gain':gain}
        try:
            reference,report=compile_program(references,basis,reference_id='gain_advice_'+evidence['input_sha256'][:16]+'_'+str(len(evidence['basis_attempts'])),sources=sources)
            admitted.append((gain,reference));attempt.update(status='admitted',strict_future_queries=report['strict_future_queries'])
        except ProgramCompileError as error:attempt.update(status='rejected',compile_status=error.report.get('status'),violations=copy.deepcopy(error.report.get('violations',[])))
        evidence['basis_attempts'].append(attempt);check()
        if len(admitted)==2:break
    if len(admitted)<2:
        evidence['reason']='fewer_than_two_admitted_bases';check();assets.verify();check();return evidence
    (g0,r0),(g1,r1)=admitted
    if not np.array_equal(r0.time_offsets_s,r1.time_offsets_s):raise ValueError('Advice basis clocks differ')
    c0=CubicHermiteSpline(r0.time_offsets_s,r0.joint_position_rad,r0.joint_velocity_rad_s,extrapolate=False).c;c1=CubicHermiteSpline(r1.time_offsets_s,r1.joint_position_rad,r1.joint_velocity_rad_s,extrapolate=False).c
    slope=(c1-c0)/(g1-g0);base=c0+(1-g0)*slope
    bounds=_conditional_bounds(base,slope,r0.time_offsets_s,list(r0.metadata['body']['joint_names']),_joint_limits(profile),check)
    evidence['numerical_bounds']=bounds
    if bounds['status']!='available':
        evidence['reason']='empty_conditional_interval';check();assets.verify();check();return evidence
    lo=bounds['lower']['gain'];hi=bounds['upper']['gain'];lower=lo+(GAIN_MARGIN if bounds['lower'].get('constraint')!='schema' else 0)
    upper=hi-(GAIN_MARGIN if bounds['upper'].get('constraint')!='schema' else 0)
    if lower>upper:
        evidence['reason']='empty_after_numerical_margin';check();assets.verify();check();return evidence
    evidence.update(status='available',lower=lower,upper=upper,numerical_margin_gain=GAIN_MARGIN)
    check();assets.verify();check();return evidence
