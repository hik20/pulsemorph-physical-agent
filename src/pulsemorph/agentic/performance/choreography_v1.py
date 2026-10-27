"""Pure named-joint waypoint compiler. No prerecorded motion, FK or execution."""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

SCHEMA='pulsemorph.performance_program.v1'
DURATION=16.8
DT=.02
HORIZON=.16
MAX_SPEED=2.
MAX_ACCELERATION=12.
MIN_SPANS={'arms':.8,'trunk':.12,'hips':.12,'knees':.20,'ankles':.06}

class ChoreographyError(ValueError):
    def __init__(self,message,report=None):super().__init__(message);self.report=report or {}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def finite(value,lo,hi):return type(value) in (int,float) and lo<=value<=hi and math.isfinite(value)
def group(name):
    if any(p in name for p in ('shoulder_','elbow_','wrist_')):return 'arms'
    if name.startswith('waist_'):return 'trunk'
    if 'hip_' in name:return 'hips'
    if 'knee_' in name:return 'knees'
    if 'ankle_' in name:return 'ankles'
    raise ChoreographyError('Unknown joint group: '+name)
def envelope(name):
    if 'shoulder_' in name or 'elbow_' in name:return 1.6
    if 'wrist_' in name:return .8
    if name=='waist_yaw_joint':return .35
    if name.startswith('waist_'):return .18
    if 'hip_pitch' in name:return .45
    if 'hip_roll' in name:return .18
    if 'hip_yaw' in name:return .16
    if 'knee_' in name:return .7
    if 'ankle_pitch' in name:return .35
    if 'ankle_roll' in name:return .15
    raise ChoreographyError('No admitted envelope for '+name)

def validate_program(program,joint_names):
    fields={'schema','music_sha256','duration_s','tempo_bpm','keyframes','rationale'}
    if type(program) is not dict or set(program)!=fields or program['schema']!=SCHEMA:raise ChoreographyError('Exact performance program required')
    digest=program['music_sha256']
    if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ChoreographyError('Music byte SHA required')
    if not finite(program['duration_s'],DURATION,DURATION) or not finite(program['tempo_bpm'],100,100):raise ChoreographyError('Fixed16.8s100BPM program required')
    if type(program['rationale']) is not str or len(program['rationale'])>1600:raise ChoreographyError('Bounded rationale required')
    program['rationale'].encode('utf8');frames=program['keyframes']
    if type(frames) is not list or not 2<=len(frames)<=57:raise ChoreographyError('Two to57 explicit waypoints required')
    beats=[];values=[]
    for frame in frames:
        if type(frame) is not dict or set(frame)!={'beat_index','joint_targets_rad'}:raise ChoreographyError('Exact beat/named-target waypoint required')
        beat=frame['beat_index'];targets=frame['joint_targets_rad']
        if not finite(beat,0,28) or abs(beat*2-round(beat*2))>1e-12:raise ChoreographyError('Beat indices0..28 onhalfbeats required')
        if type(targets) is not dict or set(targets)!=set(joint_names):raise ChoreographyError('Every waypoint must name all29 joints exactly')
        if any(not finite(targets[n],-100,100) for n in joint_names):raise ChoreographyError('Finite named joint angles, not bool, required')
        beats.append(float(beat));values.append([float(targets[n]) for n in joint_names])
    if beats[0]!=0 or beats[-1]!=28 or any(b<=a for a,b in zip(beats,beats[1:])):raise ChoreographyError('Strict waypoint order0..28 required')
    return np.asarray(beats)*.6,np.asarray(values)

def slopes(times,q):
    """Shape-preserving weighted harmonic slopes, explicit zero endpoint speeds."""
    h=np.diff(times);delta=np.diff(q,axis=0)/h[:,None];m=np.zeros_like(q)
    for i in range(1,len(times)-1):
        a,b=delta[i-1],delta[i];same=(a*b)>0;w1=2*h[i]+h[i-1];w2=h[i]+2*h[i-1]
        m[i,same]=(w1+w2)/(w1/a[same]+w2/b[same])
    return m

def coefficients(q0,q1,dq0,dq1,h):
    delta=q1-q0
    return np.stack([q0,dq0,3*delta/h**2-(2*dq0+dq1)/h,-2*delta/h**3+(dq0+dq1)/h**2])

def analytic_report(times,q,dq,names,limits,initial):
    lows=q.min(axis=0).copy();highs=q.max(axis=0).copy();speeds=np.zeros(29);accelerations=np.zeros(29)
    for i,h in enumerate(np.diff(times)):
        co=coefficients(q[i],q[i+1],dq[i],dq[i+1],float(h))
        for j in range(29):
            a,b,c,d=co[:,j];roots=np.roots([3*d,2*c,b]) if abs(d)>1e-18 else (np.roots([2*c,b]) if abs(c)>1e-18 else [])
            candidates=[0.,float(h)]+[float(r.real) for r in roots if abs(r.imag)<1e-10 and 0<r.real<h]
            vals=[a+b*t+c*t*t+d*t*t*t for t in candidates];lows[j]=min(lows[j],*vals);highs[j]=max(highs[j],*vals)
            speed_times=[0.,float(h)]+([-c/(3*d)] if abs(d)>1e-18 and 0<-c/(3*d)<h else [])
            speeds[j]=max(speeds[j],*[abs(b+2*c*t+3*d*t*t) for t in speed_times]);accelerations[j]=max(accelerations[j],abs(2*c),abs(2*c+6*d*h))
    rows=[]
    for j,name in enumerate(names):
        lo,hi=limits[j];bound=envelope(name)
        admitted=lows[j]>=lo-1e-9 and highs[j]<=hi+1e-9 and max(abs(lows[j]-initial[j]),abs(highs[j]-initial[j]))<=bound+1e-9 and speeds[j]<=MAX_SPEED+1e-9 and accelerations[j]<=MAX_ACCELERATION+1e-8
        rows.append({'joint_name':name,'minimum_rad':float(lows[j]),'maximum_rad':float(highs[j]),'speed_max_rad_s':float(speeds[j]),'acceleration_max_rad_s2':float(accelerations[j]),'range_rad':[float(lo),float(hi)],'maximum_deviation_from_q0_rad':bound,'admitted':bool(admitted)})
    activation={}
    for label,minimum in MIN_SPANS.items():
        indices=[j for j,name in enumerate(names) if group(name)==label]
        span=float(max(highs[j]-lows[j] for j in indices))
        activation[label]={'maximum_named_joint_span_rad':span,'minimum_required_span_rad':minimum,'qualifies':span>=minimum-1e-9,'joint_names':[names[j] for j in indices]}
    return rows,activation

@dataclass
class CompiledPerformance:
    duration_s:float
    times:np.ndarray
    q:np.ndarray
    dq:np.ndarray
    report:dict
    program_sha256:str
    def sample(self,time_s):
        if type(time_s) not in (int,float,np.float64,np.float32) or not math.isfinite(time_s) or not 0<=time_s<=self.times[-1]+1e-10:raise ChoreographyError('Query outside declared finite reference horizon')
        if time_s>=self.duration_s:return self.q[-1].copy(),np.zeros(29)
        i=min(int(np.searchsorted(self.times,time_s,side='right')-1),len(self.times)-2);h=float(self.times[i+1]-self.times[i]);t=float(time_s-self.times[i]);a,b,c,d=coefficients(self.q[i],self.q[i+1],self.dq[i],self.dq[i+1],h)
        return a+b*t+c*t*t+d*t*t*t,b+2*c*t+3*d*t*t

def compile_program(program,joint_names,joint_ranges,initial_reference_q,initial_reference_dq=None,*,require_fullbody=True):
    names=list(joint_names)
    if len(names)!=29 or len(set(names))!=29 or any(type(n) is not str for n in names):raise ChoreographyError('Exactly29 unique profile joint names required')
    for name in names:group(name);envelope(name)
    if type(require_fullbody) is not bool:raise ChoreographyError('Explicit activation mode required')
    initial=np.asarray(initial_reference_q,dtype=float);velocity=np.zeros(29) if initial_reference_dq is None else np.asarray(initial_reference_dq,dtype=float)
    limits=np.asarray([joint_ranges[n] for n in names] if isinstance(joint_ranges,dict) else joint_ranges,dtype=float)
    if initial.shape!=(29,) or velocity.shape!=(29,) or limits.shape!=(29,2) or not all(np.isfinite(a).all() for a in (initial,velocity,limits)) or np.any(limits[:,0]>=limits[:,1]) or np.any(velocity!=0):raise ChoreographyError('Finite profile/initial q and zero initial dq required')
    knots,values=validate_program(program,names)
    if not np.array_equal(values[0],initial):raise ChoreographyError('First authored waypoint must equal actual initial reference exactly; no replacement')
    tangent=slopes(knots,values);times=np.arange(849)*DT;q=[];dq=[]
    for t in times:
        if t>=DURATION-1e-12:q.append(values[-1]);dq.append(np.zeros(29));continue
        i=min(int(np.searchsorted(knots,t,side='right')-1),len(knots)-2);h=float(knots[i+1]-knots[i]);u=float(t-knots[i]);a,b,c,d=coefficients(values[i],values[i+1],tangent[i],tangent[i+1],h);q.append(a+b*u+c*u*u+d*u*u*u);dq.append(b+2*c*u+3*d*u*u)
    q=np.asarray(q);dq=np.asarray(dq);q[0]=initial;dq[0]=velocity
    rows,activation=analytic_report(times,q,dq,names,limits,initial)
    numeric=all(row['admitted'] for row in rows);active=all(row['qualifies'] for row in activation.values())
    report={'schema':'pulsemorph.performance_compile_report.v1','numeric_validity_passed':numeric,'activation_qualified':active,'require_fullbody':require_fullbody,'admitted':numeric and (active or not require_fullbody),'control_rows':840,'common_rows':849,'duration_s':DURATION,'horizon_s':float(times[-1]),'joint_ranges':rows,'activation':activation,'execution_authority':False,'qualification':'Deterministic interpolation of caller-authored named waypoints. Activation is reference-joint span only, not physical dance/beauty/beat success. Final20ms Hermite curve owns angle/rate/acceleration bounds; fixed torso carrier semantics belong to backend.'}
    canonical(report)
    if not numeric:raise ChoreographyError('Continuous common curve violates unchanged range/envelope/speed/acceleration',report)
    if require_fullbody and not active:raise ChoreographyError('Reference does not meet declared full-body activation scope',report)
    return CompiledPerformance(DURATION,times,q,dq,report,hashlib.sha256(canonical(program)).hexdigest())
