"""Persistent, paused-between-actions episode; no dance clip or resets per command."""
from __future__ import annotations
import copy,hashlib,json,math,re,threading
import numpy as np

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def fingerprint(v):return hashlib.sha256(canonical(v)).hexdigest()
def number(v):return type(v) in (int,float) and math.isfinite(v)

class ActionRejected(ValueError):
 def __init__(self,message,report=None):
  super().__init__(message);self.report=report or {'schema':'pulsemorph.embodied_action_rejection.v1','category':'local_action_constraint','message':message}

class ObservationMismatch(RuntimeError):pass

class JointSegment:
 """Continuous cubic Hermite, previous committed reference → terminal rest."""
 def __init__(self,q,dq,target,duration,ranges,max_speed=2.,max_acceleration=12.):
  self.q=np.asarray(q,dtype=float).copy();self.dq=np.asarray(dq,dtype=float).copy();self.target=np.asarray(target,dtype=float).copy();self.duration=float(duration)
  if any(a.shape!=(29,) or not np.isfinite(a).all() for a in (self.q,self.dq,self.target)):raise ActionRejected('Finite29joint segment required')
  T=self.duration;delta=self.target-self.q
  self.c=np.stack([self.q,self.dq,3*delta/T**2-2*self.dq/T,-2*delta/T**3+self.dq/T**2])
  self.report=[]
  for j,(lo,hi) in enumerate(ranges):
   a,b,c,d=self.c[:,j];roots=np.roots([3*d,2*c,b]) if d else (np.roots([2*c,b]) if c else [])
   times=[0.,T]+[float(x.real) for x in roots if abs(x.imag)<1e-10 and 0<x.real<T]
   vals=[a+b*t+c*t*t+d*t*t*t for t in times]
   speed_times=[0.,T]+([-c/(3*d)] if d and 0<-c/(3*d)<T else [])
   speed=max(abs(b+2*c*t+3*d*t*t) for t in speed_times);accel=max(abs(2*c),abs(2*c+6*d*T))
   admitted=min(vals)>=lo-1e-9 and max(vals)<=hi+1e-9 and speed<=max_speed+1e-9 and accel<=max_acceleration+1e-9
   self.report.append({'joint_index':j,'minimum_rad':min(vals),'maximum_rad':max(vals),'max_speed_rad_s':speed,'max_acceleration_rad_s2':accel,'admitted':bool(admitted)})
  if not all(r['admitted'] for r in self.report):raise ActionRejected('Continuous joint/rate/acceleration bounds rejected; no clamp',{'schema':'pulsemorph.embodied_action_rejection.v1','category':'continuous_joint_constraint','violations':[r for r in self.report if not r['admitted']],'allowed_max_speed_rad_s':max_speed,'allowed_max_acceleration_rad_s2':max_acceleration})
 def sample(self,time_s):
  t=float(time_s)
  if t>=self.duration:return self.target.copy(),np.zeros(29)
  if t<0:raise ActionRejected('Reference time cannot go backwards')
  a,b,c,d=self.c
  return a+b*t+c*t*t+d*t*t*t,b+2*c*t+3*d*t*t

class PersistentEpisode:
 """Backend object, simulator state, policy history and clock survive every action."""
 def __init__(self,backend,session_id,*,max_control_steps=60,cancel_event=None):
  if type(session_id) is not str or re.fullmatch('[A-Za-z0-9_-]{1,100}',session_id) is None:raise ValueError('Bounded session id required')
  if type(max_control_steps) is not int or not 1<=max_control_steps<=500:raise ValueError('Explicit bounded control-step authority required')
  if hasattr(backend,'authority') and (backend.authority['session_id']!=session_id or max_control_steps>backend.authority['max_control_steps']):raise ValueError('Episode/authority assignment differs')
  self.cancel_event=cancel_event
  self.backend=backend;self.session_id=session_id;self.max_control_steps=max_control_steps;self.revision=0;self.control_index=0;self.stopped=False;self._lock=threading.RLock();self._receipts={};self.actions=[];self.references=[];self.common_references=[]
  state=backend.observe();self.reference_q=np.asarray(state['qpos'][7:],dtype=float);self.reference_dq=np.zeros(29)
  self._observation=self._make_observation(state)
 def _make_observation(self,state):
  return {'schema':'pulsemorph.embodied_observation.v1','session_id':self.session_id,'revision':self.revision,'sim_time_s':state['sim_time_s'],'control_index':self.control_index,'remaining_control_steps':self.max_control_steps-self.control_index,'qpos':copy.deepcopy(state['qpos']),'qvel':copy.deepcopy(state['qvel']),'joint_positions_rad':dict(zip(self.backend.joint_names,state['qpos'][7:])),'previous_reference':{'q':self.reference_q.tolist(),'dq':self.reference_dq.tolist()},'processed_history_sha256':state['processed_history_sha256'],'body_poses':copy.deepcopy(state['body_poses']),'contacts':copy.deepcopy(state['contacts']),'physical_flags':{**state['physical_flags'],'stopped':self.stopped},'wrist_jacobians_world_m_per_rad':copy.deepcopy(state.get('wrist_jacobians_world_m_per_rad',{}))}
 def observe(self):
  with self._lock:return copy.deepcopy(self._observation)
 def capabilities(self):
  return {'schema':'pulsemorph.embodied_capabilities.v1','tools':{'move_joints':{'arguments_schema':{'type':'object','properties':{'duration_s':{'type':'number','minimum':.4,'maximum':1.},'joint_targets_rad':{'type':'object','properties':{n:{'type':'number','minimum':float(r[0]),'maximum':float(r[1])} for n,r in zip(self.backend.joint_names,self.backend.joint_ranges)},'additionalProperties':False}},'required':['duration_s','joint_targets_rad'],'additionalProperties':False}}},'joint_names':list(self.backend.joint_names),'joint_ranges_rad':[list(r) for r in self.backend.joint_ranges],'physics_step_s':.001,'control_step_s':.02,'future_offsets_s':[.02,.04,.08,.16],'reference_max_speed_rad_s':2.,'reference_max_acceleration_rad_s2':12.,'max_target_change_rad':.35,'max_control_steps':self.max_control_steps,'simulation_pauses_between_actions':True,'no_source_choreography_playback':True,'anchor_reference':'fixed initial torso orientation; not a balance guarantee','next_action_may_revise_previous_future_predictions':True}
 def _parse(self,value):
  fields={'action_id','expected_revision','observation_sha256','tool','arguments'}
  if type(value) is not dict or set(value)!=fields or value['tool']!='move_joints':raise ActionRejected('Exact local tool envelope required')
  if type(value['action_id']) is not str or re.fullmatch('[A-Za-z0-9_-]{1,100}',value['action_id']) is None:raise ActionRejected('Bounded action id required')
  if type(value['expected_revision']) is not int or value['expected_revision']!=self.revision or value['observation_sha256']!=fingerprint(self._observation):raise ObservationMismatch('Stale body observation/revision')
  args=value['arguments']
  if type(args) is not dict or set(args)!={'duration_s','joint_targets_rad'}:raise ActionRejected('Exact move_joints arguments required')
  T=args['duration_s']
  if type(T) not in (int,float) or not .4<=T<=1. or not math.isfinite(T) or abs(T/.02-round(T/.02))>1e-9:raise ActionRejected('Duration must be400..1000ms in20ms units')
  targets=args['joint_targets_rad']
  if type(targets) is not dict or not 1<=len(targets)<=29 or set(targets)-set(self.backend.joint_names):raise ActionRejected('Explicit known named-joint targets required')
  goal=self.reference_q.copy()
  for name,v in targets.items():
   j=self.backend.joint_names.index(name);lo,hi=self.backend.joint_ranges[j]
   if type(v) not in (int,float) or not lo<=v<=hi or not math.isfinite(v) or abs(v-goal[j])>.35+1e-10:raise ActionRejected('Joint target outside local authority')
   goal[j]=v
  steps=round(T/.02)
  if self.control_index+steps>self.max_control_steps:raise ActionRejected('Remaining physical budget exceeded before action')
  return JointSegment(self.reference_q,self.reference_dq,goal,T,self.backend.joint_ranges),steps
 def execute(self,value):
  with self._lock:
   if type(value) is dict and value.get('action_id') in self._receipts:
    old=self._receipts[value['action_id']]
    if old['request_sha256']!=fingerprint(value):raise ObservationMismatch('Changed duplicate action id')
    return {**copy.deepcopy(old),'replay_no_new_work':True}
   if self.stopped:raise RuntimeError('Episode stopped; no automatic recovery/restart')
   try:segment,count=self._parse(value)
   except ActionRejected as error:
    error.report.update(physics_steps=0,revision_unchanged=self.revision,observation_sha256=fingerprint(self._observation))
    for row in error.report.get('violations',[]):row['joint_name']=self.backend.joint_names[row['joint_index']]
    raise
   if hasattr(self.backend,'_verify'):
    try:self.backend._verify()
    except Exception:self.stopped=True;self._observation['physical_flags']['stopped']=True;raise
   before=self.revision;before_hash=fingerprint(self._observation);start=self.backend.sim_time_s
   common=None
   if hasattr(self.backend,'profile'):
    from .reference_v1 import make_reference
    try:common=make_reference(self.backend,segment,self.session_id,value['action_id'],self.revision,start,fingerprint(self._observation),self.reference_q,self.reference_dq)
    except Exception:self.stopped=True;self._observation['physical_flags']['stopped']=True;raise
    self.common_references.append(common)
   action={'request':copy.deepcopy(value),'request_sha256':fingerprint(value),'before_revision':before,'start_sim_time_s':start,'continuous_bounds':segment.report,'planned_control_steps':count,'status':'attempted'};self.actions.append(action)
   try:
    for i in range(count):
     if self.cancel_event is not None and self.cancel_event.is_set():raise RuntimeError('EPISODE_CANCELLED')
     if common is None:future=[segment.sample((i+offset)*.02) for offset in (1,2,4,8)]
     else:
      from pulsemorph.motion.g1 import sample_reference
      sampled=sample_reference(common,np.asarray([(i+offset)*.02 for offset in (1,2,4,8)]));future=list(zip(sampled['dof_pos'],sampled['dof_vel']))
     self.backend.advance(np.asarray([x[0] for x in future]),np.asarray([x[1] for x in future]))
     self.control_index+=1;self.reference_q,self.reference_dq=segment.sample((i+1)*.02)
     self.references.append({'control_index':self.control_index-1,'time_s':start+i*.02,'future_q':np.asarray([x[0] for x in future]).tolist(),'future_dq':np.asarray([x[1] for x in future]).tolist(),'committed_end_q':self.reference_q.tolist(),'committed_end_dq':self.reference_dq.tolist()})
    if hasattr(self.backend,'_verify'):self.backend._verify()
    self.revision+=1;self._observation=self._make_observation(self.backend.observe());action['status']='completed'
    receipt={'schema':'pulsemorph.embodied_transition.v1','action_id':value['action_id'],'request_sha256':fingerprint(value),'before_observation_sha256':before_hash,'before_revision':before,'after_revision':self.revision,'steps':count*20,'control_steps':count,'observation':self.observe(),'replay_no_new_work':False}
    self._receipts[value['action_id']]=copy.deepcopy(receipt);return receipt
   except Exception:
    self.stopped=True;action['status']='failed_partial';self._observation=self._make_observation(self.backend.observe());raise
 def export(self,directory):
  from pathlib import Path
  directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
  (directory/'episode.json').write_bytes(canonical({'schema':'pulsemorph.embodied_episode.v1','session_id':self.session_id,'revision':self.revision,'control_steps':self.control_index,'stopped':self.stopped,'initialization_count':self.backend.initialization_count,'actions':self.actions,'references':self.references,'final_observation':self.observe(),'capabilities':self.capabilities()}))
  if self.common_references:
   from pulsemorph.motion.contracts import save_reference
   for i,ref in enumerate(self.common_references):save_reference(ref,directory/('reference_%02d'%i))
  self.backend.export(directory)
