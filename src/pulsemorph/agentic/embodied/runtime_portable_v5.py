"""Core V2 state machine with explicit configured-profile reference assembly.

Only the reference import changes; base exception/state/journal types remain V2.
"""
import copy
import numpy as np
from .runtime_v2 import PersistentEpisode as _V2Episode,ActionRejected,ObservationMismatch,canonical,fingerprint

class PersistentEpisode(_V2Episode):
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
    error.report=json.loads(canonical(error.report))
    raise
   if hasattr(self.backend,'_verify'):
    try:self.backend._verify()
    except Exception:self.stopped=True;self._observation['physical_flags']['stopped']=True;raise
   before=self.revision;before_hash=fingerprint(self._observation);start=self.backend.sim_time_s;samples=[]
   common=None
   if hasattr(self.backend,'profile'):
    from .reference_portable_v5 import make_reference
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
     if i==count-1:self.revision+=1
     self._observation=self._make_observation(self.backend.observe());samples.append(self.observe())
    if hasattr(self.backend,'_verify'):self.backend._verify()
    action['status']='completed'
    receipt={'schema':'pulsemorph.embodied_transition.v1','action_id':value['action_id'],'request_sha256':fingerprint(value),'before_observation_sha256':before_hash,'before_revision':before,'after_revision':self.revision,'steps':count*20,'control_steps':count,'observation':self.observe(),'replay_no_new_work':False}
    receipt.update(sampled_observations_sha256=fingerprint(samples),sampled_observations_count=len(samples))
    self._sampled_receipts[value['action_id']]=copy.deepcopy(samples);self._receipts[value['action_id']]=copy.deepcopy(receipt);return receipt
   except Exception:
    self.stopped=True;action['status']='failed_partial';self._observation=self._make_observation(self.backend.observe());raise
