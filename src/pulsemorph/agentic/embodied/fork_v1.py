"""Exact-state hypothetical branches; preview is never the actual body observation.

Primary API semantics: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-copydata
Python memory ownership: https://mujoco.readthedocs.io/en/stable/python.html#structs
Installed3.3.7 exposes MjData.__copy__; copy.copy preserves the model association.
No construction, reset, warmup, inference, FK, or step occurs on module import.
"""
from dataclasses import dataclass,asdict
import copy,hashlib,json,re,threading,types
import numpy as np
from .runtime_v1 import PersistentEpisode,ActionRejected,canonical,fingerprint

class PreviewIntegrityError(RuntimeError):pass
class PreviewPhysicalFailure(RuntimeError):
 def __init__(self,report):super().__init__('Hypothetical branch physically stopped; actual body unchanged');self.report=report

def _require(ok,message):
 if not ok:raise PreviewIntegrityError(message)
def _array_hash(value):
 a=np.asarray(value);return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def _model_hash(backend):
 model=getattr(backend,'model',None);h=hashlib.sha256()
 if model is not None:
  for name in ('actuator_gainprm','actuator_biasprm','jnt_stiffness','dof_damping','dof_frictionloss'):
   if hasattr(model,name):h.update(_array_hash(getattr(model,name)).encode())
  if hasattr(model,'opt'):
   h.update(canonical({'timestep':float(model.opt.timestep),'integrator':int(model.opt.integrator),'solver':int(model.opt.solver),'gravity':np.asarray(model.opt.gravity).tolist()}))
 return h.hexdigest()
def _log_hash(backend):
 h=hashlib.sha256()
 for name in ('states','velocities','times','targets','integration_states','substeps'):
  if hasattr(backend,name):h.update(name.encode()+_array_hash(getattr(backend,name)).encode())
 for name in ('inputs','outputs'):
  for row in getattr(backend,name,[]):
   for key,value in sorted(row.items()):h.update(key.encode()+_array_hash(value).encode())
 h.update(canonical(getattr(backend,'calls',[])));return h.hexdigest()

@dataclass(frozen=True)
class EpisodeSnapshot:
 schema:str
 session_id:str
 revision:int
 control_index:int
 observation_sha256:str
 backend_closure_sha256:str
 reference_sha256:str
 model_configuration_sha256:str
 inherited_logs_sha256:str
 fingerprint:str

def snapshot_episode(parent):
 with parent._lock:
  core={'schema':'pulsemorph.embodied_episode_snapshot.v1','session_id':parent.session_id,'revision':parent.revision,'control_index':parent.control_index,'observation_sha256':fingerprint(parent.observe()),'backend_closure_sha256':parent.backend.closure_sha256(),'reference_sha256':hashlib.sha256(np.asarray(parent.reference_q).tobytes()+np.asarray(parent.reference_dq).tobytes()).hexdigest(),'model_configuration_sha256':_model_hash(parent.backend),'inherited_logs_sha256':_log_hash(parent.backend)}
  return EpisodeSnapshot(**core,fingerprint=fingerprint(core))

def _copy_backend(backend,data_copier):
 clone=object.__new__(type(backend))
 shared={'model','session','mj','upstream'}
 for key,value in backend.__dict__.items():
  if key=='data':setattr(clone,key,data_copier(value))
  elif key in shared or isinstance(value,types.ModuleType):setattr(clone,key,value)
  else:setattr(clone,key,copy.deepcopy(value))
 if hasattr(backend,'data'):
  _require(clone.data is not backend.data,'Native data copy aliased parent')
  for field in ('qpos','qvel','ctrl','qacc_warmstart'):
   if hasattr(backend.data,field):
    a,b=getattr(backend.data,field),getattr(clone.data,field)
    _require(not np.shares_memory(a,b) and _array_hash(a)==_array_hash(b),'Data field copy/memory isolation failed: '+field)
 if hasattr(backend,'model'):_require(clone.model is backend.model,'Fixed model should be shared, not reconstructed')
 if hasattr(backend,'session'):_require(clone.session is backend.session,'Stateless actor session should be shared')
 return clone

def fork_episode(parent,branch_id,*,_data_copier=copy.copy):
 """Copy FULL native data plus executor state; no ctor/reset/forward/warmup.

 _data_copier exists only for pure injected fixtures. Production uses copy.copy.
 Logical session/revision remain identical for exact observation/action binding;
 branch_id/provenance are separate and do not masquerade as a new actual state.
 """
 if type(branch_id) is not str or re.fullmatch('[A-Za-z0-9_-]{1,100}',branch_id) is None:raise ValueError('Bounded branch identity required')
 with parent._lock:
  _require(not parent.stopped,'Cannot fork a stopped actual episode')
  before=snapshot_episode(parent);branch=object.__new__(type(parent))
  for key,value in parent.__dict__.items():
   if key=='backend':setattr(branch,key,_copy_backend(value,_data_copier))
   elif key=='_lock':setattr(branch,key,threading.RLock())
   elif key=='cancel_event':setattr(branch,key,value) # Shared external cancellation only.
   elif key=='common_references':setattr(branch,key,list(value)) # MotionReference is immutable bytes-backed.
   else:setattr(branch,key,copy.deepcopy(value))
  branch.branch_id=branch_id;branch.branch_provenance={'schema':'pulsemorph.embodied_fork.v1','branch_id':branch_id,'parent_snapshot':asdict(before),'hypothetical_only':True,'new_initializations':0,'new_warmup_calls':0,'shared_readonly_model':hasattr(parent.backend,'model'),'shared_stateless_actor':hasattr(parent.backend,'session')}
  _require(snapshot_episode(branch)==before,'Full copied executor closure/reference differs')
  _require(snapshot_episode(parent)==before,'Fork creation changed actual parent')
  return branch

def _origin(backend):
 return {name:len(getattr(backend,name,[])) for name in ('states','velocities','times','targets','integration_states','substeps','inputs','outputs','calls')}|{name:int(getattr(backend,name,0)) for name in ('step_count','control_index','forward_count','jacobian_calls','initialization_count')}
def _suffix(backend,origin):
 h=hashlib.sha256();shapes={}
 for name in ('states','velocities','times','targets','integration_states','substeps'):
  value=np.asarray(getattr(backend,name,[])[origin[name]:]);h.update(name.encode()+_array_hash(value).encode());shapes[name]=list(value.shape)
 for name in ('inputs','outputs'):
  rows=getattr(backend,name,[])[origin[name]:];shapes[name]=len(rows)
  for row in rows:
   for key,value in sorted(row.items()):h.update(name.encode()+key.encode()+_array_hash(value).encode())
 return {'sha256':h.hexdigest(),'shapes':shapes,'new_physics_steps':int(getattr(backend,'step_count',0))-origin['step_count'],'new_online_calls':int(getattr(backend,'control_index',0))-origin['control_index'],'new_runtime_FK':int(getattr(backend,'forward_count',0))-origin['forward_count'],'new_wrist_Jacobian_calls':int(getattr(backend,'jacobian_calls',0))-origin['jacobian_calls'],'new_initializations':int(getattr(backend,'initialization_count',0))-origin['initialization_count'],'new_warmup_calls':sum(row.get('role')=='warmup' for row in getattr(backend,'calls',[])[origin['calls']:])}

class PreviewSession:
 """Serial preview then explicit byte-parity commit; never chooses a control."""
 def __init__(self,parent,*,max_preview_controls=50,max_total_preview_controls=150):
  if type(max_preview_controls) is not int or not 1<=max_preview_controls<=50 or type(max_total_preview_controls) is not int or not max_preview_controls<=max_total_preview_controls<=1000:raise ValueError('Bounded independent preview authority required')
  self.parent=parent;self.max_preview_controls=max_preview_controls;self.max_total_preview_controls=max_total_preview_controls;self.reserved_preview_controls=0;self.actual_preview_steps=0;self._previews={};self._commits={};self._lock=threading.RLock();self.stopped=False
 def capabilities(self):return self.parent.capabilities()
 def observe(self):return self.parent.observe()
 def preview(self,action):
  with self._lock,self.parent._lock:
   _require(not self.stopped,'Preview controller stopped')
   segment,count=self.parent._parse(action) # Pure bounds/revision checks BEFORE copying or work.
   _require(count<=self.max_preview_controls and self.reserved_preview_controls+count<=self.max_total_preview_controls,'Independent preview budget exhausted')
   self.reserved_preview_controls+=count;before=snapshot_episode(self.parent);origin=_origin(self.parent.backend);identity='preview_%03d'%len(self._previews);branch=fork_episode(self.parent,identity)
   try:transition=branch.execute(copy.deepcopy(action))
   except Exception as error:
    suffix=_suffix(branch.backend,origin);self.actual_preview_steps+=suffix['new_physics_steps'];unchanged=snapshot_episode(self.parent)==before
    _require(unchanged,'Failed preview changed actual parent')
    report={'schema':'pulsemorph.embodied_preview_failure.v1','preview_id':identity,'parent_observation_sha256':before.observation_sha256,'parent_revision':before.revision,'parent_unchanged':True,'suffix':suffix,'error_type':type(error).__name__,'hypothetical_only':True}
    self._previews[identity]={'status':'failed','branch':branch,'report':report}
    if isinstance(error,RuntimeError) and str(error)=='Actual physical finite/rootheight/warning stop':raise PreviewPhysicalFailure(report) from None
    self.stopped=True;raise
   suffix=_suffix(branch.backend,origin);self.actual_preview_steps+=suffix['new_physics_steps']
   _require(snapshot_episode(self.parent)==before,'Preview mutated actual parent/model/logs')
   _require(suffix['new_initializations']==suffix['new_warmup_calls']==0,'Preview reconstructed/reset/warmed actor')
   receipt={'schema':'pulsemorph.embodied_preview.v1','preview_id':identity,'parent_observation_sha256':before.observation_sha256,'parent_revision':before.revision,'parent_snapshot_fingerprint':before.fingerprint,'action':copy.deepcopy(action),'steps':transition['steps'],'control_steps':transition['control_steps'],'hypothetical_observation':branch.observe(),'trajectory_sha256':suffix['sha256'],'suffix':suffix,'parent_unchanged':True,'hypothetical_only':True,'new_model_calls':0,'fork_allocations':{'new_MjData_copies':1,'new_MjModel':0,'new_ONNX_sessions':0,'new_initializations':0,'new_warmup':0},'quality_scope':'Actual final wrist/body positions are hypothetical branch observations; fixed task error is evaluated by the caller, not selected or rewritten by this module.'}
   self._previews[identity]={'status':'ready','snapshot':before,'origin':origin,'branch':branch,'receipt':copy.deepcopy(receipt),'action':copy.deepcopy(action)}
   return receipt
 def commit(self,value):
  with self._lock,self.parent._lock:
   _require(not self.stopped,'Preview controller stopped')
   required={'preview_id','expected_revision','observation_sha256','commit_id'}
   _require(type(value) is dict and set(value)==required and type(value['commit_id']) is str and re.fullmatch('[A-Za-z0-9_-]{1,100}',value['commit_id']) is not None,'Exact explicit preview commit required')
   if value['commit_id'] in self._commits:
    old=self._commits[value['commit_id']];_require(old['request_sha256']==fingerprint(value),'Changed duplicate commit id');return {**copy.deepcopy(old),'replay_no_new_work':True}
   entry=self._previews.get(value['preview_id']);_require(entry is not None and entry['status']=='ready','Only completed ready preview can commit')
   before=snapshot_episode(self.parent)
   _require(type(value['expected_revision']) is int and value['expected_revision']==before.revision and value['observation_sha256']==before.observation_sha256 and before==entry['snapshot'],'Actual state/reference/history changed after preview')
   try:
    transition=self.parent.execute(copy.deepcopy(entry['action']));actual=_suffix(self.parent.backend,entry['origin']);predicted=entry['receipt']['suffix'];after=snapshot_episode(self.parent);branch_after=snapshot_episode(entry['branch'])
    _require(actual==predicted and after==branch_after and transition['observation']==entry['receipt']['hypothetical_observation'],'Preview/actual full-state/feed/output/trajectory byte parity failed')
   except Exception:
    self.stopped=True;self.parent.stopped=True;raise
   entry['status']='committed';result={**transition,'schema':'pulsemorph.embodied_preview_commit.v1','preview_id':value['preview_id'],'commit_id':value['commit_id'],'request_sha256':fingerprint(value),'preview_actual_byte_equal':True,'parity':{'suffix_sha256':actual['sha256'],'full_executor_snapshot_fingerprint':after.fingerprint,'qpos_qvel_integration_history_inputs_outputs_body_observation_exact':True},'actual_suffix':actual,'hypothetical_only':False}
   self._commits[value['commit_id']]=copy.deepcopy(result);return result
 def export_preview(self,preview_id,directory):
  from pathlib import Path
  with self._lock:
   entry=self._previews[preview_id];directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
   (directory/'preview.json').write_bytes(canonical(entry.get('receipt',entry.get('report'))));entry['branch'].export(directory/'hypothetical_trace')
