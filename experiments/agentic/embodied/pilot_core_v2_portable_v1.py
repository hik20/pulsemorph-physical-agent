"""Prepare-only default. Execute only against a separately root-approved exact authority.

Portable owned-profile native body gate with SCRIPTED provider, not a live LLM experiment.
No native libraries or project runtimes are imported during preparation.
"""
from pathlib import Path
import argparse,copy,hashlib,json,re,sys,time
ROOT=Path(__file__).resolve().parents[3]
BASE=Path.home()/'.cache/pulsemorph-agentic/portable-v5-core-pilot'
BUDGETS={'model_constructions':1,'ort_sessions':1,'initializations':1,'warmup_ONNX':1,
 'online_ONNX':140,'physics_steps':2800,'runtime_FK':141,
 'failure_terminal_extra_FK_max':1,'wrist_Jacobian_calls':282,
 'failure_terminal_extra_Jacobian_max':2,'native_data_allocations':2,
 'new_LLM_calls':0,'wall_seconds':180}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
def ordinary(path):
 path=Path(path).absolute()
 for p in (path,*path.parents):need(not p.is_symlink() and not(p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024),'Linked path forbidden')
 return path
def load(path):
 path=ordinary(path);need(path.is_file() and path.stat().st_size<2*1024**2,'Bounded JSON required');return json.loads(path.read_bytes())
def write(path,value):
 path=ordinary(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def output_path(path):
 path=ordinary(path).resolve();need(path.is_relative_to(BASE.resolve()) and path!=BASE.resolve() and not path.exists(),'Exclusive new private pilot output required');return path
def prepare(config_path,environment_path,output):
 # Config and saved environment metadata only: never import native modules here.
 sys.path.insert(0,str(ROOT/'src'))
 from pulsemorph.agentic.embodied.app.config_v5 import load_config
 config=load_config(config_path);env=load(environment_path)
 need(env.get('isolated')==1 and env.get('user_site') is False and env.get('python',[])[:2]==[3,11],'Saved isolated environment required')
 need(Path(env.get('executable','')).resolve()==config['runtime_python'] and env.get('interpreter_sha256')==sha(config['runtime_python']),'Saved configured interpreter identity differs')
 pins={str(ordinary(config_path).resolve()):sha(config_path),str(ordinary(environment_path).resolve()):sha(environment_path)}
 identity=load(config['asset_identity']);need(identity.get('schema')=='pulsemorph.body_asset_identity.v1','Exact asset identity schema required')
 for row in identity['files']:
  rel=Path(row['path']);f=ordinary(config['vendor_root']/rel).resolve();need(not rel.is_absolute() and '..' not in rel.parts and f.is_relative_to(config['vendor_root']),'Asset path escape')
  need(f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],'Portable asset differs');pins[str(f)]=row['sha256']
 manifest=config['profile_dir']/'manifest.json';profile=load(manifest)
 need(profile.get('schema')=='pulsemorph.public_body_profile.v5' and profile.get('recipe')=='owned_symmetric_stand_v1' and profile.get('physical_admission') is False,'Owned unadmitted profile required')
 need(profile['asset_identity_sha256']==sha(config['asset_identity']) and profile['array_sha256']==sha(config['profile_dir']/'standing.npz') and profile['builder_sha256']==sha(ROOT/'src/pulsemorph/agentic/embodied/app/profile_v5.py'),'Owned profile provenance mismatch')
 for f in [config['asset_identity'],manifest,config['profile_dir']/'standing.npz',config['runtime_python'],Path(__file__).resolve()]:pins[str(f.resolve())]=sha(f)
 for row in env['module_pins'].values():
  for f,h in row['files'].items():need(sha(ordinary(f))==h,'Saved runtime binary changed');pins[str(Path(f).resolve())]=h
 # The metadata probe inventories extension modules. Include MuJoCo runtime DLLs
 # required by the constructor's loaded-module admission without importing them.
 mujoco_dir=Path(env['module_pins']['mujoco']['origin']).parent
 for f in mujoco_dir.glob('mujoco*.dll'):pins[str(f.resolve())]=sha(f)
 for folder in ['src/pulsemorph/agentic','src/pulsemorph/motion']:
  for f in (ROOT/folder).rglob('*.py'):
   if f.name!='trial_export.py':pins[str(f.resolve())]=sha(f)
 return {'schema':'pulsemorph.portable_core_native_pilot_plan.v1','root':str(ROOT.resolve()),'output':str(output_path(output)),
  'config_path':str(ordinary(config_path).resolve()),'environment_path':str(ordinary(environment_path).resolve()),'source_pins':pins,
  'backend':'PortableHybridArmBackend V5, unchanged v1 hybrid control law; owned_symmetric_stand_v1 initialization',
  'episode_max_controls':120,'prelude':{'actions':2,'duration_each_s':1.,'controls':100,'reference':'constant owned standing reference'},
  'rejected_candidate':{'duration_s':.4,'joint_delta_rad':.35,'required_acceleration_rad_s2':13.125,'unchanged_limit_rad_s2':12.,'required_physics_steps':0},
  'accepted_candidate':{'duration_s':.4,'controls':20,'reference':'same owned constant reference; one preview and exact commit'},
  'task':{'body':'right_rubber_hand','target_policy':'freeze actual world position after the 2-second prelude, before preview','tolerance_m':.012,'hold_s':.2,'max_sample_gap_s':.05,'min_root_height_m':.6},
  'provider':'three scripted fixture decisions; zero LLM calls','budgets':BUDGETS,
  'qualification':'Unexecuted portable-owned-profile native gate. Existing private Wave pass does not admit this initialization. No dance quality or live LLM claim.'}
class SharedBudget:
 """Deliberately shared audit authority across parent/copy, never physical state."""
 def __init__(self,out):self.out=out;self.started=time.monotonic();self.used={'online_ONNX':0,'warmup_ONNX':0,'wrist_Jacobian_calls':0}
 def __deepcopy__(self,memo):return self
 def reserve(self,key,count):
  need(time.monotonic()-self.started<BUDGETS['wall_seconds'],'Pilot wall budget expired')
  limit=BUDGETS[key]+(BUDGETS['failure_terminal_extra_Jacobian_max'] if key=='wrist_Jacobian_calls' else 0)
  need(self.used[key]+count<=limit,'Aggregate pilot budget exhausted: '+key)
  self.used[key]+=count
  p=self.out/'aggregate_budget_latest.json';tmp=self.out/'aggregate_budget_latest.tmp';tmp.write_text(json.dumps({'reserved':self.used,'elapsed_seconds':time.monotonic()-self.started},indent=2));tmp.replace(p)
def execute(plan_path,authority_path,authority_sha):
 need(sha(authority_path)==authority_sha,'Exact root authority SHA required');plan=load(plan_path);authority=load(authority_path)
 need(set(authority)=={'schema','approved','prepared_sha256','session_id','budgets'} and authority['schema']=='pulsemorph.portable_core_native_pilot_authority.v1' and authority['approved'] is True,'Separate root approved authority required')
 need(authority['prepared_sha256']==sha(plan_path) and authority['budgets']==BUDGETS,'Root must admit this exact prepared plan and every budget')
 need(re.fullmatch('[A-Za-z0-9_-]{1,80}',authority['session_id']) is not None,'New bounded session id required')
 need(plan==prepare(plan['config_path'],plan['environment_path'],plan['output']),'Prepared pilot changed before execution')
 out=output_path(plan['output']);out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan);write(out/'authority.json',authority);write(out/'execution_attempt.json',{'retry_allowed':False,'model_constructions_max':1,'ort_sessions_max':1,'started_before_native_construction':True})
 sys.path.insert(0,str(ROOT/'src'))
 from pulsemorph.agentic.embodied.app.config_v5 import load_config
 config=load_config(plan['config_path'])
 from types import SimpleNamespace
 from pulsemorph.agentic.embodied.backend_portable_hybrid_v5 import PortableHybridArmBackend
 from pulsemorph.agentic.embodied.runtime_v2 import PersistentEpisode,ActionRejected,fingerprint
 from pulsemorph.agentic.embodied.fork_v2 import PreviewSession,PreviewPhysicalFailure,snapshot_episode
 from pulsemorph.agentic.embodied.preview_orchestrator_v2 import PreviewOrchestrator
 from pulsemorph.agentic.embodied.tasks_v2 import evaluate_task
 budget=SharedBudget(out)
 class BudgetedHybrid(PortableHybridArmBackend):
  def __init__(self,*a,**kw):self.pilot_budget=budget;super().__init__(*a,**kw)
  def _infer(self,feed,role):self.pilot_budget.reserve('warmup_ONNX' if role=='warmup' else 'online_ONNX',1);return super()._infer(feed,role)
  def observe(self):
   if self._cached_observation is None:self.pilot_budget.reserve('wrist_Jacobian_calls',2)
   return super().observe()
 ba={'schema':'pulsemorph.embodied_execution_authority.v1','session_id':authority['session_id'],'max_control_steps':plan['episode_max_controls'],'source_pins':plan['source_pins']}
 episode=backend=fork=None;result={'status':'failed','new_LLM_calls':0,'qualification':plan['qualification']}
 try:
  backend=BudgetedHybrid.start(ROOT,authority=ba,config=config);episode=PersistentEpisode(backend,ba['session_id'],max_control_steps=120)
  joint=backend.joint_names[0];initial_q=episode.observe()['previous_reference']['q']
  def action(identity,duration,target):
   o=episode.observe();return {'action_id':identity,'expected_revision':o['revision'],'observation_sha256':fingerprint(o),'tool':'move_joints','arguments':{'duration_s':duration,'joint_targets_rad':{joint:target}}}
  for i in range(2):write(out/f'prelude_{i}.json',episode.execute(action('prelude_'+str(i),1.,initial_q[0])))
  frozen=episode.observe();target=copy.deepcopy(frozen['body_poses']['right_rubber_hand']['position_world_m']);task={'schema':'pulsemorph.embodied_wrist_task.v1','task_id':'native_sampled_hold','frame':'world','target_world_m':target,**{k:v for k,v in plan['task'].items() if k!='target_policy'}};write(out/'frozen_task.json',task)
  # Pick a single permitted sign with room for the unchanged .35-radian authority.
  choices=[(n,q,lo,hi) for n,q,(lo,hi) in zip(backend.joint_names,initial_q,backend.joint_ranges) if lo<=q+.35<=hi or lo<=q-.35<=hi];need(bool(choices),'No legal rejection probe joint')
  joint,q,lo,hi=choices[0];rejected=q+(.35 if q+.35<=hi else -.35)
  class ScriptedProvider:
   model='gpt-6.1-sol'
   def __init__(self):self.index=0;self.requests=[]
   def status(self):return {'enabled':True}
   def generate_json(self,prompt,schema,**kwargs):
    req=json.loads(prompt.split('REQUEST_JSON:\n')[1]);self.requests.append(req);i=self.index;self.index+=1
    if i==1:
     history=req['history'];need(len(history)==1 and history[0]['kind']=='hypothetical_preview_rejected','Normal rejection did not reach next scripted request');report=history[0]['report'];need(report['physics_steps']==0 and report['category']=='continuous_joint_constraint','Wrong rejection feedback');need(any(abs(v['max_acceleration_rad_s2']-13.125)<1e-8 for v in report['violations']),'Original acceleration limit changed')
    if i<2:proposal={'decision':'preview','action':{'tool':'move_joints','arguments':{'duration_s':.4,'joint_targets_rad':[{'joint_name':joint,'position_rad':rejected if i==0 else q}]}},'preview_id':None}
    else:
     need(i==2,'No extra scripted attempts permitted');eligible=[p['preview_id'] for p in req['previews'] if p['eligible']];need(len(eligible)==1,'Exact sole preview required');proposal={'decision':'commit','action':None,'preview_id':eligible[0]}
    proposal.update(summary='Scripted native gate, no LLM',reason='Fixed bounded verification sequence')
    return SimpleNamespace(data=proposal,receipt={'provider':'scripted_native_gate','model_requested':self.model,'model_returned':None,'live':False,'tool_events':0,'elapsed_seconds':0.,'trust_boundary':'Scripted provider fixture only; physical backend is native'})
  provider=ScriptedProvider();fork=PreviewSession(episode,max_preview_controls=20,max_total_preview_controls=20);before=snapshot_episode(episode)
  loop=PreviewOrchestrator(provider,task_verifier=evaluate_task,max_model_calls=3,max_previews=2,max_sandbox_steps=400,max_actual_steps=400,wall_seconds=120,action_rejection_type=ActionRejected,preview_physical_failure_type=PreviewPhysicalFailure,allow_fixture=True)
  run=loop.run('Hold the frozen actual wrist position for sampled 0.2 seconds',task,fork,out/'native_gate');write(out/'scripted_provider_requests.json',provider.requests)
  need(run['status']=='completed' and run['task_verified_actual'],'Native positive hold not verified; no tolerance or target changes allowed')
  need(run['accepted_previews']==run['commit_attempts']==1 and run['preview_attempts']==2 and run['confirmed_sandbox_steps']==run['confirmed_actual_steps']==400,'Native gate schedule differed')
  need(budget.used=={'online_ONNX':140,'warmup_ONNX':1,'wrist_Jacobian_calls':282},'Aggregate work counters differed')
  branch=next(v['branch'] for v in fork._previews.values() if v['status']=='committed');native_counts={'actual_controls':backend.control_index,'actual_physics_steps':backend.step_count,'actual_FK':backend.forward_count,'actual_Jacobians':backend.jacobian_calls,'sandbox_controls':20,'sandbox_physics_steps':400,'sandbox_FK':branch.backend.forward_count-(before.control_index+1),'sandbox_Jacobians':branch.backend.jacobian_calls-2*(before.control_index+1)}
  need(native_counts['actual_controls']==120 and native_counts['actual_physics_steps']==2400 and native_counts['actual_FK']==121 and native_counts['actual_Jacobians']==242 and native_counts['sandbox_FK']==20 and native_counts['sandbox_Jacobians']==40,'Native suffix counters mismatch')
  result.update(status='passed',task_verified_actual=True,run=run,aggregate_reserved=budget.used,native_counts=native_counts,source_pins_unchanged=True)
 except Exception as error:
  result.update(error_type=type(error).__name__,retry_allowed=False,partial_evidence_preserved=True);raise
 finally:
  if episode is not None:episode.export(out/'actual_trace')
  if fork is not None:
   for pid in fork._previews:fork.export_preview(pid,out/('sandbox_'+pid))
  for f,h in plan['source_pins'].items():need(sha(f)==h,'Source changed during native pilot')
  write(out/'result.json',result)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path);p.add_argument('--environment',type=Path);p.add_argument('--output',type=Path);p.add_argument('--plan-output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--prepared',type=Path);p.add_argument('--authority',type=Path);p.add_argument('--authority-sha256');a=p.parse_args()
 if a.execute:
  need(a.prepared and a.authority and re.fullmatch('[a-f0-9]{64}',a.authority_sha256 or ''),'Exact separate root authority arguments required');print(json.dumps(execute(a.prepared,a.authority,a.authority_sha256)));return
 need(a.config and a.environment and a.output and a.plan_output,'Prepare requires portable config, saved environment and new output/plan paths');plan=prepare(a.config,a.environment,a.output);write(a.plan_output,plan);print(json.dumps({'prepared':str(a.plan_output),'sha256':sha(a.plan_output),'budgets':BUDGETS,'native_work_performed':False}))
if __name__=='__main__':main()
