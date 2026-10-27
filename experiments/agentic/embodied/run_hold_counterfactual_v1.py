"""Prepare-only by default. A root-admitted plan may execute one fresh HOLD branch."""
from pathlib import Path
import argparse,copy,hashlib,json,re,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
BASE=Path.home()/'.cache/pulsemorph-agentic/embodied-v1'

def raw_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
def ordinary(path):
 path=Path(path).absolute()
 for p in (path,*path.parents):
  need(not p.is_symlink() and not(p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024),'Linked counterfactual path forbidden')
 return path
def load(path):
 path=ordinary(path);need(path.is_file() and path.stat().st_size<=2*1024**2,'Bounded ordinary JSON required');return json.loads(path.read_bytes())
def write_new(path,value):
 path=ordinary(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def inventory(folder):
 return {str(p.relative_to(folder)):raw_sha(ordinary(p)) for p in folder.rglob('*') if p.is_file()}
def exact(a,b,label):
 a,b=np.asarray(a),np.asarray(b)
 need(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'Exact prefix mismatch: '+label)
def private_output(output,live):
 output=ordinary(output).resolve();need(output.is_relative_to(BASE.resolve()) and output!=BASE.resolve() and not output.is_relative_to(live) and not live.is_relative_to(output),'Dedicated separate private C output required');need(not output.exists(),'Exclusive new counterfactual output required');return output

def prepare(live,output):
 live=ordinary(live).resolve();need(live.is_relative_to(BASE.resolve()),'Explicit private LIVE output required');output=private_output(output,live)
 before=inventory(live);episode=load(live/'trace/episode.json');backend=load(live/'trace/backend.json');task=load(live/'frozen_task.json');run=load(live/'episode_llm_01/result.json')
 need(run.get('action_completion_unconfirmed') is False,'No unconfirmed physical action may enter matched baseline')
 need(episode['initialization_count']==backend['initialization_count']==1 and backend['mjdata_instances']==backend['onnx_sessions']==1,'Live episode identity/reset contract')
 actions=episode['actions'];need(2<=len(actions)<=8 and all(a['status']=='completed' for a in actions),'Only fully committed exported live actions supported')
 durations=[a['request']['arguments']['duration_s'] for a in actions]
 need(durations[:2]==[1.,1.] and all(type(t) in (int,float) and .4<=t<=1. and abs(t/.02-round(t/.02))<1e-9 for t in durations),'Exact2s prelude and admitted chunk durations required')
 controls=sum(round(t/.02) for t in durations);need(controls==backend['online_calls']==episode['control_steps'] and controls<=400,'Live total controls mismatch or unbounded')
 need(run['committed_actions']==len(actions)-2 and run['confirmed_physics_steps']==sum(round(t/.001) for t in durations[2:]),'Only committed model actions match duration schedule')
 # Prelude was a genuine constant reference, not a hidden movement.
 from pulsemorph.motion.contracts import load_reference
 for i in (0,1):
  ref=load_reference(live/'trace'/('reference_%02d'%i));q=ref.joint_position_rad
  need(np.array_equal(q,np.tile(q[0],(len(q),1))) and np.all(ref.joint_velocity_rad_s==0),'Live prelude is not constant HOLD')
 need(before==inventory(live),'Live export changed during preparation')
 return {'schema':'pulsemorph.embodied_hold_counterfactual_plan.v1','live_output':str(live),'base_output':str(output),'live_inventory':before,'source_pins':backend['source_pins'],'driver_sha256':raw_sha(__file__),'frozen_task':task,'committed_durations_s':durations,'rejected_live_actions_not_replayed':run['rejected_actions'],'budgets':{'initializations':1,'mjdata':1,'onnx_sessions':1,'online_ONNX':controls,'warmup_ONNX':1,'physics_steps':controls*20,'runtime_FK':controls+1,'failure_terminal_extra_FK_max':1,'cached_wrist_Jacobian_calls':2*(len(actions)+1),'new_model_calls':0},'mandatory_prefix_s':2.,'scope':'No-LLM-movement matched HOLD counterfactual on a fresh branch; same committed durations and frozen world goal. Not equal-token performance comparison or replay of movement actions on alternate states.'}

def prefix_bridge(backend,live):
 d=live/'trace';results={}
 for name,actual in [('qpos',backend.states),('qvel',backend.velocities),('state_times',backend.times),('mj_integration_states',backend.integration_states),('applied_joint_targets',backend.targets)]:
  expected=np.load(d/(name+'.npy'),allow_pickle=False);count=100 if name=='applied_joint_targets' else 101;exact(np.asarray(actual),expected[:count],name);results[name]=True
 for name,rows in [('policy_inputs',backend.inputs),('policy_outputs',backend.outputs)]:
  with np.load(d/(name+'.npz'),allow_pickle=False) as saved:
   need(len(rows)==101,'Prefix warmup+100 online call count')
   for i,row in enumerate(rows):
    for key,array in row.items():exact(array,saved[f'call_{i:04d}__'+key],name+' '+str(i)+' '+key)
  results[name]=True
 return results

def execute(prepared_path,authority_path,authority_hash):
 need(raw_sha(authority_path)==authority_hash,'Root authority SHA mismatch');plan=load(prepared_path);authority=load(authority_path)
 required={'schema','approved','prepared_sha256','live_output','base_output','backend_authority'}
 need(type(authority) is dict and set(authority)==required and authority['schema']=='pulsemorph.embodied_hold_counterfactual_authority.v1' and authority['approved'] is True,'Exact root counterfactual authority required')
 need(authority['prepared_sha256']==raw_sha(prepared_path) and plan['schema']=='pulsemorph.embodied_hold_counterfactual_plan.v1','Prepared plan not admitted')
 need(plan['driver_sha256']==raw_sha(__file__),'Driver changed after preparation')
 live=ordinary(plan['live_output']).resolve();output=private_output(plan['base_output'],live)
 need(authority['live_output']==str(live) and authority['base_output']==str(output),'Authority path binding differs')
 need(inventory(live)==plan['live_inventory'],'Live evidence changed before counterfactual')
 need(prepare(live,output)==plan,'Prepared schedule/task/source/budget no longer matches')
 ba=authority['backend_authority'];need(ba['schema']=='pulsemorph.embodied_execution_authority.v1' and ba['max_control_steps']==plan['budgets']['online_ONNX'],'Exact counterfactual step cap required')
 need(ba['session_id']!=load(live/'trace/episode.json')['session_id'],'Fresh counterfactual session id required')
 for path,h in plan['source_pins'].items():need(ba['source_pins'].get(path)==h and raw_sha(path)==h,'Same frozen physical source/profile required')
 need(ba['source_pins'].get(str(Path(__file__).resolve()))==raw_sha(__file__),'Authority must bind this new driver')
 output.mkdir(parents=True,exist_ok=False);write_new(output/'prepared_copy.json',plan);write_new(output/'authority_copy.json',authority)
 from pulsemorph.agentic.embodied.backend_v1 import TrackerBackend
 from pulsemorph.agentic.embodied.runtime_v1 import PersistentEpisode,fingerprint
 from pulsemorph.agentic.embodied.tasks_v1 import evaluate_task
 backend=episode=None;started=time.monotonic();result={'schema':'pulsemorph.embodied_hold_counterfactual_result.v1','status':'failed','new_model_calls':0,'scope':plan['scope']}
 try:
  backend=TrackerBackend.start(ROOT,authority=ba);episode=PersistentEpisode(backend,ba['session_id'],max_control_steps=ba['max_control_steps'])
  initial=episode.observe();write_new(output/'observation_00.json',initial);joint=backend.joint_names[0];target=initial['previous_reference']['q'][0];assessments=[];verifier_state=None
  for index,duration in enumerate(plan['committed_durations_s']):
   observation=episode.observe();action={'action_id':'counterfactual_hold_%02d'%index,'expected_revision':observation['revision'],'observation_sha256':fingerprint(observation),'tool':'move_joints','arguments':{'duration_s':duration,'joint_targets_rad':{joint:target}}}
   write_new(output/('action_%02d.json'%index),action);transition=episode.execute(action);write_new(output/('transition_%02d.json'%index),transition);after=episode.observe();write_new(output/('observation_%02d.json'%(index+1)),after)
   if index==1:
    bridge=prefix_bridge(backend,live);write_new(output/'prefix_bridge.json',{'status':'PASS','through_s':2.,'numeric_and_all8input4output':bridge})
   if index>=1:
    assessment=evaluate_task(plan['frozen_task'],after,verifier_state);verifier_state=assessment.get('state');assessments.append(assessment)
  need(backend.control_index==plan['budgets']['online_ONNX'] and backend.step_count==plan['budgets']['physics_steps'],'Counterfactual actual total differs')
  result.update(status='completed',online_ONNX=backend.control_index,warmup_ONNX=1,physics_steps=backend.step_count,runtime_FK=backend.forward_count,cached_wrist_Jacobian_calls=backend.jacobian_calls,initializations=backend.initialization_count,endpoint_assessments=assessments,continuous_hold_or_full_dense_goal_evaluation_proved=False,final_observation=episode.observe(),wall_seconds=time.monotonic()-started)
 except Exception as error:
  result.update(error_type=type(error).__name__,retry_allowed=False,partial_evidence_preserved=True,wall_seconds=time.monotonic()-started)
  raise
 finally:
  if episode is not None:episode.export(output/'trace')
  write_new(output/'result.json',result)
  need(inventory(live)==plan['live_inventory'],'Original live evidence changed during counterfactual')
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--plan-output',type=Path);parser.add_argument('--execute',action='store_true');parser.add_argument('--prepared',type=Path);parser.add_argument('--authority',type=Path);parser.add_argument('--authority-sha256');args=parser.parse_args()
 if args.execute:
  need(args.prepared and args.authority and re.fullmatch('[a-f0-9]{64}',args.authority_sha256 or ''),'Explicit admitted execution arguments required');result=execute(args.prepared,args.authority,args.authority_sha256);print(json.dumps({k:result.get(k) for k in ('status','online_ONNX','physics_steps','new_model_calls')}));return
 need(args.live and args.output and args.plan_output,'Prepare requires explicit live/output/NEWplan-output');plan=prepare(args.live,args.output);write_new(args.plan_output,plan);print(json.dumps({'prepared':str(args.plan_output),'sha256':raw_sha(args.plan_output),'budgets':plan['budgets'],'actual_execution':False}))
if __name__=='__main__':main()
