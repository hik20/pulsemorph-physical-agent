"""Fresh isolated four-second fullbody dance-source physical trials."""
from pathlib import Path
import importlib.util,json,os,re,shutil,subprocess,time,threading,wave
import numpy as np
from .assets_v1 import DanceAssets,ROOT_CACHE,SOURCE,SOURCE_SHA,PUBLIC_PYTHON,MODEL,parameters,baseline_parameters,ordinary,sha,write_new

def verify(pins):
 if type(pins) is not dict or not pins:raise ValueError('Exact nonempty dance pins required')
 for path,digest in pins.items():
  if type(path) is not str or type(digest) is not str or len(digest)!=64 or sha(ordinary(path))!=digest:raise RuntimeError('Dance source/environment pin changed')

def music_observation(value):
 if value is None:return {'available':False,'duration_s':0.,'observed_through_sample':None}
 fields={'available','path','sha256','sample_rate','duration_s','observed_through_sample'}
 if type(value) is not dict or set(value)!=fields or value['available'] is not True:raise ValueError('Exact actual4s PCM observation required')
 p=ordinary(value['path'])
 if p.stat().st_size>1024*1024 or sha(p)!=value['sha256']:raise ValueError('Dance music bytes changed')
 with wave.open(str(p),'rb') as w:
  if (w.getcomptype(),w.getsampwidth(),w.getnchannels(),w.getframerate(),w.getnframes())!=('NONE',2,1,48000,192000):raise ValueError('Dance music must actual4s48kmono PCM16')
 if type(value['sample_rate']) is not int or value['sample_rate']!=48000 or type(value['observed_through_sample']) is not int or value['observed_through_sample']!=192000 or type(value['duration_s']) not in (int,float) or value['duration_s']!=4:raise ValueError('Dance music observation differs')
 return {**value,'path':str(p.resolve())}

def environment(root,python):
 code="import sys,json;sys.path[:0]="+repr([str(root/'src'),str(root/'research/vendor/ProtoMotions')])+";from pulsemorph.agentic.app.phrase.worker import environment;print(json.dumps(environment(),allow_nan=False))"
 env={k:v for k,v in os.environ.items() if k.upper() not in ('PYTHONPATH','PYTHONHOME')}
 done=subprocess.run([str(python),'-I','-B','-c',code],cwd=str(ROOT_CACHE),env=env,capture_output=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 if done.returncode or len(done.stdout)>128*1024:raise RuntimeError('Dance numerical environment probe failed')
 return json.loads(done.stdout)

def validate_proposal(value,root,*,check_environment=True):
 required={'schema','candidate_id','job_id','trial_index','reason','source_id','source_motion','control_tool','parameters','duration_s','source_start_s','audio','observation_id','task_id','runtime_session_id','implementation_pins','environment'}
 if type(value) is not dict or set(value)!=required or value['schema']!='pulsemorph.dance_proposal.v1':raise ValueError('Exact dance proposal schema required')
 if type(value['job_id']) is not str or re.fullmatch('[A-Za-z0-9_-]{1,80}',value['job_id']) is None or type(value['trial_index']) is not int or not -1<=value['trial_index']<=20:raise ValueError('Invalid dance job/index')
 if value['candidate_id']!=value['job_id']+'__trial_'+str(value['trial_index']) or value['task_id']!=value['job_id'] or value['runtime_session_id']!=value['candidate_id']:raise ValueError('Dance assignment mismatch')
 if value['source_id']!='dance' or value['source_motion']!={'path':SOURCE,'sha256':SOURCE_SHA} or value['control_tool']!='global_clock_v1' or value['duration_s']!=4 or value['source_start_s']!=0:raise ValueError('Unadmitted dance source/tool/clock')
 parameters(value['parameters'])
 if value['trial_index']==-1 and value['parameters']!=baseline_parameters():raise ValueError('Exact edited dance baseline required; not unit')
 if value['audio']!={'available':False,'duration_s':0.,'observed_through_sample':None}:music_observation(value['audio'])
 assets=DanceAssets(root);expected=assets.implementation_pins()
 # Exact additional environment binaries/executable are part of the observed environment.
 for path,digest in value['environment'].get('module_pins',{}).items():expected[path]=digest
 expected[str(PUBLIC_PYTHON.resolve())]=sha(PUBLIC_PYTHON)
 if value['audio'].get('available'):expected[value['audio']['path']]=value['audio']['sha256']
 if value['implementation_pins']!=expected:raise RuntimeError('Exact dance implementation pin assignment differs')
 verify(expected)
 if check_environment:
  from pulsemorph.agentic.app.phrase.worker import environment as actual_environment
  if actual_environment()!=value['environment']:raise RuntimeError('Dance loaded environment differs')
 return value

def verify_history(observed,outputs,index):
 expected=outputs[f'call_{index-1:04d}__joint_pos_targets'][:,None,:] if index>1 else np.zeros((1,1,29),np.float32)
 if observed.dtype!=np.float32 or observed.shape!=(1,1,29) or not np.array_equal(observed,expected):raise RuntimeError('Dance processed joint-target history mismatch')

def audit_trial(root,directory):
 directory=ordinary(directory).resolve()
 spec=importlib.util.spec_from_file_location('_dance_saved_audit',root/'experiments/agentic/evaluate_multifeature.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 before={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
 for p in directory.rglob('*'):ordinary(p)
 result=module._audit(directory);manifest=result['manifest']
 if manifest['schema']!='pulsemorph.dance_physical_trial.v1' or manifest.get('producer',{}).get('id')!='pulsemorph-dance-source-tool' or manifest['motion_sha256']!=SOURCE_SHA:raise RuntimeError('Wrong dance producer/source')
 proposal=json.loads((directory/'proposal.json').read_bytes());validate_proposal(proposal,root)
 if manifest['candidate_id']!=proposal['candidate_id'] or manifest['job_id']!=proposal['job_id'] or manifest['trial_index']!=proposal['trial_index']:raise RuntimeError('Saved dance assignment mismatch')
 fresh=module._audit_fresh_anchor(directory,result)
 from pulsemorph.motion.contracts import load_reference
 from pulsemorph.motion.g1 import assemble_reference_inputs
 reference=load_reference(directory/'common_reference')
 if reference.metadata['producer']['id']!='pulsemorph-dance-source-tool':raise RuntimeError('Common dance producer mismatch')
 from .assets_v1 import FIELDS
 feeds=dict(np.load(directory/'policy_inputs.npz',allow_pickle=False));outputs=dict(np.load(directory/'policy_outputs.npz',allow_pickle=False))
 from ..app.phrase.assets import INPUTS,OUTPUTS
 for i in range(201):
  prefix=f'call_{i:04d}__'
  if {k[len(prefix):] for k in feeds if k.startswith(prefix)}!=set(INPUTS) or {k[len(prefix):] for k in outputs if k.startswith(prefix)}!=set(OUTPUTS):raise RuntimeError('Exact8input4output dance interface differs')
  if i:
   strict=assemble_reference_inputs(reference,(i-1)*.02)
   for k in ('mimic_future_dof_pos','mimic_future_dof_vel'):
    if not np.allclose(strict[k],feeds[prefix+k],rtol=0,atol=2e-6):raise RuntimeError('Dance strict common/feed mismatch')
   verify_history(feeds[prefix+'historical_processed_actions'],outputs,i)
   angular=result['arrays']['qvel'][i-1,3:6].astype('float32')[None]
   if not np.array_equal(feeds[prefix+'current_root_local_ang_vel'],angular):raise RuntimeError('Dance current root angular velocity differs from actualqvel')
  if any(v.dtype!=np.float32 or not np.isfinite(v).all() for k,v in feeds.items() if k.startswith(prefix)):raise RuntimeError('Dance feed dtype/finite mismatch')
  if any(v.dtype!=np.float32 or v.shape!=(1,29) or not np.isfinite(v).all() for k,v in outputs.items() if k.startswith(prefix)):raise RuntimeError('Dance output shape/dtype/finite mismatch')
 metrics={'min_root_height_m':float(result['sub'][:,2].min()),'mean_joint_rmse_rad':result['mean_joint_rmse_rad'],'physical_gate_passed':bool(result['sub'][:,2].min()>=.6 and result['mean_joint_rmse_rad']<=.35),'online_onnx_calls':200,'warmup_onnx_calls':1,'physics_steps':4000,'saved_state_audit_FK':200,'strict_common_rows':200,'goal_metrics':'diagnostic only; no full intent/aesthetic/dance quality pass claimed'}
 if before!={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}:raise RuntimeError('Dance artifacts changed during audit')
 return {'schema':'pulsemorph.dance_saved_audit.v1','integrity_passed':True,'inventory':before,'fresh_anchor_audit':fresh,'metrics':metrics,'new_model_onnx_physics_calls':0}

class DanceRuntime:
 def __init__(self,root,python=PUBLIC_PYTHON,artifact_root=ROOT_CACHE):
  self.root=ordinary(root).resolve();self.python=ordinary(python).resolve();self.artifact_root=ordinary(artifact_root).resolve();self.assets=DanceAssets(self.root)
  if self.python!=PUBLIC_PYTHON.resolve() or self.artifact_root!=ROOT_CACHE.resolve():raise ValueError('Exact reviewed public Python and private C dance root required')
  self.pins=self.assets.implementation_pins();self.pins[str(self.python)]=sha(self.python);self._implementation=self.pins
 def _unchanged(self):
  verify(self.pins)
 def _verify_trial(self,trial):
  return verify_trial(trial,self.root)
 def status(self):
  try:verify(self.pins);return {'available':True,'scope':'existing4sfullbodydance; boundedarm/clock edits, fresh physical proof required'}
  except (OSError,ValueError,RuntimeError):return {'available':False}
 def run_trial(self,parameters_value,output,job_id,trial_index,music_observation=None,cancel_event=None):
  pars=parameters(parameters_value);output=ordinary(output).resolve()
  if output.exists() or output==self.artifact_root or not output.is_relative_to(self.artifact_root):raise ValueError('Exclusive dedicatedC dance output required')
  if type(job_id) is not str or re.fullmatch('[A-Za-z0-9_-]{1,80}',job_id) is None or type(trial_index) is not int or not -1<=trial_index<=20:raise ValueError('Invalid job/index')
  if cancel_event and cancel_event.is_set():raise RuntimeError('Dance trial cancelled before attempt')
  if shutil.disk_usage(self.artifact_root.parent).free<256*1024**2:raise RuntimeError('Dance C disk requires256MiB free')
  verify(self.pins);audio=globals()['music_observation'](music_observation);self.artifact_root.mkdir(parents=True,exist_ok=True)
  env=environment(self.root,self.python);pins=dict(self.pins)
  pins.update(env.get('module_pins',{}))
  if audio['available']:pins[audio['path']]=audio['sha256']
  identity=job_id+'__trial_'+str(trial_index)
  proposal={'schema':'pulsemorph.dance_proposal.v1','candidate_id':identity,'job_id':job_id,'trial_index':trial_index,'reason':'Fresh bounded existingdance physical trial','source_id':'dance','source_motion':{'path':SOURCE,'sha256':SOURCE_SHA},'control_tool':'global_clock_v1','parameters':pars,'duration_s':4,'source_start_s':0,'audio':audio,'observation_id':identity+':music','task_id':job_id,'runtime_session_id':identity,'implementation_pins':pins,'environment':env}
  proposal_path=output.with_name(output.name+'.proposal.json');output.parent.mkdir(parents=True,exist_ok=True);write_new(proposal_path,proposal)
  log=output.with_name(output.name+'.worker.log')
  child_env={k:v for k,v in os.environ.items() if k.upper() not in ('PYTHONPATH','PYTHONHOME')}
  with log.open('xb') as stream:
   process=subprocess.Popen([str(self.python),'-I','-B',str(Path(__file__).with_name('worker_v1.py')),'--proposal',str(proposal_path),'--output',str(output)],cwd=str(self.artifact_root),env=child_env,stdout=stream,stderr=stream,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   deadline=time.monotonic()+330
   while process.poll() is None:
    if (cancel_event and cancel_event.is_set()) or time.monotonic()>deadline:
     subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));process.wait(timeout=10);raise RuntimeError('Dance worker cancelled or timed out; partial evidence preserved')
    time.sleep(.1)
  if process.returncode:
   report=output/'compile_report.json'
   if report.is_file():
    value=json.loads(report.read_bytes())
    if value.get('admitted') is False:
     from ..phrase_program import ProgramCompileError
     raise ProgramCompileError('Dance reference exceeds unchanged joint range',value)
   raise RuntimeError('Dance worker failed; partial evidence preserved')
  audit=audit_trial(self.root,output);write_new(output/'audit.json',audit)
  result={'schema':'pulsemorph.dance_trial.v1','status':'completed','integrity_passed':True,'id':identity,'index':trial_index,'parameters':pars,'paths':{'directory':str(output),'manifest':str(output/'manifest.json'),'audit':str(output/'audit.json'),'common_reference':str(output/'common_reference')},'metrics':audit['metrics']}
  write_new(output/'result.json',result)
  result['proof_sha256']={'manifest':sha(output/'manifest.json'),'audit':sha(output/'audit.json'),'result':sha(output/'result.json')}
  self._verify_trial(result)
  return result


def verify_trial(trial,root):
 if type(trial) is not dict or type(trial.get('id')) is not str or type(trial.get('index')) is not int:raise ValueError('Exact saved dance wrapper identity required')
 paths=trial.get('paths')
 if type(paths) is not dict or set(paths)!={'directory','manifest','audit','common_reference'} or any(type(v) is not str for v in paths.values()):raise ValueError('Exact dance path fields required')
 d=ordinary(paths['directory']).resolve()
 if not d.is_relative_to(ROOT_CACHE.resolve()) or d==ROOT_CACHE.resolve():raise ValueError('Saved dance directory outside private root')
 for key,name in [('manifest','manifest.json'),('audit','audit.json'),('common_reference','common_reference')]:
  if Path(paths[key]).resolve()!=d/name:raise ValueError('Saved dance path assignment differs')
 proofs=trial.get('proof_sha256')
 if type(proofs) is not dict or set(proofs)!={'manifest','audit','result'}:raise ValueError('Exact saved proof hashes required')
 for k,h in proofs.items():
  if sha(ordinary(d/(k+'.json')))!=h:raise RuntimeError('Saved dance proof changed')
 if (d/'audit.json').stat().st_size>256*1024:raise RuntimeError('Bounded dance audit required')
 audit=json.loads((d/'audit.json').read_bytes());saved=json.loads((d/'result.json').read_bytes());manifest=json.loads((d/'manifest.json').read_bytes())
 if audit.get('integrity_passed') is not True or saved.get('integrity_passed') is not True or saved.get('status')!='completed':raise RuntimeError('Dance result not measured/audited')
 for key in ('id','index','parameters','paths','metrics'):
  if trial.get(key)!=saved.get(key):raise RuntimeError('Saved dance wrapper binding differs')
 if saved['id']!=manifest['candidate_id'] or saved['index']!=manifest['trial_index'] or saved['metrics']!=audit['metrics']:raise RuntimeError('Saved dance proof assignment differs')
 inventory=audit['inventory'];actual={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()}
 if actual!=set(inventory)|{'audit.json','result.json'}:raise RuntimeError('Saved dance fileset differs')
 for name,h in inventory.items():
  p=ordinary(d/name)
  if not p.resolve().is_relative_to(d) or sha(p)!=h:raise RuntimeError('Saved dance artifact changed')
 verify(manifest['implementation_pins'])
 return {'integrity_passed':True,'metrics':dict(audit['metrics'])}
