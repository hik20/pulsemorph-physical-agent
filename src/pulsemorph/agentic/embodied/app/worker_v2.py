"""One fresh private body-agent job. Imports alone do not construct a simulator."""
from pathlib import Path
import argparse,copy,hashlib,json,os,re,shutil,sys,time,traceback,tempfile
ROOT=Path(__file__).resolve().parents[5];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions')]
from pulsemorph.agentic.embodied.app.assets_v1 import JOBS_ROOT,LIMITS,prepare_request,tasks,build_pins,ordinary,sha
from pulsemorph.agentic.embodied.runtime_v1 import PersistentEpisode,ActionRejected,fingerprint
from pulsemorph.agentic.embodied.fork_v1 import PreviewSession,PreviewPhysicalFailure,snapshot_episode
from pulsemorph.agentic.embodied.hybrid_arm_backend_v1 import HybridArmBackend
from pulsemorph.agentic.embodied.preview_orchestrator_v1 import PreviewOrchestrator
from pulsemorph.agentic.embodied.tasks_v1 import evaluate_task as evaluate_single
from pulsemorph.agentic.embodied.tasks_dual_v1 import evaluate_task as evaluate_dual
from pulsemorph.agentic.app.providers import CodexLocalProvider

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def write_new(path,value):
 with Path(path).open('xb') as f:f.write(canonical(value))
class MetadataPublicationError(RuntimeError):
 def __init__(self,target,attempts,elapsed):
  super().__init__('Bounded metadata replacement could not complete')
  self.target_name=Path(target).name;self.attempts=attempts;self.elapsed_seconds=elapsed

METADATA_STATS={'permission_retries':0}

def atomic(path,value,*,timeout_s=2.,sleep_s=.01):
 """Publish SAME serialized bytes; retry only Windows sharing PermissionError.

 This never retries provider generation, action execution, simulation or export.
 Every attempt uses one unique owned same-directory temporary. Interruptions and
 exhausted retries preserve the old published target and clean the owned temp.
 """
 if type(timeout_s) not in (int,float) or not 0<timeout_s<=5 or type(sleep_s) not in (int,float) or not 0<sleep_s<=.1:raise ValueError('Bounded metadata retry settings required')
 path=Path(path);data=canonical(value);started=time.monotonic();attempts=0
 fd,name=tempfile.mkstemp(prefix='.'+path.name+'.publish-',suffix='.tmp',dir=path.parent);temporary=Path(name)
 try:
  with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
  while True:
   attempts+=1
   try:os.replace(temporary,path);return
   except PermissionError:
    METADATA_STATS['permission_retries']+=1
    elapsed=time.monotonic()-started
    if elapsed>=timeout_s:raise MetadataPublicationError(path,attempts,elapsed) from None
    time.sleep(min(sleep_s,timeout_s-elapsed))
 finally:
  try:temporary.unlink()
  except FileNotFoundError:pass
  except OSError:pass # An OS-held owned temp is preserved; no recursive cleanup.

def bounded_request(path):
 path=ordinary(path)
 if not path.is_file() or path.stat().st_size>8192:raise ValueError('Bounded private request file required')
 def pairs(items):
  value={}
  for k,v in items:
   if k in value:raise ValueError('Duplicate request member')
   value[k]=v
  return value
 def bad(_):raise ValueError('Nonfinite JSON')
 return json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_constant=bad)
def safe_text(value,limit=600):
 if type(value) is not str:return ''
 try:value.encode('utf8')
 except UnicodeError:return ''
 value=re.sub('(?<![A-Za-z0-9_:/])(?:[A-Za-z]:[\\\\/][^\\s]+|\\\\\\\\[^\\s]+|/(?!/)[^\\s]+)','[local path omitted]',value)
 return value[:limit]

class FileCancel:
 def __init__(self,path,deadline,external=None):self.path=Path(path);self.deadline=deadline;self.external=external;self.cancelled=False
 def is_set(self):return self.cancelled or self.path.exists() or time.monotonic()>=self.deadline or bool(self.external is not None and self.external.is_set())
 def set(self):self.cancelled=True

class ModeEpisode(PersistentEpisode):
 def capabilities(self):return {**super().capabilities(),'controller_mode':self.backend.hybrid_config['mode'],'direct_servo_arm_joint_names':list(self.backend.hybrid_config['arm_joint_names']),'actuation_qualification':'Arms classicalPD+velocity+sampledbias;15legs/trunkjoints fixedpretrainedactor. No source choreography playback; no learnednewarm policy.'}

class Progress:
 def __init__(self,output,prepared):
  self.output=Path(output);self.prepared=prepared;self.started=time.monotonic();self.status='preparing';self.phase='preparing';self.events=[];self.backend=None;self.episode=None;self.fork=None;self.goal_spec=None;self.actual={};self.preview=None;self.model_decision=None;self.media={};self.view=None;self.visual_unavailable=False;self.phase_results=[];self._event_index=0;self.initialization_started=False
 def event(self,kind,message):
  self.events.append({'index':self._event_index,'kind':kind,'message':safe_text(message,300),'wall_seconds':round(time.monotonic()-self.started,3)});self._event_index+=1;self.events=self.events[-80:]
 def counts(self):
  attempts=list(self.output.glob('phase_*/model/*/attempt.json'));receipts=[]
  for p in self.output.glob('phase_*/model/*/receipt.json'):
   try:
    if p.stat().st_size<=128*1024:receipts.append(json.loads(p.read_bytes()))
   except (OSError,ValueError):pass
  actual_steps=int(getattr(self.backend,'step_count',0));actual_online=sum(row.get('role')=='control' and row.get('status')=='completed' for row in getattr(self.backend,'calls',[]));warm=sum(row.get('role')=='warmup' and row.get('status')=='completed' for row in getattr(self.backend,'calls',[]));sandbox_steps=sandbox_online=previews=commits=0
  if self.fork is not None:
   previews=len(self.fork._previews);commits=len(self.fork._commits)
   for entry in self.fork._previews.values():
    branch=entry['branch'];origin=branch.branch_provenance['parent_snapshot']['control_index'];sandbox_steps+=int(branch.backend.step_count)-origin*20;sandbox_online+=sum(row.get('role')=='control' and row.get('status')=='completed' for row in branch.backend.calls[origin+1:])
  return {'model_calls':sum(r.get('live') is True and r.get('completion_unconfirmed') is False for r in receipts),'model_attempts':len(attempts),'model_completion_unconfirmed':max(0,len(attempts)-len(receipts)),'actual_commits':commits,'preview_attempts':previews,'actual_steps':actual_steps,'sandbox_steps':sandbox_steps,'actual_online_ONNX':actual_online,'sandbox_online_ONNX':sandbox_online,'warmup_ONNX':warm,'source_initializations':int(getattr(self.backend,'initialization_count',0)),'wall_seconds':round(time.monotonic()-self.started,3),'visual_renders':int(getattr(self.view,'count',0)),'metadata_permission_retries':METADATA_STATS['permission_retries'],'physical_completion_unconfirmed':bool(self.initialization_started and self.backend is None or any(row.get('status')!='completed' for row in getattr(self.backend,'calls',[])) or any(r.get('completion_unconfirmed') is not None for r in self.phase_results))}
 def assess(self,observation):
  task=self.goal_spec['final_task'];report=evaluate_dual(task,observation) if 'goals' in task else evaluate_single(task,observation)
  rows=report['goals'] if 'goals' in report else [{**report,'body':task['body'],'target_world_m':task['target_world_m'],'tolerance_m':task['tolerance_m']}]
  goals=[{'label':'Sağ el' if row['body']=='right_rubber_hand' else 'Sol el','body':row['body'],'distance_m':row['distance_m'],'tolerance_m':row['tolerance_m'],'done':row['done']} for row in rows]
  return {'sim_time_s':observation['sim_time_s'],'goals':goals,'task_verified':report['done'],'physical_gate_passed':report['physical_gate_passed']},report
 def capture(self,episode,role):
  if self.view is None or self.visual_unavailable:return
  try:
   final=self.goal_spec['final_task'];goals=final.get('goals',[final]);descriptor=self.view.capture(episode,goals,self.output/'images',role)
   if descriptor:self.media['actual.jpg' if role=='actual' else 'preview.jpg']=descriptor
  except Exception as error:
   from pulsemorph.agentic.embodied.app.live_view_v1 import LiveViewIntegrityError
   if isinstance(error,LiveViewIntegrityError):raise
   self.visual_unavailable=True;self.event('visual_unavailable','Canlı görüntü alınamadı; fizik ve ham kanıt ayrı korunur.')
 def publish(self):
  if self.episode is not None:self.actual,_=self.assess(self.episode.observe())
  public={'schema':'pulsemorph.body_agent_progress.v1','status':self.status,'phase':self.phase,'actual':self.actual,'preview':self.preview,'counts':self.counts(),'model_decision':self.model_decision,'events':self.events,'media':copy.deepcopy(self.media),'qualification':'Gerçek ve varsayımsal durumlar ayrıdır. Sabit hedefte gözlenmiş endpoint; sürekli tutuş/öğrenme/donanım garantisi yoktur.'}
  atomic(self.output/'progress.json',public);atomic(self.output/'public.json',public)

class ProgressProvider:
 def __init__(self,provider,progress):self.provider=provider;self.progress=progress;self.model=provider.model
 def status(self):return self.provider.status()
 def generate_json(self,*args,**kwargs):
  self.progress.status='running';self.progress.event('model_thinking','Model gerçek beden ve önceki sonuçları değerlendiriyor.');self.progress.publish()
  result=self.provider.generate_json(*args,**kwargs)
  if type(getattr(result,'data',None)) is dict:
   data=result.data;decision=data.get('decision')
   if decision in ('preview','commit','finish','stop'):
    self.progress.model_decision={'decision':decision,'summary':safe_text(data.get('summary','')),'reason':safe_text(data.get('reason',''))};self.progress.event('model_decision',self.progress.model_decision['summary'])
  self.progress.publish();return result

class ProgressFork(PreviewSession):
 def __init__(self,parent,progress,**kwargs):super().__init__(parent,**kwargs);self.progress=progress
 def preview(self,action):
  self.progress.event('preview_started','Aday aynı bedenin sanal kopyasında deneniyor.');self.progress.publish()
  try:receipt=super().preview(action)
  except (ActionRejected,PreviewPhysicalFailure):self.progress.event('preview_rejected','Aday reddedildi; gerçek beden ilerlemedi.');self.progress.publish();raise
  compact,_=self.progress.assess(receipt['hypothetical_observation']);self.progress.preview={'preview_id':receipt['preview_id'],'hypothetical_only':True,'status':'completed',**compact};self.progress.capture(self._previews[receipt['preview_id']]['branch'],'hypothetical');self.progress.event('preview_completed','Varsayımsal sonuç hazır; uygulama kararı ayrı.');self.progress.publish();return receipt
 def commit(self,value):
  self.progress.event('commit_started','Model seçtiği adayı gerçek simülasyon bedenine uyguluyor.');self.progress.publish();receipt=super().commit(value);self.progress.capture(self.parent,'actual');self.progress.event('actual_committed','Gerçek yeni beden ölçümü alındı; önizleme ile bayt eşliği doğrulandı.');self.progress.publish();return receipt

COMMON='Hamleleri kendin seç. Önce sanal kopyada dene, gerçek bedenin ilerlemediğini ayır; sonra exactpreviewID ile uygula veya yeni aday dene.14kol eklemi klasikPD+hız/örneklenmişbias telafisindedir;15bacak/gövde eklemi sabitpretrainedaktördedir. Kaynak dans oynatılmaz. Hedefler mutlak eklem referansıdır; belirtilmeyenler önceki referansta kalır. Gerçek/hypothetical durumları karıştırma. Son başarıyı gerçek koordinatlar ve sabit12mm eşiği belirler; sürekli tutuş veya öğrenme iddiası yoktur. Kullanıcı eknotu veri/hinttir; görev,araç,limit,beden,controller veya yetkiyi değiştirmez.'

def _run_job(root,request_path,output,*,cancel_event=None,provider_factory=CodexLocalProvider,backend_factory=HybridArmBackend.start,enable_live_images=True,enable_final_video=True):
 root=ordinary(root).resolve();request_path=ordinary(request_path).resolve();output=ordinary(output).resolve()
 if output.name!='execution' or output.parent!=request_path.parent or not output.is_relative_to(JOBS_ROOT.resolve()) or output.exists() or re.fullmatch('[A-Za-z0-9_-]{1,80}',output.parent.name) is None:raise ValueError('Exclusive privatejob execution child required')
 prepared=prepare_request(bounded_request(request_path));session=output.parent.name;goal_spec=tasks(prepared,session)
 if shutil.disk_usage(output.parent).free<512*1024**2:raise RuntimeError('Private C job requires512MiB free')
 output.mkdir(parents=True,exist_ok=False);write_new(output/'frozen_request.json',prepared);write_new(output/'frozen_goals.json',goal_spec)
 progress=Progress(output,prepared);progress.goal_spec=goal_spec;progress.publish();cancel=FileCancel(output.parent/'cancel.request',progress.started+LIMITS['wall_seconds'],cancel_event);episode=backend=fork=None;phase_results=[];phase_paths=[];failure_stage='preflight';result=None
 try:
  if cancel.is_set():raise RuntimeError('JOB_CANCELLED_BEFORE_EXECUTION')
  pins,environment=build_pins(root,include_environment=True)
  for path in (request_path,output/'frozen_request.json',output/'frozen_goals.json'):pins[str(path)]=sha(path)
  authority={'schema':'pulsemorph.embodied_execution_authority.v1','session_id':session,'max_control_steps':500,'source_pins':pins};write_new(output/'authority.json',authority);write_new(output/'environment_preflight.json',environment)
  failure_stage='initializing';progress.initialization_started=True;progress.phase='initializing';progress.publish();backend=backend_factory(root,authority=authority);episode=ModeEpisode(backend,session,max_control_steps=500,cancel_event=cancel);progress.backend=backend;progress.episode=episode
  if backend.environment!=environment:raise RuntimeError('Runtime numerical environment differs from preflight')
  if enable_live_images:
   from pulsemorph.agentic.embodied.app.live_view_v1 import LiveView
   progress.view=LiveView(LIMITS['max_live_images']);progress.capture(episode,'actual')
  write_new(output/'initial_observation.json',episode.observe());progress.status='running';progress.phase='settling';progress.event('initialized','Tek fizik bedeni ve tek aktör başlatıldı; kaynak yalnız ilk poz için kullanıldı.');progress.publish()
  failure_stage='prelude'
  for i in range(2):
   if cancel.is_set():raise RuntimeError('JOB_CANCELLED')
   obs=episode.observe();name='right_shoulder_roll_joint';q=obs['previous_reference']['q'][backend.joint_names.index(name)];action={'action_id':'prelude_%02d'%i,'expected_revision':obs['revision'],'observation_sha256':fingerprint(obs),'tool':'move_joints','arguments':{'duration_s':1.,'joint_targets_rad':{name:q}}};write_new(output/('prelude_action_%02d.json'%i),action);write_new(output/('prelude_transition_%02d.json'%i),episode.execute(action))
  progress.capture(episode,'actual');write_new(output/'prelude_final_observation.json',episode.observe());progress.publish()
  fork=ProgressFork(episode,progress,max_preview_controls=50,max_total_preview_controls=400);progress.fork=fork
  phases=[('right',goal_spec['right_phase'],evaluate_single,'SAĞ elini sabit sağ hedefin12mm yakınına getir. Diğer referansları mümkün olduğunca koru.')]
  if prepared['scenario']=='both_reach':phases.append(('both',goal_spec['both_phase'],evaluate_dual,'Şimdi SOL elini sol hedefe getir; sonunda SAĞ el de sağ hedefin12mm yakınında olsun. Başarı iki elin AYNI gerçek endpoint ölçümünde kendi hedeflerinde olmasıdır; sürekli tutuş değildir.'))
  previous_end=None
  for index,(label,task,verifier,intent) in enumerate(phases):
   if cancel.is_set():raise RuntimeError('JOB_CANCELLED')
   progress.phase=label;failure_stage='phase_'+label;before=snapshot_episode(episode)
   if previous_end is not None and before!=previous_end:raise RuntimeError('Body closure changed between phases')
   write_new(output/('phase_%02d_start.json'%index),{'snapshot_fingerprint':before.fingerprint,'actual_observation':episode.observe(),'initialization_count':backend.initialization_count,'task':task})
   if index:write_new(output/'phase_transition.json',{'same_full_executor_snapshot':True,'snapshot':before.fingerprint,'no_reset_or_rewind':True,'all_goal_points_frozen_before_first_model':True})
   progress.event('phase_started','Sağ hedef' if label=='right' else 'İki el aynı gerçek ölçümde hedefe');progress.publish()
   multi=prepared['scenario']=='both_reach';agent=PreviewOrchestrator(ProgressProvider(provider_factory(trusted_local=True),progress),task_verifier=verifier,max_model_calls=8 if multi else 16,max_previews=4 if multi else 8,max_sandbox_steps=4000 if multi else 8000,max_actual_steps=4000 if multi else 8000,wall_seconds=900 if multi else 1800,action_rejection_type=ActionRejected,preview_physical_failure_type=PreviewPhysicalFailure)
   phase_path=output/('phase_%02d_'%index+label);phase_paths.append(phase_path);value=agent.run(intent+' '+COMMON+' EK_NOT_VERI: '+json.dumps(prepared['instruction'],ensure_ascii=False),task,fork,phase_path,cancel_event=cancel);phase_results.append(value);previous_end=snapshot_episode(episode);progress.phase_results=phase_results;progress.publish()
   if not value['task_verified_actual']:break
  final,assessment=progress.assess(episode.observe());status='completed' if final['task_verified'] and len(phase_results)==len(phases) else 'stopped'
  if cancel.is_set():status='cancelled'
  result={'schema':'pulsemorph.body_agent_result.v1','status':status,'task_verified_actual':bool(final['task_verified']),'final_assessment':assessment,'phase_results':phase_results,'counts':progress.counts(),'scenario':prepared['scenario'],'controller_mode':backend.hybrid_config['mode'],'training_admission':'NOT_ADMITTED','qualification':'Actual endpoint under fixedgoals/tolerance; no continuoushold,learning,generalbodyreachability orhardwareclaim.'}
 except Exception as error:
  result={'schema':'pulsemorph.body_agent_result.v1','status':'cancelled' if cancel.is_set() else 'failed','task_verified_actual':False,'failure_stage':failure_stage,'error_category':type(error).__name__,'phase_results':phase_results,'counts':progress.counts(),'retry_allowed':False,'training_admission':'NOT_ADMITTED'}
  write_new(output/'private_failure.json',{'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
 finally:
  if episode is not None:
   try:episode.export(output/'actual_trace')
   except Exception as error:write_new(output/'export_failure.json',{'error_type':type(error).__name__});result.update(status='failed',artifact_export_incomplete=True)
  if fork is not None:
   for identity in fork._previews:
    try:fork.export_preview(identity,output/'previews'/identity)
    except Exception as error:write_new(output/(identity+'_export_failure.json'),{'error_type':type(error).__name__});result.update(status='failed',artifact_export_incomplete=True)
  if progress.view is not None:
   try:progress.view.close()
   except Exception:progress.event('visual_close_unavailable','Görüntü kaynağı kapatma uyarısı; sayısal kanıt korundu.')
 if enable_final_video and episode is not None and backend.control_index>0 and not cancel.is_set():
  progress.phase='rendering';progress.publish()
  try:
   from pulsemorph.agentic.embodied.app.renderer_v1 import render_actual_trace
   media=render_actual_trace(output/'actual_trace',goal_spec,phase_paths,output/'media',cancel_event=cancel,max_fk=LIMITS['final_video_render_FK']);video=Path(media['video']);progress.media['episode.mp4']={'path':str(video.relative_to(output)),'sha256':sha(video),'role':'actual_recorded'};result['video_render_FK']=media['render_FK']
  except Exception as error:
   write_new(output/'video_failure.json',{'error_type':type(error).__name__});result.update(status='failed',media_incomplete=True)
 progress.status=result['status'];progress.phase='finished';progress.event('finished','Gerçek hedefler doğrulandı.' if result['task_verified_actual'] else 'Bütçeli deneme bitti; başarılmayan hedefler korunur.');result['counts']=progress.counts();result['media']=copy.deepcopy(progress.media);result['live_visual_renders']=int(getattr(progress.view,'count',0));write_new(output/'terminal_numeric_result.json',result);progress.publish();write_new(output/'result.json',result)
 return result

def _emergency_read(path):
 try:
  path=Path(path)
  if path.stat().st_size<=256*1024:return json.loads(path.read_bytes())
 except (OSError,ValueError):pass
 return None

def run_job(root,request_path,output,**kwargs):
 """Same job API; a terminal metadata error never causes an execution retry."""
 try:return _run_job(root,request_path,output,**kwargs)
 except Exception as error:
  output=ordinary(output).resolve()
  if output.is_relative_to(JOBS_ROOT.resolve()) and output.name=='execution' and output.is_dir():
   checkpoint=_emergency_read(output/'terminal_numeric_result.json')
   if type(checkpoint) is dict:
    failure=copy.deepcopy(checkpoint);failure.update(status='failed',terminal_publication_failed=True,failure_stage='metadata_publication',error_category=type(error).__name__)
   else:
    last=_emergency_read(output/'progress.json') or {}
    counts=copy.deepcopy(last.get('counts',{}));counts['physical_completion_unconfirmed']=True
    failure={'schema':'pulsemorph.body_agent_result.v1','status':'failed','task_verified_actual':False,'task_assessment_unconfirmed':True,'counts':counts,'counts_are_last_published':True,'failure_stage':'metadata_publication' if isinstance(error,MetadataPublicationError) else 'worker_exception','error_category':type(error).__name__,'retry_allowed':False}
   record={'schema':'pulsemorph.body_agent_terminal_publication_failure.v1','error_category':type(error).__name__,'retry_scope':'metadata replacement only; no provider/action/simulator retry','metadata_permission_retries':METADATA_STATS['permission_retries'],'numeric_checkpoint_preserved':type(checkpoint) is dict}
   if isinstance(error,MetadataPublicationError):record.update(target_name=error.target_name,attempts=error.attempts,elapsed_seconds=error.elapsed_seconds)
   for name,value in [('terminal_publication_failure.json',record),('result.json',failure)]:
    try:write_new(output/name,value)
    except OSError:pass # Preserve existing files and the exception, never overwrite.
  raise

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 result=run_job(ROOT,args.request,args.output);print(json.dumps({k:result[k] for k in ('status','task_verified_actual')},allow_nan=False));return 0 if result['status'] in ('completed','stopped','cancelled') else 1
if __name__=='__main__':raise SystemExit(main())
