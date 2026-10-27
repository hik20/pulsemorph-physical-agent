"""One private bounded LLM/music/whole-body job; import constructs no motor."""
from pathlib import Path
import argparse,copy,json,hashlib,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[5];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions')]
from pulsemorph.agentic.embodied.app.assets_v1 import build_pins
from pulsemorph.agentic.embodied.app.worker_v4 import atomic,FileCancel,safe_text,METADATA_STATS
from pulsemorph.agentic.app.providers import CodexLocalProvider
from pulsemorph.agentic.performance.backend_v1 import PerformanceBackend
from pulsemorph.agentic.performance.runtime_v1 import PerformanceEpisode,PerformancePreviews,HoldCurve
from pulsemorph.agentic.performance.music_v1 import create_demo_music
from pulsemorph.agentic.performance.orchestrator_v1 import PerformanceOrchestrator,evaluate_episode
from pulsemorph.agentic.performance.render_v1 import render_performance
from pulsemorph.agentic.embodied.runtime_v1 import canonical,fingerprint

CACHE=Path.home()/'.cache/pulsemorph-agentic';JOBS=CACHE/'performance-app-v1/jobs'
LIMITS={'model_calls':8,'physics_previews':3,'actual_controls':940,'sandbox_controls':2520,'warmup_ONNX':1,'source_initializations':1,'actual_steps':18800,'sandbox_steps':50400,'final_render_FK':420,'wall_seconds':2100}
DEFAULT='Müzikle birlikte belirgin, akıcı ve eğlenceli tam beden robot dansı üret. Kolları geniş kullan; gövde, kalça, diz ve ayak bilekleri ritmik ve anlamlı biçimde katılsın. Yanlara ağırlık aktar; farklı hareket cümleleri ve bilinçli bir bitiriş yap. Fizik önizlemesinden gelen ölçümlere göre kendi programını geliştir.'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('xb') as stream:stream.write(canonical(value))
def ordinary(p):
    p=Path(p).absolute()
    for x in (p,*p.parents):
        if x.is_symlink() or x.exists() and getattr(x.lstat(),'st_file_attributes',0)&1024:raise ValueError('Linked private path forbidden')
    return p
def implementation_pins(root,include_environment=False):
    pins,env=build_pins(root,include_environment=include_environment)
    paths=list((root/'src/pulsemorph/agentic/performance').rglob('*.py'))+[root/'src/pulsemorph/agentic/app/renderer.py']
    for p in paths:pins[str(ordinary(p).resolve())]=sha(p)
    return pins,env
def profile_status(root):
    try:
        pins,_=implementation_pins(Path(root).resolve());return {'available':False,'state':'whole_body_control_not_admitted','scope':'Whole-body music generation remains developmental; use the recorded review and validated body-agent tasks.','pin_count':len(pins),'numerical_environment_checked_in_worker':False}
    except (OSError,ValueError,RuntimeError):return {'available':False,'state':'prepared_music_body_profile_unavailable'}
def request_value(path):
    p=ordinary(path)
    if p.stat().st_size>8192:raise ValueError('Bounded instruction file required')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('Duplicate request member')
            result[k]=v
        return result
    def bad(_):raise ValueError('Nonfinite request')
    value=json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=bad)
    if type(value) is not dict or set(value)!={'instruction'} or type(value['instruction']) is not str or len(value['instruction'])>1500:raise ValueError('Exact bounded instruction required')
    value['instruction'].encode('utf8');return {'instruction':value['instruction'].strip() or DEFAULT}

class Progress:
    def __init__(self,output):
        self.output=output;self.started=time.monotonic();self.status='preparing';self.phase='Hazırlanıyor';self.events=[];self.backend=None;self.episode=None;self.pool=None;self.actual=None;self.preview=None;self.plan=None;self.model_decision=None;self.media={};self.render_fk=0;self.initialization_started=False
    def counts(self):
        b=self.backend;pool=self.pool;calls=getattr(b,'calls',[]);entries=[] if pool is None else list(pool.entries.values())
        attempts=list((self.output/'agent').glob('model_*/attempt.json'));receipts=[]
        for p in (self.output/'agent').glob('model_*/receipt.json'):
            try:receipts.append(json.loads(p.read_bytes()))
            except (OSError,ValueError):pass
        return {'model_calls':len(receipts),'model_attempts':len(attempts),'model_completion_unconfirmed':max(0,len(attempts)-len(receipts)),
          'actual_online_ONNX':sum(r['role']=='control' and r['status']=='completed' for r in calls),'sandbox_online_ONNX':sum(sum(r['role']=='control' and r['status']=='completed' for r in e['branch'].backend.calls[e['origin']['calls']:]) for e in entries),
          'actual_steps':int(getattr(b,'step_count',0)),'sandbox_steps':sum(e['branch'].backend.step_count-e['origin']['step_count'] for e in entries),'warmup_ONNX':sum(r['role']=='warmup' and r['status']=='completed' for r in calls),
          'source_initializations':int(getattr(b,'initialization_count',0)),'actual_commits':0 if pool is None else len(pool.commits),'physics_previews':len(entries),'render_FK':self.render_fk,'wall_seconds':round(time.monotonic()-self.started,3),'metadata_permission_retries':METADATA_STATS['permission_retries'],
          'physical_completion_unconfirmed':bool(self.initialization_started and b is None or any(r.get('status')!='completed' for r in calls) or any(any(r.get('status')!='completed' for r in e['branch'].backend.calls[e['origin']['calls']:]) for e in entries))}
    def publish(self):
        atomic(self.output/'progress.json',{'status':self.status,'phase_label':self.phase,'events':self.events[-100:],'counts':self.counts(),'actual':self.actual,'preview':self.preview,'plan':self.plan,'model_decision':self.model_decision,'media':self.media,'qualification':'Recorded simulation; actual/hypothetical separated. Full music available offline; not real-time hardware control or learned new weights.'})
    def event(self,kind,message,values=None):
        self.events.append({'index':len(self.events),'kind':kind,'message':safe_text(message,1000),'wall_seconds':round(time.monotonic()-self.started,3)})
        values=values or {}
        phases={'model_thinking':'LLM hareketi planlıyor','model_decision':'LLM sonraki adımı seçti','compile_rejected':'LLM hareket sınırlarını düzeltiyor','preview_started':'Tam beden fizik önizlemesi','preview_completed':'Fizik geri bildirimi değerlendiriliyor','preview_failed':'LLM fizik sonucuna göre düzeltecek','committing':'Seçilen dans bedene uygulanıyor','actual_completed':'Gerçek performans ölçüldü'}
        self.phase=phases.get(kind,self.phase)
        if kind=='model_decision':self.model_decision={'decision':values.get('decision'),'summary':safe_text(message,1000),'reason':safe_text(values.get('reason',''),1600)}
        if kind in ('preview_started','compile_rejected'):
            candidates=sorted((self.output/'agent').glob('model_*/mapped_program.json'))
            if candidates:
                path=candidates[-1];program=json.loads(path.read_bytes());report_path=path.with_name('compile_report.json');report=json.loads(report_path.read_bytes()) if report_path.is_file() else values.get('compile_report',{})
                self.plan={'summary':self.model_decision.get('summary','') if self.model_decision else '', 'keyframe_count':len(program['keyframes']),'activation':report.get('activation',{}),'program_sha256':fingerprint(program)}
        if kind in ('preview_completed','preview_failed'):
            row=values['preview'];assessment=row.get('assessment',{});feedback=assessment.get('feedback',{});evidence=assessment.get('demo_minimum_evidence',{})
            deficits=[key+': '+format(e['observed'],'.3f')+' / '+format(e['minimum'],'.3f') for key,e in evidence.items() if not e['passed']]
            self.preview={'hypothetical_only':True,'preview_id':row.get('preview_id'),'summary':safe_text(message),'deficits':deficits,'physical_gate_passed':feedback.get('physical_gate_passed',False),'report':assessment}
        if kind=='actual_completed':self.set_actual(values['assessment'])
        self.publish()
    def set_actual(self,report):
        self.actual={'summary':'LLM’nin seçtiği gerçek simülasyon performansı ölçüldü.' if report['full_window_complete'] else 'Gerçek beden hazır; dans henüz uygulanmadı.','duration_s':16.8,'sim_time_s':self.backend.sim_time_s,'group_metrics':report['actual_groups'],'physical_gate_passed':report['physical_gate']['passed'],'performance_screen_passed':report['performance_screen_passed'],'report':report}

def run_job(root,request_path,output,*,cancel_event=None):
    root=ordinary(root).resolve();request_path=ordinary(request_path).resolve();output=ordinary(output).resolve()
    if output.name!='execution' or output.parent!=request_path.parent or not output.is_relative_to(JOBS.resolve()) or output.exists():raise ValueError('New exclusive performance job execution directory required')
    if shutil.disk_usage(output.parent).free<512*1024**2:raise RuntimeError('Private output requires512MiB free')
    request=request_value(request_path);output.mkdir(parents=True,exist_ok=False);write(output/'fixed_request.json',{**request,'limits':LIMITS});p=Progress(output);p.publish();cancel=FileCancel(output.parent/'cancel.request',p.started+LIMITS['wall_seconds'],cancel_event);result=None;episode=None;pool=None;music=None;stage='preflight'
    try:
        if cancel.is_set():raise RuntimeError('CANCELLED_BEFORE_EXECUTION')
        music=create_demo_music(output/'audio');shutil.copyfile(music['path'],output/'music.wav');p.media['music.wav']={'path':'music.wav','role':'planning_music','sha256':sha(output/'music.wav')};p.publish()
        pins,environment=implementation_pins(root,True)
        for path in (request_path,output/'fixed_request.json',Path(music['path']),output/'audio/descriptor.json'):pins[str(path)]=sha(path)
        authority={'schema':'pulsemorph.performance_execution_authority.v1','session_id':output.parent.name,'max_control_steps':LIMITS['actual_controls'],'source_pins':pins};write(output/'authority.json',authority);write(output/'environment_preflight.json',environment)
        stage='initializing';p.initialization_started=True;p.phase='Fizik bedeni hazırlanıyor';p.publish();b=PerformanceBackend.start(root,authority=authority);p.backend=b
        if b.environment!=environment:raise RuntimeError('Numerical environment changed')
        episode=PerformanceEpisode(b,authority['session_id'],music=music,cancel_event=cancel);p.episode=episode;p.status='running';p.phase='Beden başlangıç duruşuna yerleşiyor';p.publish();hold=HoldCurve(episode.reference_q,2.)
        episode.execute_curve({'action_id':'root_settle','expected_revision':episode.revision,'observation_sha256':fingerprint(episode.observe()),'program_sha256':hold.program_sha256},hold)
        pool=PerformancePreviews(episode,max_previews=3,max_total_preview_controls=2520);p.pool=pool;p.set_actual(evaluate_episode(episode,music,2.));p.publish();stage='agent'
        run=PerformanceOrchestrator(CodexLocalProvider(trusted_local=True),max_model_calls=8).run(pool,music,request['instruction'],output/'agent',cancel_event=cancel,on_event=p.event)
        result={'schema':'pulsemorph.performance_app_result.v1','status':run['status'],'performance_screen_passed':run['performance_screen_passed'],'agent_reason':run['reason'],'counts':p.counts(),'training_admission':'NOT_ADMITTED'};write(output/'numeric_result.json',result)
    except Exception as error:
        result={'schema':'pulsemorph.performance_app_result.v1','status':'cancelled' if cancel.is_set() else 'failed','performance_screen_passed':False,'failure_stage':stage,'error_type':type(error).__name__,'counts':p.counts(),'retry_allowed':False,'training_admission':'NOT_ADMITTED'};write(output/'private_failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    finally:
        if episode is not None:
            try:episode.export(output/'actual_trace')
            except Exception as error:result.update(status='failed',artifact_export_incomplete=True);write(output/'export_failure.json',{'type':type(error).__name__})
        if pool is not None:
            try:pool.export(output/'previews')
            except Exception as error:result.update(status='failed',artifact_export_incomplete=True);write(output/'preview_export_failure.json',{'type':type(error).__name__})
    if result['status']=='completed' and not cancel.is_set():
        p.phase='Müzikli performans kaydı hazırlanıyor';p.publish()
        try:
            video=render_performance(output/'actual_trace',output/'render',music=music,start_s=2.,duration_s=16.8,max_fk=420,label='LLM koreografisi · müzik ve fizik geri bildirimi',cancel_event=cancel)
            p.render_fk=video['render_FK'];shutil.copyfile(video['video'],output/'episode.mp4')
            from PIL import Image
            with Image.open(video['last_frame']) as frame:
                with (output/'actual.jpg').open('xb') as stream:frame.convert('RGB').save(stream,format='JPEG',quality=90)
            for name,role in [('episode.mp4','actual_recorded'),('actual.jpg','actual')]:p.media[name]={'path':name,'role':role,'sha256':sha(output/name)}
        except Exception as error:
            attempt=output/'render/attempt.json'
            if attempt.is_file():p.render_fk=json.loads(attempt.read_bytes()).get('render_FK',0)
            result.update(status='failed',media_incomplete=True);write(output/'video_failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    if cancel.is_set():result['status']='cancelled'
    if episode is not None and music is not None:
        try:p.set_actual(evaluate_episode(episode,music,2.))
        except ValueError:pass
    p.status=result['status'];p.phase='Performans hazır' if result['status']=='completed' else 'Çalışma sona erdi';result.update(counts=p.counts(),media=copy.deepcopy(p.media));write(output/'terminal_numeric_result.json',result);p.publish();write(output/'result.json',result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=run_job(ROOT,args.request,args.output);print(json.dumps({k:result[k] for k in ('status','performance_screen_passed')}));return 0 if result['status'] in ('completed','stopped','cancelled') else 1
if __name__=='__main__':raise SystemExit(main())
