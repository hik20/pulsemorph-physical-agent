"""Process-owned Body Agent Lab jobs; actual and hypothetical work stay distinct."""
import copy,hashlib,json,os,re,subprocess,threading,time,uuid
from datetime import datetime,timezone
from pathlib import Path

CACHE=Path.home()/'.cache/pulsemorph-agentic'
JOB_ROOT=CACHE/'body-agent-v1/jobs'
PUBLIC_PYTHON=CACHE/'envs/app-public-win-cp311-v1/Scripts/python.exe'
TERMINAL={'completed','failed','cancelled','stopped'}
TARGET=[.16240383871601374,-.14286599443025425,.6080935474884148]
class BodyJobBusy(ValueError):pass
def now():return datetime.now(timezone.utc).isoformat()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ordinary(p):
    p=Path(p).absolute()
    for x in (p,*p.parents):
        if x.is_symlink() or x.exists() and getattr(x.lstat(),'st_file_attributes',0)&1024:raise ValueError('Linked job/media path forbidden')
    return p
def read_json(p,limit=256*1024):
    p=ordinary(p)
    if p.stat().st_size>limit:raise ValueError('Bounded JSON required')
    value=json.loads(p.read_bytes());encoded(value);return value
def write_new(p,value):
    with Path(p).open('xb') as f:f.write(encoded(value))
def parse_request(v):
    if type(v) is not dict or set(v)!={'scenario','instruction'} or v['scenario'] not in ('right_reach','both_reach') or type(v['instruction']) is not str or len(v['instruction'])>1500:raise ValueError('Exact bounded scenario/instruction required')
    v['instruction'].encode('utf8');return {'scenario':v['scenario'],'instruction':v['instruction'].strip()}
def fixed_goals(scenario):
    bodies=['right_rubber_hand'] if scenario=='right_reach' else ['right_rubber_hand','left_rubber_hand']
    return [{'body':b,'frame':'world','target_world_m':[TARGET[0],TARGET[1] if b.startswith('right') else abs(TARGET[1]),TARGET[2]],'tolerance_m':.012} for b in bodies]
def media_relative(name,relative):
    if type(relative) is not str:return False
    if name=='episode.mp4':return relative in ('episode.mp4','media/episode.mp4')
    prefix='actual' if name=='actual.jpg' else 'preview'
    return relative==name or re.fullmatch('images/'+prefix+'_[0-9]{3}\.jpg',relative) is not None
def clean_progress(v):
    if type(v) is not dict:raise ValueError('Progress object required')
    allowed=('schema','status','phase','counts','goals','goal_errors','assessment','model_decision','decision','sim_time_s','wall_seconds','preview','events','actual','sandbox','actual_counts','sandbox_counts','model_counts','error_category','task_verified_actual','qualification')
    result={k:copy.deepcopy(v[k]) for k in allowed if k in v}
    raw=encoded(result)
    if len(raw)>128*1024:raise ValueError('Bounded public progress required')
    # Worker is trusted local code; nevertheless paths/session values never
    # become public just because an unexpected progress member was added.
    def visit(x):
        if type(x) is dict:
            for key,item in x.items():
                if type(key) is not str or re.search('(?i)(path|token|secret|password|authorization|credential)',key):raise ValueError('Private progress member forbidden')
                visit(item)
        elif type(x) is list:
            if len(x)>1000:raise ValueError('Progress list too large')
            for item in x:visit(item)
        elif type(x) is str:
            if len(x)>3000 or re.search(r'[A-Za-z]:[\\/]',x):raise ValueError('Private/large progress text forbidden')
    visit(result);return result

class BodyAgentManager:
    def __init__(self,root,*,jobs_dir=JOB_ROOT,python=PUBLIC_PYTHON,worker=None,profile_status=None,popen=subprocess.Popen):
        self.root=Path(root).resolve();self.jobs_dir=ordinary(jobs_dir).resolve()
        if self.jobs_dir==CACHE.resolve() or not self.jobs_dir.is_relative_to(CACHE.resolve()):raise ValueError('Dedicated private C jobs root required')
        self.jobs_dir.mkdir(parents=True,exist_ok=True);self.python=ordinary(python);self.worker=ordinary(worker or Path(__file__).with_name('worker_v1.py'));self.profile_status=profile_status;self.popen=popen
        if self.python.resolve()!=PUBLIC_PYTHON.resolve():raise ValueError('Exact reviewed public Python required')
        self.lock=threading.RLock();self.jobs={};self.processes={};self.threads={};self.history_warnings=[];self.pins=self.implementation_pins();self._load()
    def implementation_pins(self):
        paths=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).with_name('web').glob('*'))+[self.worker,self.python]
        return {str(p.resolve()):sha(p) for p in paths if p.is_file()}
    def _load(self):
        for p in sorted(self.jobs_dir.glob('*/state.json'))[-100:]:
            try:
                j=read_json(p);request=parse_request(read_json(p.with_name('request.json')))
                if j.get('schema')!='pulsemorph.body_agent_job.v1' or j.get('id')!=p.parent.name or any(j.get(k)!=v for k,v in request.items()) or j.get('fixed_goals')!=fixed_goals(request['scenario']) or type(j.get('events')) is not list or type(j.get('created_at')) is not str:raise ValueError('History assignment differs')
                if j.get('status') not in TERMINAL:j.update(status='failed',finalized=not bool(j.get('worker_attempts')),error='PROCESS_INTERRUPTED',completion_unconfirmed=True)
                elif not j.get('finalized'):j.update(finalized=not bool(j.get('worker_attempts')),completion_unconfirmed=True)
                self.jobs[j['id']]=j
            except Exception:self.history_warnings.append('invalid_or_partial_body_job_preserved')
    def active_id(self):
        with self.lock:return next((j['id'] for j in self.jobs.values() if j['status'] not in TERMINAL or not j['finalized']),None)
    def status(self):
        try:
            if self.profile_status is not None:profile=self.profile_status()
            else:
                from .assets_v1 import status
                profile=status(self.root)
        except Exception:profile={'available':False,'state':'prepared_profile_unavailable'}
        current=self.implementation_pins()==self.pins
        public_profile={k:profile[k] for k in ('available','state','scope','controller_mode','numerical_environment_checked_in_worker') if k in profile}
        return {'ready':profile.get('available') is True and current and self.worker.is_file() and self.python.is_file(),'prepared_host':True,'profile':public_profile,'active_job':self.active_id(),'history_warnings':list(self.history_warnings),'scenarios':[{'id':'right_reach','label':'Sağ bileği hedefe götür'},{'id':'both_reach','label':'Önce sağ eli, sonra iki eli hedefe götür'}],'capability_scope':'Prepared host experimental body agent; endpoint observations with hold_s0, not continuous holding, general body control, learning or safety validation.'}
    def _save(self,j):
        j['updated_at']=now();p=self.jobs_dir/j['id']/'state.json';tmp=p.with_suffix('.tmp');tmp.write_bytes(encoded(j));tmp.replace(p)
    def _event(self,j,kind,message):j['events'].append({'at':now(),'kind':kind,'message':message})
    def get(self,identity,*,private=False):
        with self.lock:j=copy.deepcopy(self.jobs[identity])
        if private:return j
        keys=('mode','id','scenario','instruction','fixed_goals','status','finalized','phase','created_at','updated_at','events','progress','error','worker_attempts','completion_unconfirmed','actual_image_url','preview_image_url','video_url','counts','actual','goals','goal_errors','assessment','model_decision','preview','sim_time_s','wall_seconds','task_verified_actual')
        return {k:j[k] for k in keys if k in j}
    def list_jobs(self):
        with self.lock:ids=[j['id'] for j in sorted(self.jobs.values(),key=lambda j:j['created_at'],reverse=True)[:100]]
        return [self.get(i) for i in ids]
    def create(self,value):
        request=parse_request(value)
        with self.lock:
            if self.active_id():raise BodyJobBusy('Another body job is active')
            if not self.status()['ready']:raise ValueError('Prepared body profile unavailable')
            identity='body_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8];d=self.jobs_dir/identity;d.mkdir(exist_ok=False);write_new(d/'request.json',request)
            j={'schema':'pulsemorph.body_agent_job.v1','mode':'body_agent','id':identity,**request,'fixed_goals':fixed_goals(request['scenario']),'status':'queued','phase':'queued','finalized':False,'created_at':now(),'events':[],'error':None,'worker_attempts':0,'completion_unconfirmed':False,'source_pins':self.pins,'actual_image_url':None,'preview_image_url':None,'video_url':None}
            j['request_sha256']=sha(d/'request.json');self.jobs[identity]=j;self._event(j,'queued','Hedefler sabitlendi; yeni beden deneyi sıraya alındı.');self._save(j);thread=threading.Thread(target=self._run,args=(identity,),daemon=True);self.threads[identity]=thread;thread.start();return identity
    def cancel(self,identity):
        with self.lock:
            j=self.jobs[identity]
            if (j['status'] not in TERMINAL or not j['finalized']) and not (self.jobs_dir/identity/'cancel.request').exists():write_new(self.jobs_dir/identity/'cancel.request',{'requested_at':now()});self._event(j,'cancel_requested','İptal istendi; kısmi kanıt korunuyor.');self._save(j)
    def _progress(self,j,execution):
        path=execution/'progress.json'
        if not path.is_file():return
        raw=read_json(path);progress=clean_progress(raw);j['progress']=progress
        for event in progress.get('events',[]):
            if type(event) is not dict or type(event.get('index')) is not int or type(event.get('kind')) is not str or type(event.get('message')) is not str:raise ValueError('Typed safe worker event required')
            if event['index']>j.get('worker_event_max_index',-1):
                j['events'].append({'at':now(),'kind':event['kind'],'message':event['message'],'source':'worker','worker_index':event['index'],'wall_seconds':event.get('wall_seconds')});j['worker_event_max_index']=event['index']
        for key in ('phase','counts','actual','goals','goal_errors','assessment','model_decision','preview','sim_time_s','wall_seconds','task_verified_actual'):
            if key in progress:j[key]=copy.deepcopy(progress[key])
        for name,key,role in [('actual.jpg','actual_image_url','actual'),('preview.jpg','preview_image_url','hypothetical'),('episode.mp4','video_url','actual_recorded')]:
            item=raw.get('media',{}).get(name)
            if type(item) is not dict:continue
            relative=item.get('path',name)
            if not media_relative(name,relative):raise ValueError('Unknown worker media path')
            path=ordinary(execution/relative)
            if item.get('role')==role and item.get('sha256') and path.is_file() and path.stat().st_size<=(64*1024**2 if name.endswith('.mp4') else 2*1024**2) and sha(path)==item['sha256']:
                j.setdefault('media_integrity',{})[name]={'sha256':item['sha256'],'role':role,'path':relative};j[key]='/media/jobs/'+j['id']+'/'+name+'?v='+item['sha256'][:16]
        self._save(j)
    def _run(self,identity):
        d=self.jobs_dir/identity;execution=d/'execution';process=None
        try:
            with self.lock:
                j=self.jobs[identity]
                if (d/'cancel.request').exists():j.update(status='cancelled');return
                if self.implementation_pins()!=self.pins or sha(d/'request.json')!=j['request_sha256']:raise RuntimeError('SOURCE_OR_REQUEST_CHANGED')
                j.update(status='running',phase='initializing',worker_attempts=1,completion_unconfirmed=True);self._event(j,'worker','Yeni fizik oturumu başlatılıyor.');self._save(j);write_new(d/'worker_attempt.json',{'at':now(),'completion_unconfirmed':True,'source_pins':self.pins})
            env={k:v for k,v in os.environ.items() if k.upper() not in ('PYTHONPATH','PYTHONHOME')}
            with (d/'worker.log').open('xb') as log:
                process=self.popen([str(self.python),'-I','-B',str(self.worker),'--request',str(d/'request.json'),'--output',str(execution)],cwd=str(self.root),env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.processes[identity]=process;start=time.monotonic();cancel_at=None
                while process.poll() is None:
                    with self.lock:self._progress(self.jobs[identity],execution)
                    if (d/'cancel.request').exists():cancel_at=cancel_at or time.monotonic()
                    if time.monotonic()-start>2400 or cancel_at is not None and time.monotonic()-cancel_at>8:
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));process.wait(timeout=10);break
                    time.sleep(.25)
            with self.lock:
                j=self.jobs[identity];self._progress(j,execution)
                if self.implementation_pins()!=self.pins or sha(d/'request.json')!=j['request_sha256']:raise RuntimeError('SOURCE_OR_REQUEST_CHANGED')
                result=read_json(execution/'result.json') if (execution/'result.json').is_file() else None
                if result is not None and type(result) is dict and result.get('status') in TERMINAL and type(result.get('task_verified_actual')) is bool and (process.poll()==0 or result['status'] in ('failed','cancelled')):
                    counts=result.get('counts',{});unknown=counts.get('physical_completion_unconfirmed') is True or bool(counts.get('model_completion_unconfirmed',0))
                    j.update(status=result['status'],completion_unconfirmed=unknown,process_termination_confirmed=True,task_verified_actual=result.get('task_verified_actual',False))
                    if result['status']=='failed':j['error']='ARTIFACT_EXPORT_INCOMPLETE' if result.get('artifact_export_incomplete') is True else 'MEDIA_INCOMPLETE' if result.get('media_incomplete') is True else 'BODY_JOB_FAILED'
                else:j.update(status='cancelled' if (d/'cancel.request').exists() else 'failed',error=None if (d/'cancel.request').exists() else 'BODY_JOB_STOPPED')
        except Exception:
            with self.lock:self.jobs[identity].update(status='cancelled' if (d/'cancel.request').exists() else 'failed',error=None if (d/'cancel.request').exists() else 'BODY_JOB_STOPPED')
        finally:
            cleanup_confirmed=True
            if process is not None and process.poll() is None:
                try:
                    if not (d/'cancel.request').exists():write_new(d/'cancel.request',{'requested_at':now(),'reason':'manager_stopped'})
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));process.wait(timeout=10)
                except Exception:cleanup_confirmed=False
            with self.lock:
                j=self.jobs[identity];j['finalized']=cleanup_confirmed
                if not cleanup_confirmed:j['completion_unconfirmed']=True
                self._event(j,'terminal' if cleanup_confirmed else 'cleanup_unconfirmed','Beden oturumu sona erdi; gerçek ve sanal denemeler ayrı kaydedildi.' if cleanup_confirmed else 'Alt işlemin sonlandığı doğrulanamadı; yeni işler engellendi.');self._save(j)
            if cleanup_confirmed:self.processes.pop(identity,None)
    def media_path(self,identity,name):
        if re.fullmatch('body_[A-Za-z0-9_]{1,80}',identity) is None or name not in ('actual.jpg','preview.jpg','episode.mp4'):raise KeyError('Unknown media')
        j=self.get(identity,private=True);entry=j.get('media_integrity',{}).get(name)
        if not entry:raise ValueError('Media not verified')
        relative=entry.get('path',name)
        if not media_relative(name,relative):raise ValueError('Recorded media path changed')
        p=ordinary(self.jobs_dir/identity/'execution'/relative)
        if p.stat().st_size>(64*1024**2 if name.endswith('.mp4') else 2*1024**2):raise ValueError('Bounded media required')
        if sha(p)!=entry['sha256']:raise ValueError('Media changed')
        return p
