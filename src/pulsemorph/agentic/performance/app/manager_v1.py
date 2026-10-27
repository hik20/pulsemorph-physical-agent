"""Small process supervisor for music-conditioned physical performance jobs."""
import copy,hashlib,importlib,json,os,re,subprocess,threading,time,uuid,traceback
from datetime import datetime,timezone
from pathlib import Path

CACHE=Path.home()/'.cache/pulsemorph-agentic'
JOBS=CACHE/'performance-app-v1/jobs'
PYTHON=CACHE/'envs/app-public-win-cp311-v1/Scripts/python.exe'
TERMINAL={'completed','stopped','failed','cancelled'}
MEDIA={'episode.mp4':('video/mp4','actual_recorded',64*1024**2),'music.wav':('audio/wav','planning_music',4*1024**2),'actual.jpg':('image/jpeg','actual',2*1024**2),'preview.jpg':('image/jpeg','hypothetical',2*1024**2)}
class Busy(ValueError):pass
def now():return datetime.now(timezone.utc).isoformat()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ordinary(p):
    p=Path(p).absolute()
    for x in (p,*p.parents):
        if x.is_symlink() or x.exists() and getattr(x.lstat(),'st_file_attributes',0)&1024:raise ValueError('Linked private path forbidden')
    return p
def read(p,limit=512*1024):
    p=ordinary(p)
    if p.stat().st_size>limit:raise ValueError('Bounded metadata required')
    value=json.loads(p.read_bytes());encoded(value);return value
def write_new(p,v):
    with Path(p).open('xb') as f:f.write(encoded(v))
def parse_request(v):
    if type(v) is not dict or set(v)!={'instruction'} or type(v['instruction']) is not str or len(v['instruction'])>1500:raise ValueError('Exact bounded instruction required')
    v['instruction'].encode('utf8');return {'instruction':v['instruction'].strip()}
def public_progress(v):
    if type(v) is not dict:raise ValueError('Progress object required')
    keys=('status','phase_label','events','counts','actual','preview','model_decision','plan','qualification')
    result={k:copy.deepcopy(v[k]) for k in keys if k in v}
    if len(encoded(result))>256*1024:raise ValueError('Progress too large')
    def check(x):
        if type(x) is dict:
            for key,item in x.items():
                if type(key) is not str or key.lower() in ('path','directory','source_pins') or re.search('(?i)(_path$|token|secret|password|credential|authorization)',key):raise ValueError('Private progress member')
                check(item)
        elif type(x) is list:
            if len(x)>1000:raise ValueError('Progress list too large')
            for item in x:check(item)
        elif type(x) is str and (len(x)>4000 or re.search(r'[A-Za-z]:[\\/]|\\\\[^ ]+',x)):raise ValueError('Private progress text')
    check(result);return result

class PerformanceManager:
    def __init__(self,root,*,jobs_dir=JOBS,worker=None,profile_status=None,popen=subprocess.Popen):
        self.root=Path(root).resolve();self.directory=ordinary(jobs_dir).resolve();self.worker=ordinary(worker or Path(__file__).with_name('worker_v1.py'));self.popen=popen;self.profile_status=profile_status
        if self.directory==CACHE.resolve() or not self.directory.is_relative_to(CACHE.resolve()):raise ValueError('Dedicated C job directory required')
        self.directory.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.jobs={};self.processes={};self.threads={};self.history_warnings=[];self.pins=self.code_pins();self._load()
    def code_pins(self):
        paths=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.parent.glob('*.py'))+list(Path(__file__).with_name('web').glob('*'))+[PYTHON,self.worker]
        return {str(p.resolve()):sha(p) for p in paths if p.is_file()}
    def _save(self,j):
        j['updated_at']=now();p=self.directory/j['id']/'state.json';tmp=p.with_suffix('.tmp');tmp.write_bytes(encoded(j));tmp.replace(p)
    def _load(self):
        for p in sorted(self.directory.glob('*/state.json'))[-100:]:
            try:
                j=read(p);req=parse_request(read(p.with_name('request.json')))
                if j.get('schema')!='pulsemorph.performance_app_job.v1' or j.get('id')!=p.parent.name or j.get('instruction')!=req['instruction'] or type(j.get('events')) is not list or type(j.get('created_at')) is not str:raise ValueError('History assignment')
                if j.get('status') not in TERMINAL or not j.get('finalized'):j.update(status='failed',error='PROCESS_INTERRUPTED',finalized=not bool(j.get('worker_attempts')),completion_unconfirmed=True)
                self.jobs[j['id']]=j
            except Exception:self.history_warnings.append('invalid_or_partial_performance_history_preserved')
    def active_id(self):
        with self.lock:return next((j['id'] for j in self.jobs.values() if j['status'] not in TERMINAL or not j['finalized']),None)
    def get(self,identity,*,private=False):
        with self.lock:j=copy.deepcopy(self.jobs[identity])
        if private:return j
        keys=('id','mode','instruction','status','finalized','phase_label','created_at','updated_at','events','counts','actual','preview','model_decision','plan','video_url','music_url','actual_image_url','preview_image_url','error','worker_attempts','process_termination_confirmed','completion_unconfirmed','performance_screen_passed')
        return {k:j[k] for k in keys if k in j}
    def list_jobs(self):
        with self.lock:ids=[j['id'] for j in sorted(self.jobs.values(),key=lambda j:j['created_at'],reverse=True)[:100]]
        return [self.get(i) for i in ids]
    def status(self):
        try:profile=self.profile_status() if self.profile_status else importlib.import_module('pulsemorph.agentic.performance.app.worker_v1').profile_status(self.root)
        except Exception:profile={'available':False,'state':'prepared_profile_unavailable'}
        completed=[j for j in self.list_jobs() if j['status']=='completed' and j.get('video_url')]
        latest=completed[0] if completed else None
        return {'ready':profile.get('available') is True and self.code_pins()==self.pins and self.worker.is_file() and PYTHON.is_file(),'prepared_host':True,'profile':{k:profile[k] for k in ('available','state','scope') if k in profile},'active_job':self.active_id(),'history_warnings':list(self.history_warnings),'example':{'url':latest['video_url'],'kind':'recorded_physics','generated_for_request':False,'music_included':True,'title':'Kayıtlı müzikli performans'} if latest else None,'music':{'url':latest['music_url'],'label':'Özgün groove · 100 BPM','duration_s':16.8,'bpm':100} if latest and latest.get('music_url') else None,'scope':'Prepared-host experimental choreography; physical/movement screens are not aesthetic or general dance certification.'}
    def create(self,v):
        req=parse_request(v)
        with self.lock:
            if self.active_id():raise Busy('One performance job already active')
            if not self.status()['ready']:raise ValueError('Prepared profile unavailable')
            identity='performance_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8];d=self.directory/identity;d.mkdir(exist_ok=False);write_new(d/'request.json',req)
            j={'schema':'pulsemorph.performance_app_job.v1','mode':'music_performance','id':identity,**req,'created_at':now(),'status':'queued','phase_label':'Sırada','finalized':False,'events':[],'worker_attempts':0,'completion_unconfirmed':False,'error':None,'video_url':None,'music_url':None,'actual_image_url':None,'preview_image_url':None,'source_pins':self.pins,'request_sha256':sha(d/'request.json')};self.jobs[identity]=j;self._save(j)
            thread=threading.Thread(target=self._run,args=(identity,),daemon=True);self.threads[identity]=thread;thread.start();return identity
    def cancel(self,identity):
        with self.lock:
            j=self.jobs[identity];p=self.directory/identity/'cancel.request'
            if (j['status'] not in TERMINAL or not j['finalized']) and not p.exists():write_new(p,{'at':now()});j['events'].append({'at':now(),'kind':'cancel_requested','message':'İptal istendi; kısmi kanıt korunuyor.'});self._save(j)
    def _progress(self,j,execution):
        if not (execution/'progress.json').is_file():return
        raw=read(execution/'progress.json');p=public_progress(raw)
        for key in ('phase_label','counts','actual','preview','plan','model_decision'):
            if key in p:j[key]=p[key]
        for e in p.get('events',[]):
            if type(e) is not dict or type(e.get('index')) is not int or type(e.get('message')) is not str or type(e.get('kind')) is not str:raise ValueError('Typed worker event required')
            if e['index']>j.get('last_worker_event',-1):j['events'].append({**e,'source':'worker'});j['last_worker_event']=e['index']
        for name,descriptor in raw.get('media',{}).items():
            if name not in MEDIA or type(descriptor) is not dict:continue
            mime,role,limit=MEDIA[name]
            if descriptor.get('path')!=name or descriptor.get('role')!=role or type(descriptor.get('sha256')) is not str:raise ValueError('Exact POSIX canonical media descriptor required')
            path=ordinary(execution/name)
            if not path.is_file() or path.stat().st_size>limit or sha(path)!=descriptor['sha256']:raise ValueError('Published media bytes differ')
            signature=path.open('rb')
            with signature as stream:head=stream.read(12)
            if name.endswith('.jpg') and head[:2]!=b'\xff\xd8' or name.endswith('.wav') and (head[:4]!=b'RIFF' or head[8:12]!=b'WAVE') or name.endswith('.mp4') and head[4:8]!=b'ftyp':raise ValueError('Media MIME signature differs')
            j.setdefault('media',{})[name]={'sha256':descriptor['sha256'],'role':role}
            key={'episode.mp4':'video_url','music.wav':'music_url','actual.jpg':'actual_image_url','preview.jpg':'preview_image_url'}[name];j[key]='/media/jobs/'+j['id']+'/'+name+'?v='+descriptor['sha256'][:16]
        self._save(j)
    def _kill(self,process):
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));process.wait(timeout=10)
    def _run(self,identity):
        d=self.directory/identity;execution=d/'execution';process=None;stage='preflight'
        try:
            with self.lock:
                j=self.jobs[identity]
                if (d/'cancel.request').exists():j['status']='cancelled';return
                if self.code_pins()!=self.pins or sha(d/'request.json')!=j['request_sha256']:raise RuntimeError('Source/request changed')
                j.update(status='running',worker_attempts=1,completion_unconfirmed=True);self._save(j);write_new(d/'worker_attempt.json',{'at':now(),'completion_unconfirmed':True,'source_pins':self.pins})
            env={k:v for k,v in os.environ.items() if k.upper() not in ('PYTHONPATH','PYTHONHOME')}
            stage='worker_start'
            with (d/'worker.log').open('xb') as log:
                process=self.popen([str(PYTHON),'-I','-B',str(self.worker),'--request',str(d/'request.json'),'--output',str(execution)],cwd=str(self.root),env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.processes[identity]=process;started=time.monotonic();cancel_at=None
                while process.poll() is None:
                    stage='progress_or_media'
                    with self.lock:self._progress(self.jobs[identity],execution)
                    if (d/'cancel.request').exists():cancel_at=cancel_at or time.monotonic()
                    if time.monotonic()-started>2400 or cancel_at is not None and time.monotonic()-cancel_at>8:self._kill(process);break
                    time.sleep(.25)
            with self.lock:
                stage='terminal_progress_or_receipt'
                j=self.jobs[identity];self._progress(j,execution)
                if self.code_pins()!=self.pins or sha(d/'request.json')!=j['request_sha256']:raise RuntimeError('Source/request changed')
                r=read(execution/'result.json') if (execution/'result.json').is_file() else None
                if type(r) is dict and r.get('status') in TERMINAL and type(r.get('performance_screen_passed')) is bool and (process.poll()==0 or r['status'] in ('failed','cancelled')):
                    counts=r.get('counts',{});j.update(status=r['status'],performance_screen_passed=r['performance_screen_passed'],process_termination_confirmed=True,completion_unconfirmed=counts.get('physical_completion_unconfirmed') is True or bool(counts.get('model_completion_unconfirmed',0)))
                    if r['status']=='failed':j['error']='ARTIFACT_EXPORT_INCOMPLETE' if r.get('artifact_export_incomplete') is True else 'MEDIA_INCOMPLETE' if r.get('media_incomplete') is True else 'PERFORMANCE_JOB_FAILED'
                else:j.update(status='cancelled' if (d/'cancel.request').exists() else 'failed',error=None if (d/'cancel.request').exists() else 'PERFORMANCE_JOB_STOPPED')
        except Exception as error:
            failure=d/'supervisor_failure.json'
            if not failure.exists():write_new(failure,{'schema':'pulsemorph.performance_supervisor_failure.v1','stage':stage,'error_type':type(error).__name__,'stack_locations':[{'file':f.filename,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(error.__traceback__)],'qualification':'Private stack locations only; exception text, source lines and locals omitted.'})
            with self.lock:self.jobs[identity].update(status='cancelled' if (d/'cancel.request').exists() else 'failed',error=None if (d/'cancel.request').exists() else 'PERFORMANCE_JOB_STOPPED')
        finally:
            ended=True
            if process is not None and process.poll() is None:
                try:
                    if not (d/'cancel.request').exists():write_new(d/'cancel.request',{'at':now(),'reason':'supervisor_stopped'})
                    self._kill(process)
                except Exception:ended=False
            with self.lock:
                j=self.jobs[identity];j['finalized']=ended
                if not ended:j['completion_unconfirmed']=True
                self._save(j)
            if ended:self.processes.pop(identity,None)
    def media_path(self,identity,name):
        if re.fullmatch('performance_[A-Za-z0-9_]{1,80}',identity) is None or name not in MEDIA:raise KeyError('Unknown media')
        j=self.get(identity,private=True);entry=j.get('media',{}).get(name);path=ordinary(self.directory/identity/'execution'/name)
        if not entry or path.stat().st_size>MEDIA[name][2] or sha(path)!=entry['sha256']:raise ValueError('Media not verified or changed')
        return path
