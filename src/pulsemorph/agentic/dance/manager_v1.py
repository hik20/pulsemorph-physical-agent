"""Bounded existing-dance editing; a physical screen never means intent success."""
import copy
import hashlib
import json
import re
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from ..phrase_program import ProgramCompileError
from .planner_v1 import plan_dance,parse_parameters,FIELDS
from .assets_v1 import ROOT_CACHE,baseline_parameters,BASELINE_QUALIFICATION

CACHE=Path.home()/'.cache/pulsemorph-agentic'
JOB_ROOT=ROOT_CACHE/'jobs'
TERMINAL={'completed','failed','unsupported','cancelled'}
MAX_MODELS=MAX_CANDIDATES=3
WALL_SECONDS=1800

def now():return datetime.now(timezone.utc).isoformat()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,v):
    with Path(p).open('xb') as f:f.write(encoded(v))
def safe_path(p,root):
    p,root=Path(p).absolute(),Path(root).resolve()
    if p.resolve()==root or not p.resolve().is_relative_to(root):raise ValueError('Dedicated private dance path required')
    for x in (p,*p.parents):
        if x.is_symlink() or x.exists() and getattr(x.lstat(),'st_file_attributes',0)&1024:raise ValueError('Linked dance path forbidden')
    return p
def parse_request(v):
    if type(v) is not dict or set(v)!={'instruction','music_id'}:raise ValueError('Exact instruction/music request required')
    if type(v['instruction']) is not str or not 3<=len(v['instruction'].strip())<=3000:raise ValueError('Bounded instruction required')
    v['instruction'].encode('utf8')
    if type(v['music_id']) is not str or re.fullmatch('[A-Za-z0-9_]{1,120}',v['music_id']) is None:raise ValueError('Catalog music ID required')
    return {'instruction':v['instruction'].strip(),'music_id':v['music_id']}
def unit_parameters():return {k:0. if k=='timing_shift_s' else 1. for k in FIELDS}
def public_trial(t):
    return None if t is None else {k:copy.deepcopy(t[k]) for k in ('id','index','status','parameters','metrics','reason','rejection_report','baseline_qualification') if k in t}
def public_job(j):
    keys=('mode','id','status','finalized','created_at','updated_at','instruction','music_id','events','plans',
        'selected_id','result','video_url','comparison_url','export_url','error','new_model_calls','model_call_attempts',
        'model_completion_unconfirmed','new_physical_trials','worker_attempts','physical_completion_unconfirmed',
        'reference_compile_attempts','measurement_counts_unconfirmed','integration_status','assessment')
    return {**{k:copy.deepcopy(j.get(k)) for k in keys},'baseline':public_trial(j.get('baseline')),
            'trials':[public_trial(t) for t in j.get('trials',[])]}

class DanceManager:
    def __init__(self,root,runtime,provider,music_factory,*,jobs_dir=JOB_ROOT,renderer,integrations=None):
        self.root=Path(root).resolve();self.runtime=runtime;self.provider=provider;self.music_factory=music_factory
        self.jobs_dir=safe_path(jobs_dir,CACHE);self.jobs_dir.mkdir(parents=True,exist_ok=True)
        self.renderer=renderer;self.integrations=integrations;self._lock=threading.RLock();self._pool=ThreadPoolExecutor(max_workers=1)
        self._jobs={};self._cancel={};self._deadlines={};self._timers={};self.history_warnings=[];self._live=False
        self.pins=self.implementation_pins();self._load_history()
    def implementation_pins(self):
        paths=list(Path(__file__).parent.glob('*.py'))
        paths += [self.root/'src/pulsemorph/agentic/app/providers.py',self.root/'src/pulsemorph/agentic/app/integrations.py',self.root/'experiments/agentic/app/meterless_bridge.ts',self.root/'src/pulsemorph/agentic/expression/planner_v2.py',self.root/'src/pulsemorph/agentic/phrase_planner.py']
        return {**getattr(self.runtime,'_implementation',{}),**{str(p.resolve()):sha(p) for p in paths if p.is_file()}}
    def _load_history(self):
        for p in sorted(self.jobs_dir.glob('*/state.json'))[-100:]:
            try:
                safe_path(p,self.jobs_dir)
                if p.stat().st_size>2*1024**2:raise ValueError('History too large')
                j=json.loads(p.read_bytes());request=parse_request(json.loads(p.with_name('request.json').read_bytes()))
                if j.get('schema')!='pulsemorph.dance_job.v1' or j.get('mode')!='dance' or j.get('id')!=p.parent.name or request!={k:j[k] for k in request}:raise ValueError('History assignment')
                if (type(j.get('events')) is not list or type(j.get('plans')) is not list or len(j['plans'])>3 or type(j.get('trials')) is not list or len(j['trials'])>3 or type(j.get('created_at')) is not str or type(j.get('updated_at')) is not str or j.get('status') not in TERMINAL|{'queued','planning','baseline','trial','feedback','rendering'}):raise ValueError('History shape')
                encoded(j)
                if j['status'] not in TERMINAL:j.update(status='failed',error='PROCESS_INTERRUPTED',finalized=True)
                elif not j.get('finalized'):j.update(finalized=True,completion_note='Interrupted final persistence; partial evidence preserved.')
                self._jobs[j['id']]=j
            except Exception:self.history_warnings.append('invalid_or_partial_dance_history_preserved')
    def active_id(self):
        with self._lock:return next((j['id'] for j in self._jobs.values() if j['status'] not in TERMINAL or not j['finalized']),None)
    def status(self):
        r=self.runtime.status();p=self.provider.status();current=self.implementation_pins()==self.pins
        return {'ready':current and r.get('available') is True and p.get('enabled') is True,'runtime':r,
            'provider':{**p,'authentication_verified':self._live,'authentication_basis':'completed_live_call_this_session' if self._live else 'not_observed'},
            'active_job':self.active_id(),'history_warnings':list(self.history_warnings),'mode':'dance',
            'baseline_profile':{'parameters':baseline_parameters(),'qualification':BASELINE_QUALIFICATION},
            'integrations':self.integrations.status() if self.integrations is not None else {'state':'not_configured','source_id':'dance_reference_edit_v1'},
            'assessment':'Physical tracking/stability screen only; no measured intent/beauty/style success.',
            'budgets':{'model_calls':3,'candidate_proposals':3,'physical_trials_including_baseline':4}}
    def get(self,identity,*,private=False):
        with self._lock:j=copy.deepcopy(self._jobs[identity])
        return j if private else public_job(j)
    def list_jobs(self):
        with self._lock:return [public_job(j) for j in sorted(self._jobs.values(),key=lambda j:j['created_at'],reverse=True)[:100]]
    def _update(self,identity,**changes):
        with self._lock:
            j=self._jobs[identity];j.update(changes,updated_at=now());p=self.jobs_dir/identity/'state.json';tmp=p.with_suffix('.tmp');tmp.write_bytes(encoded(j));tmp.replace(p)
    def _event(self,identity,kind,message):
        with self._lock:
            events=copy.deepcopy(self._jobs[identity]['events']);events.append({'seq':len(events),'at':now(),'kind':kind,'message':message});self._update(identity,events=events)
    def _check(self,identity):
        if self._cancel[identity].is_set():raise RuntimeError('JOB_CANCELLED')
        if time.monotonic()>=self._deadlines[identity]:self._cancel[identity].set();raise RuntimeError('JOB_WALL_LIMIT')
        if self.implementation_pins()!=self.pins:raise RuntimeError('SOURCE_CHANGED')
        self.runtime._unchanged()
    def _music(self,identity):
        m=self.music_factory(identity)
        if type(m) is not dict or m.get('id')!=identity or m.get('duration_s')!=4 or m.get('sample_rate')!=48000 or m.get('sample_frames')!=192000 or type(m.get('path')) is not str or sha(m['path'])!=m.get('sha256'):raise ValueError('Verified four-second music required')
        return copy.deepcopy(m)
    def create(self,value):
        req=parse_request(value)
        with self._lock:
            if self.active_id():raise ValueError('Another dance job active')
            if not self.status()['ready']:raise ValueError('Dance profile/provider unavailable')
            m=self._music(req['music_id']);identity='dance_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8]
            d=safe_path(self.jobs_dir/identity,self.jobs_dir);d.mkdir(exist_ok=False);write_new(d/'request.json',req);write_new(d/'music.json',m)
            j={'schema':'pulsemorph.dance_job.v1','mode':'dance','id':identity,**req,'status':'queued','finalized':False,'created_at':now(),'updated_at':now(),
                'events':[],'plans':[],'baseline':None,'trials':[],'selected_id':None,'result':None,'video_url':None,'comparison_url':None,'export_url':None,'error':None,
                'source_pins':self.pins,'new_model_calls':0,'new_physical_trials':0,'assessment':'No measured full-intent/beauty/style score; physical screen only.'}
            self._jobs[identity]=j;self._cancel[identity]=threading.Event();self._deadlines[identity]=time.monotonic()+WALL_SECONDS
            timer=threading.Timer(WALL_SECONDS,self._cancel[identity].set);timer.daemon=True;self._timers[identity]=timer;timer.start();self._update(identity);self._pool.submit(self._run,identity)
        return identity
    def cancel(self,identity):
        with self._lock:
            if identity not in self._jobs:raise KeyError('Unknown dance job')
            if identity in self._cancel and self._jobs[identity]['status'] not in TERMINAL:self._cancel[identity].set();self._event(identity,'cancel_requested','İptal istendi; kısmi kanıt korunuyor.')
    def _counts(self,identity):
        d=self.jobs_dir/identity;receipts=[json.loads(p.read_bytes()) for p in (d/'model').glob('*/receipt.json')];attempts=list((d/'model').glob('*/attempt.json'));started=list((d/'trials').glob('*/worker_started.json'));results=[json.loads(p.read_bytes()) for p in (d/'trials').glob('*/result.json')];completed=sum(r.get('status')=='completed' and r.get('integrity_passed') is True for r in results)
        self._update(identity,new_model_calls=sum(r.get('live') is True for r in receipts),model_call_attempts=len(attempts),model_completion_unconfirmed=max(0,len(attempts)-len(receipts)),new_physical_trials=completed,worker_attempts=len(started),physical_completion_unconfirmed=max(0,len(started)-completed),reference_compile_attempts=len(list((d/'physical_attempts').glob('*.json'))))
    def _remember(self,identity,trial=None):
        if self.integrations is None:return
        try:
            j=self.get(identity,private=True)
            if trial is None:self.integrations.record_job({'id':identity,'source_id':'dance_reference_edit_v1','status':j['status'],'result':j['result'],'instruction':j['instruction'],'selected_id':j['selected_id']})
            else:self.integrations.record_trial(identity,{**copy.deepcopy(trial),'source_id':'dance_reference_edit_v1','metrics':{**(trial.get('metrics') or {}),'integrity_passed':trial.get('integrity_passed',False),'intent_passed':False},'qualification':'Physical screen only; no full-intent assessment.'})
            self._update(identity,integration_status=self.integrations.status())
        except Exception:self._event(identity,'memory_unavailable','Bellek kaydı kullanılamadı; yerel kanıt korundu.')
    def _physical(self,identity,parameters,index,music):
        self._check(identity);d=self.jobs_dir/identity;name='baseline' if index==-1 else 'candidate_%02d'%index;(d/'physical_attempts').mkdir(exist_ok=True);write_new(d/'physical_attempts'/(name+'.json'),{'parameters':parameters,'index':index,'completion_unconfirmed':True})
        t=self.runtime.run_trial(parameters,d/'trials'/name,identity,index,{'available':True,'path':music['path'],'sha256':music['sha256'],'duration_s':4,'sample_rate':48000,'observed_through_sample':192000},self._cancel[identity])
        self.runtime._verify_trial(t)
        if t.get('index')!=index or t.get('parameters')!=parameters or type(t.get('metrics')) is not dict or type(t['metrics'].get('physical_gate_passed')) is not bool:raise RuntimeError('TRIAL_ASSIGNMENT_INVALID')
        t=copy.deepcopy(t);t['status']='measured';self._counts(identity);return t
    def _run(self,identity):
        try:
            self._check(identity);j=self.get(identity,private=True);m=self._music(j['music_id']);stored=json.loads((self.jobs_dir/identity/'music.json').read_bytes())
            if m!=stored:raise RuntimeError('MUSIC_CHANGED')
            summary={k:m[k] for k in ('schema','id','sha256','duration_s','sample_rate','sample_frames','channels','pcm_bits','rms','peak_amplitude','accent_time_s','accent_measurement','onset_times_s','sections','bpm','bpm_source','limitations','source8s') if k in m};context={'music_summary':summary,'source':{'id':'output_dance','kind':'existing_full_body_reference','baseline_parameters':baseline_parameters(),'baseline_qualification':BASELINE_QUALIFICATION},'constraints':{'duration_s':4,'physical_screen_only':True}};write_new(self.jobs_dir/identity/'context.json',context);feedback=None
            for index in range(3):
                self._check(identity);self._update(identity,status='planning' if index==0 else 'feedback');self._event(identity,'model','Var olan dans için sınırlı referans düzenlemesi planlanıyor.')
                proposal=plan_dance(self.provider,{'instruction':j['instruction']},context,self.jobs_dir/identity/'model'/('call_%02d'%index),cancel_event=self._cancel[identity],feedback=feedback)
                self._counts(identity);self._check(identity)
                if proposal['receipt']['live'] is not True:raise RuntimeError('LIVE_RECEIPT_REQUIRED')
                self._live=True;self._update(identity,plans=self.get(identity)['plans']+[{'index':index,**proposal}])
                if not proposal['locally_supported']:self._update(identity,status='unsupported',result='unsupported');return
                if index==0:
                    self._update(identity,status='baseline');b=self._physical(identity,baseline_parameters(),-1,m);b['baseline_qualification']=BASELINE_QUALIFICATION;self._update(identity,baseline=b);self._remember(identity,b)
                    if not b['metrics']['physical_gate_passed']:raise RuntimeError('BASELINE_PHYSICAL_GATE_FAILED')
                self._update(identity,status='trial')
                try:t=self._physical(identity,proposal['parameters'],index,m)
                except ProgramCompileError as e:t={'id':identity+'__trial_'+str(index),'index':index,'parameters':proposal['parameters'],'status':'rejected','metrics':None,'reason':'reference_compile_rejected','rejection_report':e.report}
                trials=self.get(identity,private=True)['trials']+[t];self._update(identity,trials=trials);self._remember(identity,t)
                if t['status']=='measured' and t['metrics']['physical_gate_passed']:break
                feedback={'schema':'pulsemorph.dance_feedback.v1','baseline':public_trial(self.get(identity,private=True)['baseline']),'trials':[public_trial(t) for t in trials],
                    'candidate_budget_remaining':2-index,'qualification':'Actual compiler/physical feedback. Physical screen is not intent/beauty validation.'}
            self._check(identity);j=self.get(identity,private=True);passing=[t for t in j['trials'] if t['status']=='measured' and t['metrics']['physical_gate_passed']];selected=passing[0] if passing else j['baseline'];result='completed' if passing else 'partial'
            self._update(identity,selected_id=selected['id'],result=result,status='rendering')
            media=self.renderer(self.root,self.runtime,j['baseline'],selected,m,self.jobs_dir/identity/'media',cancel_event=self._cancel[identity]);self._check(identity)
            video=Path(media['comparison']);single=Path(media['selected']);manifest=Path(media['manifest']);meta=json.loads(manifest.read_bytes())
            if meta['baseline_id']!=j['baseline']['id'] or meta['selected_id']!=selected['id'] or meta['comparison_sha256']!=sha(video):raise RuntimeError('MEDIA_ASSIGNMENT_CHANGED')
            if meta.get('selected_sha256')!=sha(single):raise RuntimeError('SELECTED_MEDIA_CHANGED')
            self._export(identity,selected);self._update(identity,status='completed',video_url='/media/dance/jobs/'+identity+'/selected.mp4',comparison_url='/media/dance/jobs/'+identity+'/comparison.mp4',export_url='/api/dance/jobs/'+identity+'/export',media_integrity={'video_sha256':sha(single),'comparison_sha256':sha(video),'manifest_sha256':sha(manifest),'export_sha256':sha(self.jobs_dir/identity/'reference.zip')})
            self._event(identity,'complete','Var olan dans düzenlendi; sayısal fizik kontrolü geçti. Talimatın tamamı veya estetik kalite ölçülmüş sayılmaz.' if passing else 'Adaylar fizik kontrolünü geçmedi; başlangıç korundu.')
        except Exception:
            expired=time.monotonic()>=self._deadlines[identity];cancelled=self._cancel[identity].is_set() and not expired;self._update(identity,status='cancelled' if cancelled else 'failed',error=None if cancelled else 'DANCE_JOB_STOPPED');self._event(identity,'cancelled' if cancelled else 'failed','İş durdu; kısmi kayıtlar korundu.')
        finally:
            self._timers[identity].cancel()
            try:self._counts(identity)
            except Exception:self._update(identity,measurement_counts_unconfirmed=True)
            self._remember(identity);self._update(identity,finalized=True)
    def _export(self,identity,selected):
        self.runtime._verify_trial(selected);d=self.jobs_dir/identity
        with zipfile.ZipFile(d/'reference.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            summary=public_job(self.get(identity,private=True));summary.update(status='completed',video_url='/media/dance/jobs/'+identity+'/selected.mp4',comparison_url='/media/dance/jobs/'+identity+'/comparison.mp4',export_url='/api/dance/jobs/'+identity+'/export');summary['export_scope']='Private local existing-dance reference and physical summary; original host provenance retained; no training/rights/intent-quality admission.';z.writestr('summary.json',encoded(summary))
            for n in ('manifest.json','arrays.npz'):z.write(Path(selected['paths']['common_reference'])/n,'common_reference/'+n)
        with zipfile.ZipFile(d/'reference.zip') as z:
            if z.testzip() is not None:raise RuntimeError('EXPORT_CRC_FAILED')
    def media_path(self,identity,kind):
        if re.fullmatch('dance_[A-Za-z0-9_]{1,80}',identity) is None:raise KeyError('Unknown dance job')
        j=self.get(identity,private=True)
        if j['status']!='completed':raise ValueError('Completed media only')
        self.runtime._unchanged();d=self.jobs_dir/identity;integrity=j['media_integrity']
        if sha(d/'media/manifest.json')!=integrity['manifest_sha256']:raise ValueError('Media changed')
        if kind not in ('video','selected','comparison','export'):raise ValueError('Unknown media kind')
        name,key=('media/selected.mp4','video_sha256') if kind in ('video','selected') else ('media/comparison.mp4','comparison_sha256') if kind=='comparison' else ('reference.zip','export_sha256')
        p=safe_path(d/name,self.jobs_dir)
        if sha(p)!=integrity[key]:raise ValueError('Media changed')
        return p
