"""Local dance facade; prior expression histories remain read-only."""
import argparse,json,re,secrets,threading
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit
from ..expression.server_v1 import Handler as BaseHandler
from ..expression.server_v3 import decode_import,public_music,MAX_JSON
from .assets_v1 import ROOT_CACHE,PUBLIC_PYTHON,ordinary,sha
from .manager_v1 import DanceManager
from .music_v1 import DanceMusic

WEB=Path(__file__).with_name('web')
SHOWCASE='artifacts/media/motor/dance_v2/simulation.mp4'
SHOWCASE_SHA='3cb5612af7d7c3267e9ddb3c2643fb7a209025ac116a283249d2e4a5e1d02801'
class Facade:
    def __init__(self,root,dance,legacy,music,*,preview_only=False):
        self.root=Path(root).resolve();self.dance=dance;self.legacy=legacy;self.music=music;self.preview_only=preview_only;self.lock=threading.RLock()
    def active(self):return self.dance.active_id() if self.dance and self.dance.active_id() else self.legacy.active_id()
    def status(self):
        old=self.legacy.status();d=self.dance.status() if self.dance else {'ready':False,'runtime':{'available':False,'state':'preview_only'}}
        return {**old,'ready':bool(d.get('ready')) and not self.preview_only,'active_job':self.active(),
            'music':[public_music(x) for x in self.music.catalog()],
            'dance':{'available':bool(d.get('ready')) and not self.preview_only,'state':'preview_only' if self.preview_only else d.get('runtime',{}).get('state','configured'),'duration_s':4},
            'dance_runtime':d,'legacy_integrations':old.get('integrations'),
            'dance_integrations':d.get('integrations',{'state':'not_configured','scope':'preview_only'}),
            'showcase':{'url':'/media/showcase/dance.mp4','title':'Kayıtlı dans örneği','kind':'recorded_physics','generated_for_request':False}}
    def create(self,value):
        with self.lock:
            if self.preview_only or self.dance is None:raise ValueError('Dance execution unavailable')
            if self.active():raise ValueError('Another job is active')
            return self.dance.create(value)
    def import_music(self,data,label,start):
        with self.lock:
            if self.preview_only:raise ValueError('Preview is read-only')
            if self.active():raise ValueError('Another job is active')
            return self.music.import_wav(data,label,start)
class Handler(BaseHandler):
    def do_GET(self):
        if not self._guard():return self._json(403,{'error':'LOOPBACK_ORIGIN_REQUIRED'})
        f=self.server.facade;p=urlsplit(self.path).path
        try:
            if p in ('/','/index.html','/app.js','/style.css'):
                n='index.html' if p=='/' else p[1:];return self._file(WEB/n,{'index.html':'text/html; charset=utf-8','app.js':'application/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[n])
            if p=='/api/status':return self._json(200,{**f.status(),'csrf_token':self.server.csrf})
            if p=='/media/showcase/dance.mp4':
                path=ordinary(f.root/SHOWCASE)
                if sha(path)!=SHOWCASE_SHA:raise ValueError('Showcase changed')
                return self._file(path,'video/mp4')
            if p=='/api/dance/jobs':return self._json(200,{'jobs':f.dance.list_jobs() if f.dance else []})
            m=re.fullmatch(r'/api/dance/jobs/(dance_[A-Za-z0-9_]{1,80})(/export)?',p)
            if m:
                if m[2]:return self._file(f.dance.media_path(m[1],'export'),'application/zip',True)
                return self._json(200,f.dance.get(m[1]))
            m=re.fullmatch(r'/media/dance/jobs/(dance_[A-Za-z0-9_]{1,80})/(selected|comparison)\.mp4',p)
            if m:return self._file(f.dance.media_path(m[1],m[2]),'video/mp4')
            m=re.fullmatch(r'/media/music/(phrase_(?:original_pulse_v1|upload_[a-f0-9]{24}))\.wav',p)
            if m:return self._file(f.music.get(m[1])['path'],'audio/wav')
        except (ValueError,KeyError,OSError,AttributeError):return self._json(409,{'error':'LOCAL_MEDIA_OR_JOB_UNAVAILABLE'})
        # Frozen legacy GET routes only; no legacy mutation delegation.
        return super().do_GET()
    def do_POST(self):
        if not self._guard(True):return self._json(403,{'error':'LOCAL_SESSION_REQUIRED'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self._json(415,{'error':'JSON_REQUIRED'})
        p=urlsplit(self.path).path;f=self.server.facade
        try:
            self.connection.settimeout(15);length=int(self.headers.get('Content-Length','0'));limit=MAX_JSON if p=='/api/music/import' else 64*1024
            if not 2<=length<=limit:return self._json(413,{'error':'PAYLOAD_TOO_LARGE'})
            raw=self.rfile.read(length)
            if len(raw)!=length:raise ValueError('Incomplete body')
            if p=='/api/music/import':
                data,label,start=decode_import(raw)
                with f.lock:
                    if f.active():return self._json(409,{'error':'JOB_ACTIVE_IMPORT_BLOCKED'})
                    return self._json(201,{'music':public_music(f.import_music(data,label,start))})
            def pairs(items):
                d={}
                for k,v in items:
                    if k in d:raise ValueError('Duplicate member')
                    d[k]=v
                return d
            def bad(_):raise ValueError('Nonfinite JSON')
            value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
            if p=='/api/dance/jobs':return self._json(202,{'id':f.create(value)})
            m=re.fullmatch(r'/api/dance/jobs/(dance_[A-Za-z0-9_]{1,80})/cancel',p)
            if m and value=={}:f.dance.cancel(m[1]);return self._json(202,{'cancel_requested':True})
            return self._json(404,{'error':'READ_ONLY_OR_UNKNOWN_ROUTE'})
        except (ValueError,KeyError,TypeError,UnicodeError):return self._json(400,{'error':'INVALID_OR_UNAVAILABLE_REQUEST'})
        except Exception:return self._json(503,{'error':'LOCAL_DANCE_UNAVAILABLE'})
class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,port,facade):
        super().__init__(('127.0.0.1',port),Handler);self.facade=facade;self.manager=facade.legacy;self.csrf=secrets.token_urlsafe(32)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8774);parser.add_argument('--trusted-local',action='store_true');parser.add_argument('--preview-only',action='store_true');args=parser.parse_args()
    if not args.trusted_local or not 1024<=args.port<=65535:raise ValueError('Explicit local launch required')
    root=Path(__file__).resolve().parents[4]
    # Constructors are reached only on explicit launcher invocation, never import.
    from ..app.phrase.music import PhraseMusic
    from ..app.providers import CodexLocalProvider
    from ..expression.runtime_v1 import ExpressionRuntime
    from ..expression.runner_v3 import ExpressionManager,JOB_ROOT as LEGACY_JOBS
    music=PhraseMusic(root);provider=CodexLocalProvider(trusted_local=True);legacy=ExpressionManager(root,ExpressionRuntime(root,PUBLIC_PYTHON,LEGACY_JOBS),provider,music)
    dance=None
    if not args.preview_only:
        from .runtime_v1 import DanceRuntime
        from .render_v1 import render_dance
        from ..app.integrations import AppIntegrations
        dance=DanceManager(root,DanceRuntime(root),provider,DanceMusic(music),jobs_dir=ROOT_CACHE/'jobs',renderer=render_dance,
            integrations=AppIntegrations(root,root/'artifacts/runs/agentic/dance_studio_v1'))
    server=Server(args.port,Facade(root,dance,legacy,music,preview_only=args.preview_only));print('PulseMorph Dance Studio: http://127.0.0.1:'+str(args.port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:
        if dance:
            for identity in list(dance._cancel):dance.cancel(identity)
    finally:server.server_close()
if __name__=='__main__':main()
