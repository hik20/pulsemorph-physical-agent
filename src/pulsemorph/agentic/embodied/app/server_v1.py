"""Loopback-only Body Agent Lab, separate from the older dance service."""
import argparse,json,re,secrets
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit
from ...expression.server_v1 import Handler as GuardedHandler
from .manager_v1 import BodyAgentManager,BodyJobBusy,CACHE,ordinary,sha

WEB=Path(__file__).with_name('web')
EXAMPLE=CACHE/'renders/hybrid-preview-live-v2/render_prepared_v1/episode.mp4'
EXAMPLE_SHA='11b45330bbd7dbe8ab8194cc5ca1f1e3e7ed61d10f6bd2de6837eb65b16531ce'
class Handler(GuardedHandler):
    def do_GET(self):
        if not self._guard():return self._json(403,{'error':'LOOPBACK_ORIGIN_REQUIRED'})
        p=urlsplit(self.path).path;m=self.server.manager
        try:
            if p in ('/','/index.html','/app.js','/style.css'):
                name='index.html' if p=='/' else p[1:];return self._file(WEB/name,{'index.html':'text/html; charset=utf-8','app.js':'application/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name])
            if p=='/api/status':return self._json(200,{**m.status(),'csrf_token':self.server.csrf,'example':{'url':'/media/showcase/episode.mp4','title':'Kayıtlı beden ajanı örneği','kind':'recorded_physics','generated_for_request':False,'duration_s':2.6}})
            if p=='/api/jobs':return self._json(200,{'jobs':m.list_jobs()})
            q=re.fullmatch('/api/jobs/(body_[A-Za-z0-9_]{1,80})',p)
            if q:return self._json(200,m.get(q[1]))
            if p=='/media/showcase/episode.mp4':
                path=ordinary(EXAMPLE)
                if sha(path)!=EXAMPLE_SHA:raise ValueError('Recorded example changed')
                return self._file(path,'video/mp4')
            q=re.fullmatch(r'/media/jobs/(body_[A-Za-z0-9_]{1,80})/(actual\.jpg|preview\.jpg|episode\.mp4)',p)
            if q:return self._file(m.media_path(q[1],q[2]),'video/mp4' if q[2].endswith('.mp4') else 'image/jpeg')
            return self._json(404,{'error':'NOT_FOUND'})
        except (ValueError,KeyError,OSError):return self._json(409,{'error':'LOCAL_RECORD_UNAVAILABLE'})
        except Exception:return self._json(503,{'error':'LOCAL_BODY_STATUS_UNAVAILABLE'})
    def do_POST(self):
        if not self._guard(True):return self._json(403,{'error':'LOCAL_SESSION_REQUIRED'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self._json(415,{'error':'JSON_REQUIRED'})
        try:
            self.connection.settimeout(15);n=int(self.headers.get('Content-Length','0'))
            if not 2<=n<=16*1024:return self._json(413,{'error':'PAYLOAD_TOO_LARGE'})
            raw=self.rfile.read(n)
            if len(raw)!=n:raise ValueError('Incomplete body')
            def pairs(items):
                result={}
                for k,v in items:
                    if k in result:raise ValueError('Duplicate member')
                    result[k]=v
                return result
            def bad(_):raise ValueError('Nonfinite JSON')
            value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad);p=urlsplit(self.path).path
            if p=='/api/jobs':return self._json(202,{'id':self.server.manager.create(value)})
            q=re.fullmatch('/api/jobs/(body_[A-Za-z0-9_]{1,80})/cancel',p)
            if q and value=={}:self.server.manager.cancel(q[1]);return self._json(202,{'cancel_requested':True})
            return self._json(404,{'error':'NOT_FOUND'})
        except BodyJobBusy:return self._json(409,{'error':'BODY_JOB_ACTIVE'})
        except (ValueError,KeyError,TypeError,UnicodeError):return self._json(400,{'error':'INVALID_OR_UNAVAILABLE_BODY_REQUEST'})
        except Exception:return self._json(503,{'error':'LOCAL_BODY_JOB_UNAVAILABLE'})
class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,port,manager):
        super().__init__(('127.0.0.1',port),Handler);self.manager=manager;self.csrf=secrets.token_urlsafe(32)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8777);parser.add_argument('--trusted-local',action='store_true');args=parser.parse_args()
    if not args.trusted_local or not 1024<=args.port<=65535:raise ValueError('Explicit private local launch required')
    root=Path(__file__).resolve().parents[5];manager=BodyAgentManager(root);server=Server(args.port,manager)
    print('PulseMorph Body Agent Lab: http://127.0.0.1:'+str(args.port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:
        for identity in list(manager.processes):manager.cancel(identity)
    finally:server.server_close()
if __name__=='__main__':main()
