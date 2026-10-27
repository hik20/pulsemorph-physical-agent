"""Versioned local PCM import; frozen V1/V2 HTTP remains unchanged."""
import argparse
import base64
import binascii
import json
import math
from pathlib import Path
import re
import secrets
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit
from .server_v1 import Handler as BaseHandler
from .runner_v3 import ExpressionManager, JOB_ROOT, MusicImportBusy
from ..app.providers import CodexLocalProvider
from ..app.phrase.assets import PUBLIC_PYTHON
from ..app.phrase.music import PhraseMusic
from ..app.uploads import _label
from .runtime_v1 import ExpressionRuntime

WEB = Path(__file__).with_name('web_v3')
MAX_FILE = 32 * 1024**2
MAX_JSON = 45 * 1024**2

def public_music(record):
    return {**{k:v for k,v in record.items() if k != 'path'},
            'url':'/media/music/' + record['id'] + '.wav'}

def decode_import(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError('Duplicate member')
            result[k]=v
        return result
    def bad(_): raise ValueError('Nonfinite JSON')
    if len(raw)>MAX_JSON: raise ValueError('Payload too large')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
    if type(value) is not dict or set(value)!={'file_b64','label','start_s'}: raise ValueError('Exact fields required')
    if type(value['file_b64']) is not str or len(value['file_b64'])>4*((MAX_FILE+2)//3): raise ValueError('Bounded base64 required')
    if type(value['label']) is not str or not 1<=len(value['label'].strip())<=80: raise ValueError('Bounded label required')
    value['label'].encode('utf8')
    label=_label(value['label'])
    start=value['start_s']
    if type(start) not in (int,float) or not 0<=start<=112 or not math.isfinite(start): raise ValueError('Bounded crop required')
    data=base64.b64decode(value['file_b64'],validate=True)
    if not 1<=len(data)<=MAX_FILE: raise ValueError('Bounded WAV required')
    return data,label,start

class Handler(BaseHandler):
    def do_GET(self):
        if not self._guard(): return self._json(403,{'error':'LOOPBACK_ORIGIN_REQUIRED'})
        path=urlsplit(self.path).path
        try:
            if path in ('/','/index.html','/app.js','/style.css'):
                name='index.html' if path=='/' else path[1:]
                kind={'index.html':'text/html; charset=utf-8','app.js':'application/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name]
                return self._file(WEB/name,kind)
            match=re.fullmatch(r'/media/music/(phrase_(?:original_pulse_v1|upload_[a-f0-9]{24}))\.wav',path)
            if match: return self._file(self.server.manager.music.get(match[1])['path'],'audio/wav')
        except Exception: return self._json(409,{'error':'LOCAL_MUSIC_UNAVAILABLE'})
        return super().do_GET()
    def do_POST(self):
        if urlsplit(self.path).path!='/api/music/import': return super().do_POST()
        if not self._guard(True): return self._json(403,{'error':'LOCAL_SESSION_REQUIRED'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json': return self._json(415,{'error':'JSON_REQUIRED'})
        try:
            self.connection.settimeout(15)
            length=int(self.headers.get('Content-Length','0'))
            if not 2<=length<=MAX_JSON: return self._json(413,{'error':'MUSIC_PAYLOAD_TOO_LARGE'})
            raw=self.rfile.read(length)
            if len(raw)!=length: raise ValueError('Incomplete body')
            data,label,start=decode_import(raw)
            record=self.server.manager.import_music(data,label,start)
            return self._json(201,{'music':public_music(record)})
        except MusicImportBusy: return self._json(409,{'error':'JOB_ACTIVE_IMPORT_BLOCKED'})
        except (ValueError,TypeError,UnicodeError,binascii.Error): return self._json(400,{'error':'INVALID_PCM16_MUSIC_IMPORT'})
        except Exception: return self._json(503,{'error':'LOCAL_MUSIC_IMPORT_UNAVAILABLE'})

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,port,manager):
        super().__init__(('127.0.0.1',port),Handler)
        self.manager,self.csrf=manager,secrets.token_urlsafe(32)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8774)
    parser.add_argument('--trusted-local',action='store_true')
    args=parser.parse_args()
    if not args.trusted_local or not 1024<=args.port<=65535: raise ValueError('Explicit private local launch required')
    root=Path(__file__).resolve().parents[4]
    runtime=ExpressionRuntime(root,PUBLIC_PYTHON,JOB_ROOT)
    manager=ExpressionManager(root,runtime,CodexLocalProvider(trusted_local=True),PhraseMusic(root))
    server=Server(args.port,manager)
    print('PulseMorph Expression Studio v3: http://127.0.0.1:'+str(args.port),flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt:
        for identity in list(manager._cancel): manager.cancel(identity)
    finally: server.server_close()

if __name__=='__main__': main()
