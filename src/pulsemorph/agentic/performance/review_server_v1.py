"""Read-only loopback viewer for one hash-pinned HYPOTHETICAL candidate.

Standard library only. No model/controller imports, job route or POST mutation.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

CACHE=Path.home()/'.cache/pulsemorph-agentic'
VIDEO=CACHE/'renders/llm-music-01-preview-000/performance.mp4'
VIDEO_SHA='d9ef4e5f3fef24d2dcd5268a375188c013328eaea6a24e4255393debaa155433'
METADATA=VIDEO.parent/'review_metadata.json'
METADATA_SHA='0053788c728e883b09b745c448fe23143521e5c46d28966ce55da977782edb49'
DESCRIPTOR=CACHE/'performance-v1/llm_music_01/audio/descriptor.json'
AUDIO_SHA='df25a0b55dbe6b7368995e634225239fcadfe969f9b2b2d841e4ec8ae8ea0d02'
WEB=Path(__file__).with_name('review_web')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def ordinary(path):
    path=Path(path).absolute()
    if any(p.is_symlink() or(p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024) for p in (path,*path.parents)):raise ValueError('Linked viewer source forbidden')
    return path
def byte_range(header,size):
    if header is None:return 0,size-1,200
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',header)
    if not match or not any(match.groups()):raise ValueError('Invalid range')
    first,last=match.groups()
    if not first:
        length=int(last)
        if length<=0:raise ValueError('Invalid suffix')
        start,end=max(0,size-length),size-1
    else:start,end=int(first),min(size-1,int(last)) if last else size-1
    if start>=size or end<start:raise ValueError('Unsatisfied range')
    return start,end,206

class ReviewData:
    def __init__(self):
        if sha(ordinary(VIDEO))!=VIDEO_SHA or sha(ordinary(METADATA))!=METADATA_SHA:raise ValueError('Pinned video/review evidence changed')
        descriptor=json.loads(ordinary(DESCRIPTOR).read_bytes());audio=ordinary(descriptor['path'])
        if not audio.resolve().is_relative_to(DESCRIPTOR.parent.resolve()) or descriptor['sha256']!=AUDIO_SHA or sha(audio)!=AUDIO_SHA:raise ValueError('Pinned owned score changed')
        metadata=json.loads(METADATA.read_bytes())
        if metadata['video_sha256']!=VIDEO_SHA:raise ValueError('Review/video assignment differs')
        self.audio=audio
        self.routes={'/media/candidate.mp4':(VIDEO,'video/mp4',VIDEO_SHA),'/media/music.wav':(audio,'audio/wav',AUDIO_SHA)}
        self.public={'kind':'hypothetical_recorded_physics','actual_parent_committed':False,
            'title':'Sanal aday · uygulanmadı · sürüklenme var','video_url':'/media/candidate.mp4','music_url':'/media/music.wav',
            'duration_s':16.8,'bpm':100,'model_calls_in_run':4,'physical_previews_in_run':3,'actual_commits_in_run':0,
            'root_translation_end_minus_start_m':metadata['root_translation_end_minus_start_m'],
            'root_axis_peak_to_peak_m':metadata['root_axis_peak_to_peak_m'],
            'joint_spans_rad':{k:v['max_span_rad'] for k,v in metadata['actual_joint_spans_during_music'].items()},
            'qualification':'Gerçek fizik sanal kopyada yürütüldü. Ajan ana bedene uygulamayı seçmedi. Bu ekran yeni üretim başlatmaz.'}

class Handler(BaseHTTPRequestHandler):
    server_version='PulseMorphRecordedReview/1'
    def log_message(self,*_args):pass
    def guard(self):
        hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        return self.client_address[0]=='127.0.0.1' and self.headers.get('Host') in hosts and (not self.headers.get('Origin') or self.headers['Origin'] in {'http://'+h for h in hosts})
    def headers_safe(self):
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'none'; media-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
    def json(self,status,value,head=False):
        raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf8');self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.headers_safe();self.end_headers()
        if not head:self.wfile.write(raw)
    def serve(self,head=False):
        if not self.guard():return self.json(403,{'error':'LOCAL_REVIEW_ONLY'},head)
        path=urlsplit(self.path).path
        try:
            if path=='/api/review':return self.json(200,self.server.data.public,head)
            if path in ('/','/index.html','/app.js','/style.css'):
                name='index.html' if path=='/' else path[1:];file=ordinary(WEB/name);mime={'index.html':'text/html; charset=utf-8','app.js':'application/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name]
            elif path in self.server.data.routes:
                file,mime,digest=self.server.data.routes[path]
                if sha(ordinary(file))!=digest:return self.json(409,{'error':'RECORDED_BYTES_CHANGED'},head)
            else:return self.json(404,{'error':'REVIEW_FILE_NOT_FOUND'},head)
            size=file.stat().st_size
            try:start,end,status=byte_range(self.headers.get('Range'),size)
            except ValueError:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.headers_safe();self.end_headers();return
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.headers_safe();self.end_headers()
            if not head:
                with file.open('rb') as stream:
                    stream.seek(start);remaining=end-start+1
                    while remaining:
                        block=stream.read(min(65536,remaining))
                        if not block:break
                        self.wfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):return
        except (OSError,ValueError,KeyError):return self.json(503,{'error':'RECORDED_REVIEW_UNAVAILABLE'},head)
    def do_GET(self):self.serve()
    def do_HEAD(self):self.serve(True)
    def do_POST(self):
        self.close_connection=True;self.json(405,{'error':'READ_ONLY_NO_JOB_EXECUTION'})

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,port=8779):
        data=ReviewData();super().__init__(('127.0.0.1',port),Handler);self.data=data

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8779);args=parser.parse_args()
    if not 1024<=args.port<=65535:raise ValueError('Local unprivileged port required')
    server=Server(args.port)
    try:server.serve_forever()
    finally:server.server_close()
