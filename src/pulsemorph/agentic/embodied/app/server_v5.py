"""Portable read-only startup; execution remains blocked pending native admission."""
import argparse,json,secrets
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit
from .config_v5 import load_config,metadata_status,probe_environment

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def _guard(self,mutation=False):
        allowed={'127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port)}
        return self.headers.get('Host')in allowed and (not self.headers.get('Origin')or self.headers['Origin']in {'http://'+h for h in allowed}) and (not mutation or secrets.compare_digest(self.headers.get('X-PulseMorph-Token',''),self.server.csrf))
    def _bytes(self,status,data,mime):
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def _json(self,status,value):return self._bytes(status,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8')
    def _file(self,path,mime):
        if path.is_symlink() or not path.is_file()or path.stat().st_size>512*1024:return self._json(404,{'error':'STATIC_UNAVAILABLE'})
        return self._bytes(200,path.read_bytes(),mime)
    def do_GET(self):
        if not self._guard():return self._json(403,{'error':'LOOPBACK_ORIGIN_REQUIRED'})
        p=urlsplit(self.path).path
        if p in ('/','/index.html'):
            html='''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PulseMorph — taşınabilir kurulum</title><style>body{max-width:850px;margin:4rem auto;padding:1rem;background:#10151c;color:#edf2f7;font:18px system-ui;line-height:1.6}pre{white-space:pre-wrap;background:#19232e;padding:1rem}a{color:#6de0e0}</style><h1>Taşınabilir kurulum ve kabul denetimi</h1><p>Bu sayfa hareket üreten stüdyo değildir. Yeni bağımsız başlangıç profili için hazırlık durumunu gösterir. Buradan model çağrısı veya fizik işi başlatılmaz.</p><p>Kaydedilmiş ajan demosu ve araştırma sonuçları ayrı tutulur. Kısa HOLD testi genel beden, dans ya da müziğe uyum başarısı değildir.</p><h2>Hazırlık durumu</h2><pre id="state">Durum okunuyor…</pre><h2>Bir sonraki adım</h2><p><code>design/agentic/BODY_AGENT_PORTABLE_V5.md</code> rehberini izleyin. CLI önce plan üretir; yürütme yalnız incelenmiş planın özetiyle açık <code>--execute --approved-plan-sha256</code> seçeneği verildiğinde başlar.</p><p>Ana hedef: müzik ve doğal dilden tam beden ifadesi. Bu ekran yalnız mühendislik hazırlığıdır.</p><script>fetch('/api/status').then(r=>r.json()).then(x=>{document.querySelector('#state').textContent=JSON.stringify({ready:x.ready,profile:x.profile,scope:x.capability_scope},null,2)}).catch(()=>{document.querySelector('#state').textContent='Yerel hazırlık durumu okunamadı.'})</script></html>'''
            return self._bytes(200,html.encode(),'text/html; charset=utf-8')
        if p in ('/','/index.html','/app.js','/style.css'):
            name='index.html' if p=='/' else p[1:]
            return self._file(Path(__file__).with_name('web')/name,{'index.html':'text/html; charset=utf-8','app.js':'application/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name])
        if p=='/api/status':
            return self._json(200,{'ready':False,'prepared_host':False,'profile':metadata_status(self.server.config),'active_job':None,'history_warnings':[],'scenarios':[{'id':'right_reach','label':'Sağ bileği hedefe götür'},{'id':'both_reach','label':'Önce sağ, sonra iki el'}],'example':None,'csrf_token':self.server.csrf,'capability_scope':'Portable initialization awaiting explicit native physical admission; old prepared-host success is not inherited.'})
        if p=='/api/jobs':return self._json(200,{'jobs':[]})
        return self._json(404,{'error':'NOT_FOUND'})
    def do_POST(self):
        if not self._guard(True):return self._json(403,{'error':'LOCAL_SESSION_REQUIRED'})
        return self._json(503,{'error':'NATIVE_PHYSICAL_ADMISSION_REQUIRED'})

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,config):
        self.config=config;self.csrf=secrets.token_urlsafe(32)
        super().__init__(('127.0.0.1',config['port']),Handler)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--trusted-local',action='store_true');args=parser.parse_args()
    if not args.trusted_local:raise ValueError('Explicit local trust required')
    config=load_config(args.config)
    import sys
    if Path(sys.executable).resolve()!=config['runtime_python']:raise ValueError('Configured interpreter differs')
    probe_environment(config['runtime_python'])
    server=Server(config)
    print('PulseMorph portable V5 read-only UI: http://127.0.0.1:'+str(config['port']),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
