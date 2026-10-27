"""V3 upload fixtures: actual PCM processing, no model/controller work."""
import base64,copy,io,json,sys,tempfile,threading,unittest,wave
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.expression.runner_v3 import parse_request,ExpressionManager,MusicImportBusy
from pulsemorph.agentic.expression.server_v3 import Server,decode_import
from pulsemorph.agentic.app.phrase.music import PhraseMusic

def wav():
    out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setparams((1,2,8000,72000,'NONE','not compressed'))
        w.writeframes((np.sin(np.arange(72000)*.12)*4000).astype('<i2').tobytes())
    return out.getvalue()

def payload(data=None):
    return {'file_b64':base64.b64encode(data or wav()).decode(),'label':'Local fixture','start_s':.5}

class ParseTests(unittest.TestCase):
    def test_music_id_exact_shape(self):
        req={'instruction':'fixture intent','music_id':'phrase_upload_'+'a'*24,'goals':{'schema':'pulsemorph.expression_goals.v1','mean_pose_ratio':1.2,'temporal_ratio':1.2}}
        self.assertEqual(parse_request(req)['music_id'],req['music_id'])
        for bad in ('../audio','phrase_upload_'+'a'*25,'phrase_upload_'+'A'*24,None):
            with self.assertRaises(ValueError):parse_request({**req,'music_id':bad})
    def test_strict_json_base64_and_crop(self):
        for raw in (b'{"a":1,"a":2}',b'{"start_s":NaN}',json.dumps({**payload(),'start_s':True}).encode(),json.dumps({**payload(),'file_b64':'%%%'}).encode(),json.dumps({**payload(),'start_s':10**1000}).encode()):
            with self.assertRaises((ValueError,OverflowError)):decode_import(raw)
    def test_label_boundary_and_literal_music_extension(self):
        self.assertEqual(decode_import(json.dumps({**payload(),'label':'a'*80}).encode())[1],'a'*80)
        for label in ('a'*81,'path/name','bad\\name'):
            with self.assertRaises(ValueError):decode_import(json.dumps({**payload(),'label':label}).encode())

    def test_real_pcm_crop_identity_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            music=PhraseMusic(ROOT,Path(d));data,label,start=decode_import(json.dumps(payload()).encode())
            record=music.import_wav(data,label,start)
            self.assertTrue(record['id'].startswith('phrase_upload_'))
            self.assertEqual((record['sample_rate'],record['sample_frames'],record['channels']),(48000,384000,1))
            self.assertEqual(record['provenance']['start_frame'],4000)
            self.assertFalse(record['provenance']['source_bytes_retained'])
            self.assertEqual(music.get(record['id'])['sha256'],record['sha256'])
    def test_import_active_exclusion(self):
        manager=ExpressionManager.__new__(ExpressionManager);manager._lock=threading.RLock();manager.active_id=lambda:'active'
        with self.assertRaises(MusicImportBusy):manager.import_music(b'x','fixture',0)

class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.music=PhraseMusic(ROOT,Path(self.temp.name))
        class Manager:
            active=False
            def __init__(self,music):self.music=music
            def import_music(self,data,label,start):
                if self.active:raise MusicImportBusy('private detail')
                return self.music.import_wav(data,label,start)
        self.manager=Manager(self.music);self.server=Server(0,self.manager)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
    def request(self,path,body=None,token=True,extra=None):
        headers={'Content-Type':'application/json',**(extra or {})}
        if token:headers['X-PulseMorph-Token']=self.server.csrf
        req=Request(self.base+path,data=None if body is None else body.encode(),headers=headers)
        try:
            with urlopen(req,timeout=4) as r:return r.status,r.read()
        except HTTPError as r:return r.code,r.read()
    def test_import_and_generic_music_get(self):
        code,raw=self.request('/api/music/import',json.dumps(payload()));self.assertEqual(code,201)
        record=json.loads(raw)['music'];self.assertNotIn('path',record)
        self.assertEqual(self.request(record['url'])[0],200)
        self.assertEqual(self.request(record['url'].replace('.wav','Xwav'))[0],404)
        self.assertEqual(self.request('/media/music/../../runner_v3.py')[0],404)
    def test_guards_busy_and_safe_errors(self):
        body=json.dumps(payload())
        self.assertEqual(self.request('/api/music/import',body,False)[0],403)
        self.assertEqual(self.request('/api/music/import',body,extra={'Origin':'https://invalid.example'})[0],403)
        self.manager.active=True
        code,raw=self.request('/api/music/import',body);self.assertEqual(code,409);self.assertNotIn(b'private detail',raw)
        self.manager.active=False
        for bad in ('{"a":1,"a":2}',json.dumps(payload(b'not wav'))):self.assertEqual(self.request('/api/music/import',bad)[0],400)

if __name__=='__main__':unittest.main()
