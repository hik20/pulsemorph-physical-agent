"""Pure music/call-routing fixtures, no application constructors or server launch."""
import io,json,sys,tempfile,unittest,wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.dance.music_v1 import DanceMusic
from pulsemorph.agentic.dance.server_v1 import Facade,Handler
from pulsemorph.agentic.dance.assets_v1 import sha,ROOT_CACHE

class Tests(unittest.TestCase):
    def test_exact_first4_pcm_lineage_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=ROOT_CACHE.parent) as d:
            d=Path(d);source=d/'source.wav';pcm=(np.sin(np.arange(384000)*.11)*5000).astype('<i2')
            with wave.open(str(source),'wb') as w:w.setparams((1,2,48000,384000,'NONE','not compressed'));w.writeframes(pcm.tobytes())
            class Eight:
                def get(self,identity):return {'id':identity,'label':'owned synthetic fixture','path':str(source),'sha256':sha(source)}
            factory=DanceMusic(Eight(),d/'music4');m=factory('fixture')
            with wave.open(m['path'],'rb') as w:self.assertEqual(w.readframes(w.getnframes()),pcm[:192000].tobytes())
            self.assertEqual(m['sample_frames'],192000);self.assertEqual(m['source8s']['selected_sample_start'],0);self.assertEqual(m['source8s']['sha256'],sha(source));self.assertEqual(len(m['sections']),2)
            self.assertEqual(factory('fixture'),m);Path(m['path']).write_bytes(b'tampered')
            with self.assertRaises(ValueError):factory('fixture')
    def test_global_exclusion_and_preview(self):
        class Manager:
            active=None
            def active_id(self):return self.active
            def create(self,v):return 'not executed fixture'
            def status(self):return {'ready':True,'runtime':{'available':True},'provider':{'enabled':True}}
        dance=Manager();legacy=Manager();music=SimpleNamespace(catalog=lambda:[],import_wav=lambda *a:'fixture')
        facade=Facade(ROOT,dance,legacy,music);legacy.active='legacy-finalizing'
        with self.assertRaises(ValueError):facade.create({})
        with self.assertRaises(ValueError):facade.import_music(b'x','name',0)
        legacy.active=None;dance.active='dance-finalizing'
        with self.assertRaises(ValueError):facade.import_music(b'x','name',0)
        dance.active=None;facade.preview_only=True
        with self.assertRaises(ValueError):facade.create({})
        self.assertFalse(facade.status()['dance']['available']);self.assertFalse(facade.status()['showcase']['generated_for_request'])
    def test_legacy_post_never_delegates_and_guard_beforebody(self):
        handler=Handler.__new__(Handler);handler.path='/api/jobs';handler.headers={'Content-Type':'application/json','Content-Length':'2'};handler.rfile=io.BytesIO(b'{}');handler.connection=SimpleNamespace(settimeout=lambda _:None);handler.server=SimpleNamespace(facade=SimpleNamespace())
        replies=[];handler._json=lambda code,value:replies.append((code,value));handler._guard=lambda mutation=False:True
        handler.do_POST();self.assertEqual(replies[-1][0],404)
        handler._guard=lambda mutation=False:False;handler.rfile=SimpleNamespace(read=lambda _:self.fail('must not read'))
        handler.do_POST();self.assertEqual(replies[-1][0],403)
    def test_showcase_pinned_and_statictraversal_notforwardedfile(self):
        handler=Handler.__new__(Handler);handler.path='/media/showcase/dance.mp4';handler._guard=lambda *_:True;handler.server=SimpleNamespace(facade=SimpleNamespace(root=ROOT));files=[];handler._file=lambda p,*a:files.append(p);handler._json=lambda *a:self.fail(str(a))
        # Only read and hash the existing movie; no playback/rendering.
        handler.do_GET();self.assertEqual(files[0],ROOT/'artifacts/media/motor/dance_v2/simulation.mp4')

if __name__=='__main__':unittest.main()
