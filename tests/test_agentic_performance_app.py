"""Fresh supervisor/HTTP/media-contract fixtures; no real worker or simulation."""
import io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.performance.app import manager_v1 as m,server_v1 as s

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=m.CACHE);self.addCleanup(self.temp.cleanup);self.d=Path(self.temp.name);self.worker=self.d/'fixture_worker.py';self.worker.write_text('# never executed');self.argv=[]
    def manager(self,result=None,media_path='episode.mp4'):
        def spawn(argv,**kwargs):
            self.argv.append(argv);out=Path(argv[argv.index('--output')+1]);out.mkdir();video=out/'episode.mp4';video.write_bytes(b'\x00\x00\x00\x18ftypisomfixture');music=out/'music.wav';music.write_bytes(b'RIFF\x00\x00\x00\x00WAVEfixture');progress={'phase_label':'Bitti','counts':{'model_calls':0,'actual_steps':0,'sandbox_steps':0},'actual':{'summary':'Synthetic no-physics fixture','report':{'path_length_m':0.}},'preview':{'hypothetical_only':True},'events':[{'index':0,'kind':'model_decision','message':'Sentetik karar'}],'media':{'episode.mp4':{'path':media_path,'sha256':m.sha(video),'role':'actual_recorded'},'music.wav':{'path':'music.wav','sha256':m.sha(music),'role':'planning_music'}}};(out/'progress.json').write_text(json.dumps(progress));(out/'result.json').write_text(json.dumps(result or {'status':'completed','performance_screen_passed':True,'counts':{}}));return SimpleNamespace(pid=999999,poll=lambda:0)
        manager=m.PerformanceManager(ROOT,jobs_dir=self.d/'jobs',worker=self.worker,profile_status=lambda:{'available':True,'state':'fixture'},popen=spawn);manager.code_pins=lambda:{'fixture':'fixed'};manager.pins=manager.code_pins();return manager
    def runjob(self,manager):
        identity=manager.create({'instruction':'Müzikte zıt beden cümleleri kullan'});manager.threads[identity].join(3);return identity,manager.get(identity)
    def test_complete_media_and_provenance_roles_with_exact_cli(self):
        manager=self.manager();identity,job=self.runjob(manager);self.assertEqual(job['status'],'completed');self.assertTrue(job['finalized']);self.assertEqual(job['counts']['model_calls'],0);self.assertIn('-I',self.argv[0]);self.assertIn('-B',self.argv[0]);self.assertIsNotNone(job['video_url']);self.assertIsNotNone(job['music_url']);self.assertIsNone(job.get('preview_image_url'));self.assertFalse(manager.status()['example']['generated_for_request']);self.assertIn('path_length_m',job['actual']['report']);self.assertNotIn('source_pins',job)
        before=len(job['events']);manager._progress(manager.jobs[identity],manager.directory/identity/'execution');self.assertEqual(len(manager.get(identity)['events']),before)
    def test_windows_or_nested_media_path_rejected_without_frozen_consumer_alias(self):
        manager=self.manager(media_path='media\\episode.mp4');identity,job=self.runjob(manager);self.assertEqual(job['status'],'failed');self.assertIsNone(job['video_url']);failure=m.read(manager.directory/identity/'supervisor_failure.json');self.assertEqual(failure['error_type'],'ValueError');self.assertIn('stage',failure)
    def test_failed_media_preserves_numeric_screen_and_unknown_counts(self):
        manager=self.manager(result={'status':'failed','performance_screen_passed':True,'media_incomplete':True,'counts':{'physical_completion_unconfirmed':True}});_,j=self.runjob(manager);self.assertEqual(j['error'],'MEDIA_INCOMPLETE');self.assertTrue(j['performance_screen_passed']);self.assertTrue(j['completion_unconfirmed']);self.assertTrue(j['process_termination_confirmed'])
    def test_media_hash_tamper_and_wrong_mime_rejected(self):
        manager=self.manager();identity,_=self.runjob(manager);path=manager.media_path(identity,'episode.mp4');path.write_bytes(b'changed')
        with self.assertRaises(ValueError):manager.media_path(identity,'episode.mp4')
        with self.assertRaises(ValueError):m.public_progress({'actual':{'file_path':'private'}})
    def test_unknown_restart_holds_exclusivity_without_history_write(self):
        manager=self.manager();d=manager.directory/'performance_fixture';d.mkdir();m.write_new(d/'request.json',{'instruction':''});m.write_new(d/'state.json',{'schema':'pulsemorph.performance_app_job.v1','id':d.name,'instruction':'','events':[],'created_at':'fixture','status':'running','finalized':False,'worker_attempts':1});raw=(d/'state.json').read_bytes();manager._load();self.assertEqual(manager.active_id(),d.name)
        with self.assertRaises(m.Busy):manager.create({'instruction':''})
        self.assertEqual((d/'state.json').read_bytes(),raw)
    def test_HTTP_guard_and_duplicate_json_before_create(self):
        h=s.Handler.__new__(s.Handler);h.server=SimpleNamespace(server_port=8778,csrf='fixture-token',manager=SimpleNamespace(create=lambda v:self.fail('must not create')));h.path='/api/jobs';h.connection=SimpleNamespace(settimeout=lambda _:None);h.headers={'Host':'127.0.0.1:8778','Content-Type':'application/json','Content-Length':'2'};replies=[];h._json=lambda c,v:replies.append((c,v));h.rfile=io.BytesIO(b'{}');h.do_POST();self.assertEqual(replies[-1][0],403)
        h.headers.update({'X-PulseMorph-Token':'fixture-token','Origin':'https://foreign.invalid'});h.do_POST();self.assertEqual(replies[-1][0],403);h.headers.pop('Origin');raw=b'{"instruction":"a","instruction":"b"}';h.headers['Content-Length']=str(len(raw));h.rfile=io.BytesIO(raw);h.do_POST();self.assertEqual(replies[-1][0],400)

if __name__=='__main__':unittest.main()
