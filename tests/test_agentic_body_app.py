"""Body Lab facade/process fixtures only; no worker, provider or simulator."""
import io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.app import manager_v1 as m,server_v1 as s

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=m.CACHE);self.addCleanup(self.temp.cleanup);self.d=Path(self.temp.name);self.worker=self.d/'fixture_worker.py';self.worker.write_text('# never executed');self.argv=[]
    def manager(self,complete=True,result=None):
        def spawn(argv,**kwargs):
            self.argv.append(argv);out=Path(argv[argv.index('--output')+1]);out.mkdir();(out/'progress.json').write_text(json.dumps({'status':'completed','phase':'finished','actual':{'sim_time_s':0.,'goals':[],'task_verified':False},'counts':{'model_calls':0,'actual_steps':0,'sandbox_steps':0},'preview':{'hypothetical_only':True,'status':'none'}}))
            if complete:(out/'result.json').write_text(json.dumps(result or {'status':'stopped','task_verified_actual':False}))
            return SimpleNamespace(pid=999999,poll=lambda:0)
        return m.BodyAgentManager(ROOT,jobs_dir=self.d/'jobs',worker=self.worker,profile_status=lambda:{'available':True,'state':'fixture'},popen=spawn)
    def test_fixture_process_arguments_fixed_goals_and_honest_counts(self):
        manager=self.manager();identity=manager.create({'scenario':'both_reach','instruction':'İki hedefi aynı anda koru'});manager.threads[identity].join(3);j=manager.get(identity)
        self.assertTrue(j['finalized']);self.assertEqual(j['status'],'stopped');self.assertEqual(len(j['fixed_goals']),2);self.assertEqual(j['counts']['model_calls'],0);self.assertEqual(j['counts']['actual_steps'],0);self.assertIn('-I',self.argv[0]);self.assertIn('-B',self.argv[0]);self.assertNotIn('source_pins',j);self.assertNotIn('pid',j)
        self.assertEqual(m.read_json(manager.jobs_dir/identity/'request.json'),{'scenario':'both_reach','instruction':'İki hedefi aynı anda koru'})
    def test_missing_terminal_receipt_is_unknown_not_zero_work(self):
        manager=self.manager(False);identity=manager.create({'scenario':'right_reach','instruction':''});manager.threads[identity].join(3);j=manager.get(identity);self.assertEqual(j['status'],'failed');self.assertTrue(j['completion_unconfirmed']);self.assertEqual(j['worker_attempts'],1)
    def test_strict_request_and_progress_no_private_paths(self):
        for v in ({'scenario':'new_skill','instruction':''},{'scenario':'right_reach','instruction':'','target':0},{'scenario':'right_reach','instruction':None}):
            with self.assertRaises(ValueError):m.parse_request(v)
        with self.assertRaises(ValueError):m.clean_progress({'actual':{'path':'C:/private'}})
        with self.assertRaises(ValueError):m.clean_progress({'phase':'C:/private'})
        self.assertNotIn('source_pins',m.clean_progress({'phase':'thinking','source_pins':{'private':'ignored'}}))
    def test_unfinished_history_restart_remains_preserved(self):
        manager=self.manager();d=manager.jobs_dir/'body_fixture';d.mkdir();request={'scenario':'right_reach','instruction':''};m.write_new(d/'request.json',request);state={'schema':'pulsemorph.body_agent_job.v1','id':d.name,**request,'fixed_goals':m.fixed_goals('right_reach'),'status':'running','finalized':False,'created_at':'fixture','events':[]};m.write_new(d/'state.json',state);before=(d/'state.json').read_bytes();manager._load();j=manager.get(d.name);self.assertEqual(j['status'],'failed');self.assertTrue(j['completion_unconfirmed']);self.assertEqual((d/'state.json').read_bytes(),before)
    def test_media_roles_and_hash_binding(self):
        manager=self.manager();identity=manager.create({'scenario':'right_reach','instruction':''});manager.threads[identity].join(3);execution=manager.jobs_dir/identity/'execution';(execution/'preview.jpg').write_bytes(b'fixture hypothetical picture');raw={'phase':'preview','media':{'preview.jpg':{'role':'hypothetical','sha256':m.sha(execution/'preview.jpg')}}};(execution/'progress.json').write_text(json.dumps(raw));manager._progress(manager.jobs[identity],execution);j=manager.get(identity);self.assertIn('/preview.jpg',j['preview_image_url']);self.assertIsNone(j['actual_image_url']);manager.media_path(identity,'preview.jpg').write_bytes(b'tamper')
        with self.assertRaises(ValueError):manager.media_path(identity,'preview.jpg')
    def test_http_token_host_origin_and_duplicate_body(self):
        h=s.Handler.__new__(s.Handler);h.server=SimpleNamespace(server_port=8777,csrf='fixture-token',manager=SimpleNamespace(create=lambda v:self.fail('must not create')));h.path='/api/jobs';h.connection=SimpleNamespace(settimeout=lambda _:None);h.headers={'Host':'127.0.0.1:8777','Content-Type':'application/json','Content-Length':'2'};replies=[];h._json=lambda c,v:replies.append((c,v));h.rfile=io.BytesIO(b'{}');h.do_POST();self.assertEqual(replies[-1][0],403)
        h.headers.update({'X-PulseMorph-Token':'fixture-token','Origin':'https://foreign.invalid'});h.do_POST();self.assertEqual(replies[-1][0],403)
        raw=b'{"scenario":"right_reach","scenario":"both_reach","instruction":""}';h.headers.pop('Origin');h.headers['Content-Length']=str(len(raw));h.rfile=io.BytesIO(raw);h.do_POST();self.assertEqual(replies[-1][0],400)
    def test_restart_unknown_worker_keeps_exclusive_lock(self):
        manager=self.manager();d=manager.jobs_dir/'body_unknown';d.mkdir();request={'scenario':'right_reach','instruction':''};m.write_new(d/'request.json',request);m.write_new(d/'state.json',{'schema':'pulsemorph.body_agent_job.v1','id':d.name,**request,'fixed_goals':m.fixed_goals('right_reach'),'status':'failed','finalized':False,'worker_attempts':1,'created_at':'fixture','events':[]});manager._load();self.assertEqual(manager.active_id(),d.name)
        with self.assertRaises(m.BodyJobBusy):manager.create(request)
    def test_worker_video_path_and_decision_events_merge_once(self):
        manager=self.manager();identity=manager.create({'scenario':'right_reach','instruction':''});manager.threads[identity].join(3);execution=manager.jobs_dir/identity/'execution';(execution/'media').mkdir();video=execution/'media/episode.mp4';video.write_bytes(b'fixture recorded actual video');progress={'phase':'finished','events':[{'index':0,'kind':'model_decision','message':'Sanal adayı dene','wall_seconds':1.}], 'media':{'episode.mp4':{'path':'media/episode.mp4','role':'actual_recorded','sha256':m.sha(video)}}};(execution/'progress.json').write_text(json.dumps(progress));manager._progress(manager.jobs[identity],execution);manager._progress(manager.jobs[identity],execution);job=manager.get(identity);self.assertEqual(sum(e['kind']=='model_decision' for e in job['events']),1);self.assertEqual(manager.media_path(identity,'episode.mp4'),video);self.assertIn('episode.mp4',job['video_url']);self.assertFalse(m.media_relative('episode.mp4','images/preview_000.jpg'))
    def test_failed_media_exposes_safe_error_preserves_achieved_measurement(self):
        manager=self.manager(result={'status':'failed','task_verified_actual':True,'media_incomplete':True});identity=manager.create({'scenario':'right_reach','instruction':''});manager.threads[identity].join(3);job=manager.get(identity);self.assertEqual(job['error'],'MEDIA_INCOMPLETE');self.assertTrue(job['task_verified_actual']);self.assertFalse(job['completion_unconfirmed']);self.assertTrue(job['finalized'])
    def test_terminal_worker_still_reports_uncertain_physics_or_model_completion(self):
        manager=self.manager(result={'status':'failed','task_verified_actual':False,'counts':{'physical_completion_unconfirmed':True,'model_completion_unconfirmed':1}});identity=manager.create({'scenario':'right_reach','instruction':''});manager.threads[identity].join(3);job=manager.get(identity);self.assertTrue(job['completion_unconfirmed']);self.assertTrue(job['finalized']);self.assertTrue(manager.get(identity,private=True)['process_termination_confirmed'])

if __name__=='__main__':unittest.main()
