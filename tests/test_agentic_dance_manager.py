"""Injected runtime/music/render fixtures: no model, physics, FK or compilation."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.dance import manager_v1 as m
from pulsemorph.agentic.dance.planner_v1 import FIELDS
from pulsemorph.agentic.phrase_program import ProgramCompileError

class Immediate:
    def submit(self,fn,*args):fn(*args)
    def shutdown(self,**kwargs):pass
class Runtime:
    _implementation={}
    def __init__(self,outcomes):self.outcomes=iter(outcomes);self.calls=[]
    def status(self):return {'available':True}
    def _unchanged(self):pass
    def _verify_trial(self,t):
        if t['metrics']!=json.loads((Path(t['paths']['directory'])/'result.json').read_bytes())['metrics']:raise RuntimeError('raw changed')
    def run_trial(self,p,d,jid,index,music,cancel):
        self.calls.append((index,copy.deepcopy(p),music));outcome=next(self.outcomes);d.mkdir(parents=True);(d/'common_reference').mkdir()
        if outcome=='reject':raise ProgramCompileError('fixture range rejection',{'fixture':'continuous range rejection'})
        (d/'worker_started.json').write_text('{}');metrics={'physical_gate_passed':outcome,'min_root_height_m':.8,'mean_joint_rmse_rad':.1}
        (d/'result.json').write_text(json.dumps({'status':'completed','integrity_passed':True,'metrics':metrics}))
        for name in ['manifest.json','arrays.npz']:(d/'common_reference'/name).write_bytes(b'fixture reference')
        return {'id':jid+'__'+str(index),'index':index,'parameters':copy.deepcopy(p),'metrics':metrics,'integrity_passed':True,'paths':{'directory':str(d),'common_reference':str(d/'common_reference')}}
class Provider:
    def status(self):return {'enabled':True}
def renderer(root,runtime,baseline,selected,music,d,**kwargs):
    d.mkdir();(d/'comparison.mp4').write_bytes(b'fixture compare');(d/'selected.mp4').write_bytes(b'fixture selected');meta={'baseline_id':baseline['id'],'selected_id':selected['id'],'comparison_sha256':m.sha(d/'comparison.mp4'),'selected_sha256':m.sha(d/'selected.mp4')};(d/'manifest.json').write_text(json.dumps(meta));return {k:str(d/v) for k,v in [('comparison','comparison.mp4'),('selected','selected.mp4'),('manifest','manifest.json')]}

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=m.CACHE);self.d=Path(self.temp.name);self.audio=self.d/'clip.wav';self.audio.write_bytes(b'fixture audio');self.music={'id':'fixture','path':str(self.audio),'sha256':m.sha(self.audio),'duration_s':4,'sample_rate':48000,'sample_frames':192000}
        self.managers=[]
    def tearDown(self):
        for manager in self.managers:manager._pool.shutdown(wait=True)
        self.temp.cleanup()
    def manager(self,outcomes):
        with patch.object(m.DanceManager,'implementation_pins',lambda s:{'fixture':'unchanged'}):
            manager=m.DanceManager(ROOT,Runtime(outcomes),Provider(),lambda _:copy.deepcopy(self.music),jobs_dir=self.d/'jobs',renderer=renderer)
        manager.implementation_pins=lambda:{'fixture':'unchanged'};manager._pool.shutdown();manager._pool=Immediate();self.managers.append(manager);return manager
    def planner(self,supported=True):
        def fn(provider,request,context,out,**kw):
            out.mkdir(parents=True);(out/'attempt.json').write_text('{}');receipt={'live':True};(out/'receipt.json').write_text(json.dumps(receipt));return {'supported':supported,'locally_supported':supported,'intent_summary':'fixture','unsupported_requests':[] if supported else ['new steps'],'parameters':m.unit_parameters(),'expected_changes':[],'receipt':receipt}
        return fn
    def runjob(self,manager,planner=None):
        with patch.object(m,'plan_dance',planner or self.planner()):identity=manager.create({'instruction':'edit existing dance arms','music_id':'fixture'})
        return manager.get(identity,private=True)
    def test_first_physical_pass_stops_without_intent_score(self):
        manager=self.manager([True,True]);job=self.runjob(manager)
        self.assertEqual(job['result'],'completed');self.assertTrue(job['finalized']);self.assertEqual(job['new_model_calls'],1);self.assertEqual(job['new_physical_trials'],2);self.assertEqual(len(manager.runtime.calls),2);self.assertNotIn('goals',job);self.assertIn('/selected.mp4',job['video_url'])
        self.assertEqual(manager.runtime.calls[0][1],m.baseline_parameters());self.assertIn('notunit',job['baseline']['baseline_qualification'].lower().replace(' ',''))
        self.assertEqual(manager.media_path(job['id'],'comparison').name,'comparison.mp4')
    def test_failures_get_all_actual_feedback_and_baseline_selected(self):
        manager=self.manager([True,False,'reject',False]);feedback=[];base=self.planner()
        def plan(*args,**kwargs):feedback.append(copy.deepcopy(kwargs['feedback']));return base(*args,**kwargs)
        job=self.runjob(manager,plan);self.assertEqual(job['result'],'partial');self.assertEqual(job['selected_id'],job['baseline']['id']);self.assertEqual(job['new_model_calls'],3);self.assertEqual(job['new_physical_trials'],3);self.assertEqual(job['reference_compile_attempts'],4)
        self.assertEqual(len(feedback[2]['trials']),2);self.assertIsNone(feedback[2]['trials'][1]['metrics']);self.assertEqual(feedback[2]['trials'][0]['metrics']['physical_gate_passed'],False)
    def test_unsupported_no_physics(self):
        manager=self.manager([]);job=self.runjob(manager,self.planner(False));self.assertEqual(job['status'],'unsupported');self.assertEqual(manager.runtime.calls,[])
    def test_provider_error_no_retry_and_finalized(self):
        manager=self.manager([])
        def fail(*args,**kwargs):raise RuntimeError('private provider detail')
        job=self.runjob(manager,fail);self.assertEqual(job['status'],'failed');self.assertEqual(manager.runtime.calls,[]);self.assertNotIn('private',job['error'])
    def test_baseline_failed_stops_before_candidate(self):
        manager=self.manager([False]);job=self.runjob(manager);self.assertEqual(job['status'],'failed');self.assertEqual(len(manager.runtime.calls),1)
    def test_raw_tamper_blocks_media_export(self):
        manager=self.manager([True,True]);job=self.runjob(manager);path=manager.media_path(job['id'],'video');path.write_bytes(b'tampered')
        with self.assertRaises(ValueError):manager.media_path(job['id'],'video')
    def test_busy_cancel_before_provider_preserves_request(self):
        manager=self.manager([])
        class Pending(Immediate):
            def submit(self,fn,*args):self.work=(fn,args)
        manager._pool=Pending();identity=manager.create({'instruction':'edit dance arms','music_id':'fixture'})
        with self.assertRaises(ValueError):manager.create({'instruction':'edit again','music_id':'fixture'})
        manager.cancel(identity);fn,args=manager._pool.work
        with patch.object(m,'plan_dance',side_effect=AssertionError('provider must not run')):fn(*args)
        job=manager.get(identity,private=True);self.assertEqual(job['status'],'cancelled');self.assertEqual(job['model_call_attempts'],0);self.assertTrue((manager.jobs_dir/identity/'request.json').exists())
    def test_changed_sources_deny_before_model(self):
        manager=self.manager([]);manager.implementation_pins=lambda:{'fixture':'changed'}
        with self.assertRaises(ValueError):manager.create({'instruction':'edit dance arms','music_id':'fixture'})
        self.assertEqual(manager.list_jobs(),[])
    def test_source_change_during_model_stops_without_worker(self):
        manager=self.manager([]);base=self.planner()
        def plan(*args,**kwargs):
            result=base(*args,**kwargs);manager.implementation_pins=lambda:{'fixture':'changed'};return result
        job=self.runjob(manager,plan);self.assertEqual(job['status'],'failed');self.assertEqual(job['new_model_calls'],1);self.assertEqual(manager.runtime.calls,[])
    def test_malformed_history_preserved_and_ignored(self):
        manager=self.manager([]);bad=manager.jobs_dir/'broken';bad.mkdir();raw=b'{"schema":"wrong"}';(bad/'state.json').write_bytes(raw)
        manager._load_history();self.assertTrue(manager.history_warnings);self.assertEqual((bad/'state.json').read_bytes(),raw)

if __name__=='__main__':unittest.main()
