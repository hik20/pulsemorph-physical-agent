"""Versioned desired goals and independent controls; fixtures never execute physics."""
import copy,json,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_agentic_app_phrase_manager import MusicFixture,ProviderFixture,RuntimeFixture,MemoryFixture,renderer,PROGRAM
from pulsemorph.agentic.app.phrase.goals_v1 import validate_goals
from pulsemorph.agentic.app.phrase.manager_goals_v4 import PhraseGoalsManager,INITIAL_SCHEMA_V4
from pulsemorph.agentic.app.phrase import exports_goals_v4 as ex
GOALS={'schema':'pulsemorph.phrase_goals.v1','ratios':[.75,1.25,1.25,.75]}
class GoalTests(unittest.TestCase):
    def test_strict_goal_dto(self):
        self.assertEqual(validate_goals(GOALS),GOALS)
        for value in (None,[],{**GOALS,'extra':0},{'schema':GOALS['schema'],'ratios':[True,1,1,1]},{'schema':GOALS['schema'],'ratios':[float('nan'),1,1,1]},{'schema':GOALS['schema'],'ratios':[.74,1,1,1]},{'schema':GOALS['schema'],'ratios':[10**1000,1,1,1]}):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_goals(value)
    def make(self,initial=None,runtime=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);p=ProviderFixture(initial=initial or self.initial());r=runtime or RuntimeFixture()
        m=PhraseGoalsManager(root,p,r,MusicFixture(),MemoryFixture(),renderer,root/'app');self.addCleanup(m.shutdown)
        return m,p,r
    def initial(self,**kw):return dict(supported=True,intent_summary='fixture typed goals',unsupported_requests=[],program=copy.deepcopy(PROGRAM),goals=copy.deepcopy(GOALS),**kw)
    def done(self,m,identity):
        for _ in range(300):
            j=m.get(identity)
            if j['status'] in ('complete','failed','cancelled'):return j
            time.sleep(.01)
        self.fail('fixture timeout')
    def create(self,m,explicit=True,budget=1):return m.create({'instruction':'Use application measured feedback, not tool calls in this model response','music_id':'fixture_music','max_iterations':budget,**({'goals':GOALS} if explicit else {})})
    def test_explicit_persisted_before_model_independent_control_partial(self):
        m,p,r=self.make();entered=threading.Event();release=threading.Event();old=p.generate_json
        def blocked(*a,**k):entered.set();release.wait(2);return old(*a,**k)
        p.generate_json=blocked;identity=self.create(m);self.assertTrue(entered.wait(2));early=m.get(identity)
        self.assertEqual(early['targets'],GOALS['ratios']);self.assertEqual(early['request']['goals'],GOALS);self.assertEqual(early['goal_origin'],'explicit_user');self.assertIsNone(early['plan'])
        release.set();j=self.done(m,identity);self.assertEqual(j['status'],'complete');self.assertEqual(j['targets'],GOALS['ratios']);self.assertEqual(j['plan']['program'],PROGRAM);self.assertEqual(j['outcome'],'partial');self.assertEqual(r.calls,[-1,0]);self.assertEqual(len(p.calls),1)
        prepared=json.loads((m.jobs_dir/identity/'prepared_plan_v4.json').read_text());self.assertEqual(prepared['runtime_capability']['max_candidates'],1);self.assertEqual(prepared['runtime_capability']['max_model_calls'],1);self.assertEqual(prepared['runtime_capability']['max_feedback_calls'],0)
        self.assertIn('application, not this call',p.calls[0][0]);self.assertNotIn('Unsupported desires must be described as limitations',p.calls[0][0])
    def test_explicit_echo_mismatch_zero_physics(self):
        initial=self.initial();initial['goals']['ratios'][0]=1.;m,p,r=self.make(initial);j=self.done(m,self.create(m));self.assertEqual(j['status'],'failed');self.assertEqual(j['error'],'EXPLICIT_GOALS_ECHO_MISMATCH');self.assertEqual(r.calls,[]);self.assertEqual(j['targets'],GOALS['ratios'])
    def test_unsupported_saved_response_still_zero_physics(self):
        initial=self.initial();initial.update(supported=False,unsupported_requests=['fixture unsupported guarantee']);m,p,r=self.make(initial);j=self.done(m,self.create(m));self.assertEqual(j['outcome'],'unsupported');self.assertEqual(r.calls,[])
    def test_model_interpretation_explicitly_labelled(self):
        m,p,r=self.make();j=self.done(m,self.create(m,False));self.assertEqual(j['goal_origin'],'model_interpretation');self.assertEqual(j['plan']['goal_origin'],'model_interpretation');self.assertNotIn('goals',j['request']);self.assertEqual(j['targets'],GOALS['ratios'])
    def test_rejection_feedback_preserves_goals_controls_independent(self):
        m,p,r=self.make(runtime=RuntimeFixture(reject_first=True));j=self.done(m,self.create(m,budget=2));self.assertEqual(j['status'],'complete');self.assertEqual(r.calls,[-1,0,1]);self.assertEqual(j['trials'][0]['status'],'rejected');self.assertIsNone(j['trials'][0]['metrics']);self.assertEqual(j['targets'],GOALS['ratios']);self.assertIn('reference_compile_rejected',p.calls[1][0]);self.assertIn(json.dumps(GOALS['ratios']),p.calls[1][0])
    def binding(self):
        j={'schema':'pulsemorph.phrase_app_job.v2','goal_origin':'explicit_user','request':{'goals':copy.deepcopy(GOALS),'instruction':'fixture'},'plan':{'goals':copy.deepcopy(GOALS),'goal_origin':'explicit_user'},'music':{'sha256':'a'*64}}
        prepared={'schema':'pulsemorph.phrase_app_prepared_plan.v4','explicit_goals':copy.deepcopy(GOALS),'instruction':'fixture','response_schema':copy.deepcopy(INITIAL_SCHEMA_V4),'prompt':'fixture prompt','computed_music_summary':{'sha256':'a'*64}}
        return j,{'goals':copy.deepcopy(GOALS)},prepared,{'schema':copy.deepcopy(INITIAL_SCHEMA_V4),'prompt':'fixture prompt'}
    def test_export_goal_model_request_binding_and_tamper(self):
        args=self.binding();self.assertEqual(ex.validate_goal_binding(*args),GOALS)
        for index,key,value in ((0,'goal_origin','invented'),(2,'explicit_goals',None),(3,'prompt','different')):
            changed=copy.deepcopy(args);changed[index][key]=value
            with self.subTest(key=key),self.assertRaises(ex.PhraseResultExportError):ex.validate_goal_binding(*changed)
    def test_old_history_export_dispatch_unchanged(self):
        class Manager:
            def get(self,_):return {'schema':'pulsemorph.phrase_app_job.v1'}
        m=Manager()
        with patch.object(ex.legacy,'build_phrase_result_bundle',return_value=b'legacy saved bytes') as build:self.assertEqual(ex.build_phrase_result_bundle(m,'legacy'),b'legacy saved bytes');build.assert_called_once_with(m,'legacy')
    def test_unknown_export_schema_never_downgrades(self):
        class Manager:
            def get(self,_):return {'schema':'unknown_future_job'}
        with patch.object(ex.legacy,'build_phrase_result_bundle') as build:
            with self.assertRaises(ex.PhraseResultExportError):ex.build_phrase_result_bundle(Manager(),'unknown')
            build.assert_not_called()
    def test_exported_json_goal_dto_distinct_from_measurement(self):
        import io,zipfile
        measurement={'features':[.81,1.2,1.1,.86],'targets':GOALS['ratios'],'goals_passed':2}
        result={'goal_measurement':copy.deepcopy(measurement),'selected_program':copy.deepcopy(PROGRAM)}
        ex.attach_goal_metadata(result,GOALS,'explicit_user')
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:archive.writestr('result.json',json.dumps(result))
        with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:exported=json.loads(archive.read('result.json'))
        self.assertEqual(exported['goals'],GOALS);self.assertEqual(exported['goal_measurement'],measurement);self.assertNotEqual(exported['goals'],exported['goal_measurement']);self.assertEqual(exported['selected_program'],PROGRAM)
    def test_full_builder_fixture_does_not_shadow_goal_dto(self):
        """Whole orchestration/serialization fixture; proof helpers explicitly mocked."""
        import io,zipfile,wave
        from contextlib import ExitStack
        args=self.binding();job,initial,prepared,request=args
        initial.update(program=copy.deepcopy(PROGRAM),intent_summary='fixture')
        music={'id':'phrase_original_pulse_v1','sha256':'a'*64,'label':'fixture','provenance':{},'rights':'fixture'}
        selected={'index':-1,'program':dict(PROGRAM,**{k:1. for k in ('first_left_gain','first_right_gain','second_left_gain','second_right_gain')})}
        job.update(id='fixture_job',source_id=ex.SOURCE_ID,status='complete',outcome='partial',selected=selected,baseline=selected,trials=[],implementation_hashes={},targets=GOALS['ratios'],tolerances=[.08]*4,music=music,selection_policy=ex.SELECTION_POLICY)
        job['plan'].update(fixed_targets=GOALS['ratios'],program=PROGRAM,supported=True,unsupported_requests=[])
        audio=io.BytesIO()
        with wave.open(audio,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(48000);w.writeframes(b'\0'*768000)
        measured={'features':[1.]*4,'targets':GOALS['ratios'],'goals_passed':0}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobdir=root/'jobs/fixture_job';trialdir=jobdir/'trials/baseline';trialdir.mkdir(parents=True);call=jobdir/'model_calls/00_plan';call.mkdir(parents=True);(call/'request.json').write_text('{}');source=root/'fixture.py';source.write_text('# fixture');job['implementation_hashes']={str(source):'a'*64}
            class Reader:
                cache={}
                def __init__(self,*a):pass
                def json(self,path,*a,**kw):
                    if path.name=='state.json':return copy.deepcopy(job)
                    if path.name=='prepared_plan_v4.json':return prepared
                    if path.name=='request.json':return request
                    if path.parent.name in ('wave','high5'):return {'metadata':{'reference_id':'fixture','provenance':{'sources':[]}}}
                    return music
                def data(self,*a,**k):return audio.getvalue()
                def hash(self,*a,**k):return 'a'*64
                def finish(self,*a):pass
            trial={'program':selected['program'],'physical':{'fixture':True},'common':{'metadata':{'provenance':{'sources':[]}}},'arrays_bytes':b'fixture numeric bytes','manifest_bytes':b'{}','data':{name:b'fixture numeric bytes' for name in ('qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy')},'proof':{},'payload_paths':[],'directory':trialdir,'raw_files':set()}
            class Manager:
                jobs_dir=root/'jobs'
                def get(self,_):return copy.deepcopy(job)
            manager=Manager();manager.root=root
            with ExitStack() as stack:
                for name,value in {'_Reader':Reader,'_model_plan':lambda *a:initial,'_profile':lambda *a:({},{}),'_trial':lambda *a:trial,'_metrics':lambda *a:copy.deepcopy(measured),'_portable_manifest':lambda *a:({},[]),'_reference':lambda *a:None,'_music_provenance':lambda *a:{'kind':'fixture'}}.items():stack.enter_context(patch.object(ex,name,value))
                bundle=ex.build_phrase_result_bundle(manager,'fixture_job')
            with zipfile.ZipFile(io.BytesIO(bundle)) as archive:result=json.loads(archive.read('result.json'));self.assertEqual(len(archive.namelist()),8)
            self.assertEqual(result['goals'],GOALS);self.assertEqual(result['goal_measurement'],measured);self.assertEqual(result['selected_index'],-1)
            self.assertIn('frozen BEFORE baseline',ex.README_V4);self.assertIn('ONLY the ratio denominator',ex.README_V4);self.assertNotIn('defines the fixed four RMS goals',ex.README_V4)

if __name__=='__main__':unittest.main()
