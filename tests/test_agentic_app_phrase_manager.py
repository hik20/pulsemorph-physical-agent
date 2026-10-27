"""Mock orchestration boundaries only; no actual LLM or physical execution."""
import copy,tempfile,threading,time,unittest,json
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from pulsemorph.agentic.app.phrase.manager import PhraseManager,SOURCE_ID,PhraseBusy
from pulsemorph.agentic.phrase_program import ProgramCompileError
from pulsemorph.agentic.app.providers import ProviderError

PROGRAM={'schema':'pulsemorph.phrase_program.v1','order':['wave','high5'],'heading_policy':'align_join_heading','first_left_gain':.85,'first_right_gain':1.,'second_left_gain':1.15,'second_right_gain':1.15,'rationale':'mock bounded intent, not physical evidence'}
class MusicFixture:
    def get(self,identity):return {'id':'fixture_music','label':'mock music','schema':'fixture','path':'C:/fixture-not-executed.wav','sha256':'a'*64,'duration_s':8,'sample_rate':48000,'sample_frames':384000,'channels':1,'pcm_bits':16,'sections':[]}
    def catalog(self):return [self.get('fixture_music')]
class ProviderFixture:
    model='gpt-6.1-sol'
    def __init__(self,initial=None,feedback=None,error=None):self.initial=initial or {'supported':True,'intent_summary':'mock','unsupported_requests':[],'program':copy.deepcopy(PROGRAM)};self.feedback=feedback or copy.deepcopy(PROGRAM);self.error=error;self.calls=[]
    def status(self):return {'provider':'mock','enabled':True}
    def generate_json(self,prompt,schema,**kwargs):
        self.calls.append((prompt,schema))
        if self.error:raise self.error
        value=self.initial if len(self.calls)==1 else self.feedback
        return SimpleNamespace(data=copy.deepcopy(value),receipt={'provider':'fixture','model_requested':self.model,'model_returned':None,'tool_events':0,'live':False,'usage':{}})
class MemoryFixture:
    def __init__(self):self.records=[]
    def status(self):return {'meterless':{'state':'fixture'},'zetaris':{'state':'missing_configuration'}}
    def query_sources(self,*a,**k):return {'memory':[],'zetaris':{'executed':False,'records':[],'usable_context_verified':False}}
    def record_job(self,j):self.records.append(('job',j))
    def record_trial(self,j,t):self.records.append(('trial',t))
class RuntimeFixture:
    def __init__(self,reject_first=False,integrity_fail=False,block=False):self.calls=[];self.reject=reject_first;self.integrity=integrity_fail;self.block=block
    def run_trial(self,program,output,job_id,trial_index,music_observation,cancel_event):
        self.calls.append(trial_index)
        if self.block:
            cancel_event.wait(2)
            if cancel_event.is_set():raise RuntimeError('fixture cancelled')
        if trial_index==0 and self.reject:raise ProgramCompileError('fixture reference rejection',{'category':'fixture range'})
        return {'id':job_id+f'_{trial_index}','index':trial_index,'program':program,'paths':{'directory':str(output)},'metrics':{'arm_rms_rad':[1]*4}}
    def evaluate_trial(self,trial,baseline,targets):
        for item in (trial,baseline):
            raw=item.get('physical_metrics',item['metrics'])
            if raw!={'arm_rms_rad':[1]*4}:raise RuntimeError('fixture physical metrics binding lost')
        if trial['index']==0 and self.integrity:raise RuntimeError('fixture integrity failure')
        features=[1.]*4 if trial['index']==-1 else [trial['program'][k] for k in ('first_left_gain','first_right_gain','second_left_gain','second_right_gain')];errors=[abs(a-b)/.08 for a,b in zip(features,targets)]
        return {'integrity_passed':True,'features':features,'targets':targets,'tolerances':[.08]*4,'normalized_errors':errors,'physical_gate_passed':True,'worst_error':max(errors),'utility_l2':sum(v*v for v in errors)}
def renderer(root,baseline,selected,music,output,cancel_event=None):
    assert baseline['physical_metrics']=={'arm_rms_rad':[1]*4}
    assert selected['physical_metrics']=={'arm_rms_rad':[1]*4}
    assert selected['metrics']['arm_rms_rad']==[1]*4
    return {n:str(output/(n+'.mp4')) for n in ('before','after','comparison')}
class PhraseManagerTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.memory=MemoryFixture();self.instances=[]
    def make(self,provider=None,runtime=None):
        m=PhraseManager(self.root,provider or ProviderFixture(),runtime or RuntimeFixture(),MusicFixture(),self.memory,renderer,self.root/'app');self.instances.append(m);return m
    def tearDown(self):
        for m in self.instances:m.shutdown()
    def done(self,m,identity):
        for _ in range(300):
            j=m.get(identity)
            if j['status'] in {'complete','failed','cancelled'}:return j
            time.sleep(.01)
        self.fail('mock job did not finish')
    def create(self,m,budget=2):return m.create({'instruction':'mock wave/high5 instruction','music_id':'fixture_music','max_iterations':budget})
    def test_actual_fixed_targets_baseline_candidate_and_history_reopen(self):
        runtime=RuntimeFixture();provider=ProviderFixture();m=self.make(provider,runtime);j=self.done(m,self.create(m));self.assertEqual(j['status'],'complete');self.assertEqual(runtime.calls,[-1,0]);self.assertEqual(j['selected']['index'],0);self.assertEqual(j['targets'],[.85,1.,1.15,1.15]);self.assertEqual(j['selected']['metrics']['goals_passed'],4);self.assertEqual(len(provider.calls),1);self.assertIn('before_url',j['media']);self.assertTrue(all(t[1].get('source_id')==SOURCE_ID for t in self.memory.records if t[0]=='trial'));m.shutdown();reopened=self.make();self.assertEqual(reopened.get(j['id'])['selected']['index'],0)
    def test_explicit_unsupported_or_partial_request_zero_physics(self):
        for supported,unsupported in [(False,['backflip']), (True,['guaranteed style'])]:
            p=ProviderFixture({'supported':supported,'intent_summary':'unsupported request','unsupported_requests':unsupported,'program':PROGRAM});r=RuntimeFixture();m=self.make(p,r);j=self.done(m,self.create(m));self.assertEqual(j['outcome'],'unsupported');self.assertEqual(r.calls,[]);self.assertIsNone(j['selected'])
    def test_reverse_order_not_admitted(self):
        program=dict(PROGRAM,order=['high5','wave']);p=ProviderFixture({'supported':True,'intent_summary':'reverse','unsupported_requests':[],'program':program});r=RuntimeFixture();m=self.make(p,r);j=self.done(m,self.create(m));self.assertEqual(j['outcome'],'unsupported');self.assertEqual(r.calls,[])
    def test_compile_rejection_consumes_budget_feedback_targets_frozen(self):
        p=ProviderFixture(feedback=dict(PROGRAM,first_left_gain=1.1));r=RuntimeFixture(reject_first=True);m=self.make(p,r);j=self.done(m,self.create(m,2));self.assertEqual(j['status'],'complete');self.assertEqual(r.calls,[-1,0,1]);self.assertEqual(j['trials'][0]['status'],'rejected');self.assertIsNone(j['trials'][0]['metrics']);self.assertEqual(j['targets'],[.85,1.,1.15,1.15]);self.assertEqual(len(p.calls),2);self.assertIn('Recorded feedback',p.calls[1][0])
    def test_integrity_failure_stops_and_preserves_baseline(self):
        r=RuntimeFixture(integrity_fail=True);m=self.make(runtime=r);j=self.done(m,self.create(m,3));self.assertEqual(j['status'],'failed');self.assertEqual(r.calls,[-1,0]);self.assertEqual(j['selected']['index'],-1);self.assertEqual(j['trials'][0]['status'],'execution_failed');self.assertIsNone(j['trials'][0]['metrics'])
    def test_provider_failure_durable_unconfirmed_no_retry(self):
        p=ProviderFixture(error=ProviderError('Codex failed; check local authentication/model access'));r=RuntimeFixture();m=self.make(p,r);j=self.done(m,self.create(m));self.assertEqual(j['status'],'failed');self.assertEqual(len(p.calls),1);self.assertTrue(j['model_calls'][0]['completion_unconfirmed']);self.assertEqual(r.calls,[])
    def test_disabled_provider_and_explicit_unavailable_runtime_no_queue(self):
        provider=ProviderFixture();provider.status=lambda:{'enabled':False};m=self.make(provider)
        with self.assertRaises(ValueError):self.create(m)
        self.assertEqual(m.list_jobs(),[]);self.assertEqual(provider.calls,[])
        runtime=RuntimeFixture();runtime.status=lambda:{'available':False};provider=ProviderFixture();m=self.make(provider,runtime)
        with self.assertRaises(ValueError):self.create(m)
        self.assertEqual(m.list_jobs(),[]);self.assertEqual(provider.calls,[])
    def test_live_requires_completed_live_receipts_not_mock_or_unknown(self):
        provider=ProviderFixture();m=self.make(provider);j=self.done(m,self.create(m));self.assertEqual(j['outcome'],'target_met');self.assertFalse(j['live'])
        provider=ProviderFixture(error=ProviderError('fixture unknown'));m=self.make(provider);j=self.done(m,self.create(m));self.assertFalse(j['live']);self.assertTrue(j['model_calls'][0]['completion_unconfirmed'])
    def test_completed_receipt_saved_when_cancel_arrives_at_return(self):
        started=threading.Event();release=threading.Event();provider=ProviderFixture();original=provider.generate_json
        def model(*args,**kwargs):
            started.set();release.wait(2);return original(*args,**kwargs)
        provider.generate_json=model;m=self.make(provider);identity=self.create(m);self.assertTrue(started.wait(2));m.cancel(identity);release.set();j=self.done(m,identity);self.assertEqual(j['status'],'cancelled');self.assertEqual(len(j['model_calls']),1);self.assertFalse(j['model_calls'][0]['completion_unconfirmed']);self.assertFalse(j['live'])
    def test_restart_malformed_states_are_ignored_and_preserved(self):
        directory=self.root/'app/jobs';directory.mkdir(parents=True)
        for i,value in enumerate([[],None,{}, {'id':'bad','source_id':SOURCE_ID,'request':{},'status':'complete'}]):
            path=directory/f'bad_{i}'/'state.json';path.parent.mkdir();path.write_text(json.dumps(value))
        m=self.make();self.assertEqual(m.list_jobs(),[]);self.assertEqual(len(list(directory.glob('*/state.json'))),4)
    def test_source_change_after_startup_rejected_before_queue(self):
        provider=ProviderFixture();m=self.make(provider);changed=dict(m._implementation);changed['fixture_new_code']='changed'
        with patch.object(m,'_current_implementation',return_value=changed),self.assertRaisesRegex(ValueError,'restart required'):self.create(m)
        self.assertEqual(m.list_jobs(),[]);self.assertEqual(provider.calls,[])
    def test_invalid_returned_JSON_preserves_completed_receipt_safely(self):
        initial={'supported':True,'intent_summary':'fixture','unsupported_requests':[],'program':dict(PROGRAM,first_left_gain=float('nan'))};provider=ProviderFixture(initial);runtime=RuntimeFixture();m=self.make(provider,runtime);j=self.done(m,self.create(m));self.assertEqual(j['status'],'failed');self.assertEqual(runtime.calls,[]);self.assertEqual(len(j['model_calls']),1);self.assertFalse(j['model_calls'][0]['completion_unconfirmed']);self.assertTrue(j['model_calls'][0]['response_rejected']);self.assertFalse(j['live'])
    def test_cancel_busy_and_invalid_budget(self):
        r=RuntimeFixture(block=True);m=self.make(runtime=r);identity=self.create(m)
        with self.assertRaises(PhraseBusy):self.create(m)
        m.cancel(identity);j=self.done(m,identity);self.assertEqual(j['status'],'cancelled')
        for budget in [True,0,4]:
            with self.assertRaises(ValueError):self.create(m,budget)
if __name__=='__main__':unittest.main()
