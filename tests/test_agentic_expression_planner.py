"""Mock one-call planning boundaries; no real provider/compiler/controller calls."""
import copy,json,sys,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from pulsemorph.agentic.expression import planner_v1 as p
from pulsemorph.agentic.app.providers import ProviderError
from pulsemorph.agentic.phrase_expression_v1 import describe_bases,GAIN_FIELDS
from test_agentic_phrase_expression import reference
GOALS={'schema':'pulsemorph.expression_goals.v1','mean_pose_ratio':1.2,'temporal_ratio':1.2}
def expression():return {'schema':'pulsemorph.phrase_expression.v1','order':['wave','high5'],'heading_policy':'align_join_heading',**{k:1. for k in GAIN_FIELDS},'second_left_pose_gain':1.2,'second_left_motion_gain':1.2,'rationale':'Bounded proposal, not measured success.'}
def request():return {'instruction':'High5 sol ortalama eklem farkı ve zamansal değişim +%20; diğer kollar aynı.','goals':copy.deepcopy(GOALS)}
def context():return {'music_summary':{'schema':'pulsemorph.phrase_music_summary.v1','sha256':'a'*64,'duration_s':8,'sample_rate':48000,'sample_frames':384000,'channels':1,'pcm_bits':16,'sections':[{'start_s':0,'end_s':4,'rms':.02},{'start_s':4,'end_s':8,'rms':.04}],'bpm':None},'source_descriptor':describe_bases({'wave':reference(),'high5':reference()}),'baseline':{'kind':'historical_unit_not_current_job','metrics':{'physical_gate_passed':True}},'observations':[{'qualification':'one_historical_realization_not_calibration','expression':expression(),'mean_pose_ratio':.812,'temporal_ratio':.996}]}
def response():return {'supported':True,'intent_summary':'Bounded two-control application request.','unsupported_requests':[],'expression':expression(),'goals':copy.deepcopy(GOALS)}
class Provider:
    model='gpt-6.1-sol'
    def __init__(self,data=None,error=None):self.data=data if data is not None else response();self.error=error;self.calls=[]
    def status(self):return {'enabled':True}
    def generate_json(self,prompt,schema,**kwargs):
        self.calls.append((prompt,schema,kwargs))
        if self.error:raise self.error
        return SimpleNamespace(data=copy.deepcopy(self.data),receipt={'provider':'fixture','model_requested':self.model,'model_returned':None,'usage':{'input_tokens':123,'output_tokens':45},'tool_events':0,'live':False,'elapsed_seconds':.01,'trust_boundary':'fixture','api_key':'NEVER_PERSIST_THIS_SECRET'})
class PlannerTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name);self.patch=patch.object(p,'OUTPUT_ROOT',self.base);self.patch.start();self.addCleanup(self.patch.stop)
    def call(self,provider=None,name='call',**kwargs):return p.plan_expression(provider or Provider(),request(),context(),self.base/name,**kwargs)
    def test_initial_prompt_fixed_goals_actual_music_roles_no_tools(self):
        provider=Provider();req=request();ctx=context();result=p.plan_expression(provider,req,ctx,self.base/'call')
        self.assertTrue(result['locally_supported']);self.assertEqual(result['goals'],GOALS);self.assertFalse(result['receipt']['live']);self.assertIsNone(result['receipt']['model_returned']);self.assertEqual(len(provider.calls),1)
        prompt,schema,kwargs=provider.calls[0];self.assertIn('Initial planning precedes the fresh baseline',prompt);self.assertIn('no raw audio is supplied/heard',prompt);self.assertIn('do not use tools',prompt);self.assertIn('0.04',prompt);self.assertIn('second_left_motion_gain',prompt);self.assertNotIn('uniqueItems',json.dumps(schema))
        req['goals']['mean_pose_ratio']=.75;ctx['music_summary']['sections'][0]['rms']=999;stored=json.loads((self.base/'call/request.json').read_text());self.assertEqual(stored['request']['goals'],GOALS);self.assertEqual(stored['context']['music_summary']['sections'][0]['rms'],.02)
        self.assertNotIn('NEVER_PERSIST_THIS_SECRET',''.join(f.read_text() for f in (self.base/'call').iterdir()))
    def test_revision_contains_all_actual_partial_and_rejection_feedback(self):
        feedback={'baseline':{'kind':'fresh_verified_unit','metrics':{'physical_gate_passed':True}},'trials':[{'index':0,'status':'measured_partial','actual_ratios':[1.05,1.1]},{'index':1,'status':'rejected','metrics':None,'report':{'violations':['left_elbow_joint']}}]};provider=Provider();result=self.call(provider,feedback=feedback)
        stored=json.loads((self.base/'call/request.json').read_text());self.assertEqual(stored['purpose'],'revision');self.assertEqual(stored['feedback'],feedback);self.assertIn('left_elbow_joint',provider.calls[0][0]);self.assertIn('1.05',provider.calls[0][0]);self.assertEqual(result['goals'],GOALS)
    def test_echo_and_frozen_six_rejections_preserve_completed_receipt(self):
        for field in ('goals','frozen'):
            data=response()
            if field=='goals':data['goals']['mean_pose_ratio']=1.
            else:data['expression']['first_left_pose_gain']=.9
            with self.subTest(field=field),self.assertRaises(p.PlannerResponseRejected):self.call(Provider(data),name=field)
            self.assertTrue((self.base/field/'response.json').is_file());receipt=json.loads((self.base/field/'receipt.json').read_text());self.assertFalse(receipt['completion_unconfirmed']);failed=json.loads((self.base/field/'failed_status.json').read_text());self.assertEqual(failed['error_category'],'EXPLICIT_GOALS_ECHO_MISMATCH' if field=='goals' else 'FROZEN_GAIN_CHANGED')
    def test_unsupported_or_partial_intent_never_forced_supported(self):
        for i,(supported,parts) in enumerate(((False,['backflip']),(True,['change Wave arm']))):
            data=response();data.update(supported=supported,unsupported_requests=parts);result=self.call(Provider(data),name=str(i));self.assertFalse(result['locally_supported']);self.assertEqual(result['unsupported_requests'],parts)
    def test_unknown_bool_huge_nan_and_surrogate_model_values_reject(self):
        values=[True,10**1000,float('nan'),'1','\ud800']
        for i,value in enumerate(values):
            data=response();data['expression']['second_left_pose_gain']=value
            with self.subTest(value=str(value)[:20]),self.assertRaises(ValueError):self.call(Provider(data),name=str(i))
            self.assertTrue((self.base/str(i)/'receipt.json').is_file())
        data=response();data['unexpected']=1
        with self.assertRaises(p.PlannerResponseRejected):self.call(Provider(data),name='extra')
    def test_invalid_request_music_descriptor_and_secret_metadata_zero_calls(self):
        for i,change in enumerate(('goals','music','descriptor','secret')):
            req=request();ctx=context();provider=Provider()
            if change=='goals':req['goals']['mean_pose_ratio']=True
            elif change=='music':ctx['music_summary']['sample_frames']=1
            elif change=='descriptor':ctx['source_descriptor']['sources']['wave']=None
            else:ctx['session_token']='NEVER_USE'
            with self.subTest(change=change),self.assertRaises(ValueError):p.plan_expression(provider,req,ctx,self.base/str(i))
            self.assertEqual(provider.calls,[])
    def test_provider_failure_unconfirmed_no_retry_and_exclusive_directory(self):
        provider=Provider(error=ProviderError('Codex failed; check local authentication/model access'))
        with self.assertRaises(ProviderError):self.call(provider)
        failed=json.loads((self.base/'call/failed_status.json').read_text());self.assertTrue(failed['completion_unconfirmed']);self.assertEqual(failed['error_category'],'CLI_NONZERO_EXIT_CAUSE_UNCONFIRMED')
        with self.assertRaises(FileExistsError):self.call(provider)
        self.assertEqual(len(provider.calls),1)
    def test_cancel_before_call_and_at_return_preserves_completed_receipt(self):
        event=threading.Event();event.set();provider=Provider()
        with self.assertRaises(p.PlannerCancelled):self.call(provider,cancel_event=event)
        self.assertEqual(provider.calls,[]);self.assertFalse((self.base/'call').exists());event.clear();old=provider.generate_json
        def at_return(*args,**kwargs):result=old(*args,**kwargs);event.set();return result
        provider.generate_json=at_return
        with self.assertRaises(p.PlannerCancelled):self.call(provider,cancel_event=event)
        self.assertFalse(json.loads((self.base/'call/receipt.json').read_text())['completion_unconfirmed']);self.assertFalse(json.loads((self.base/'call/failed_status.json').read_text())['completion_unconfirmed'])
    def test_wrong_model_heading_and_tool_receipt_rejected(self):
        provider=Provider();provider.model='other'
        with self.assertRaises(ValueError):self.call(provider)
        self.assertEqual(provider.calls,[])
        data=response();data['expression']['heading_policy']='recorded_world'
        with self.assertRaises(p.PlannerResponseRejected):self.call(Provider(data),name='heading')
        provider=Provider();original=provider.generate_json
        def tools(*a,**k):result=original(*a,**k);result.receipt['tool_events']=1;return result
        provider.generate_json=tools
        with self.assertRaises(p.PlannerResponseRejected):self.call(provider,name='tools')
    def test_receipt_live_and_elapsed_strict_no_truthiness_claim(self):
        cases=[('live','false'),('live',[]),('live',None),('elapsed_seconds',True),('elapsed_seconds',float('nan')),('elapsed_seconds',-1)]
        for i,(key,value) in enumerate(cases):
            provider=Provider();original=provider.generate_json
            def malformed(*a,_original=original,_key=key,_value=value,**kw):
                result=_original(*a,**kw);result.receipt[_key]=_value;return result
            provider.generate_json=malformed
            with self.subTest(key=key,value=value),self.assertRaises(p.PlannerResponseRejected):self.call(provider,name=str(i))
            self.assertTrue((self.base/str(i)/'response.json').exists());self.assertTrue((self.base/str(i)/'receipt_rejected.json').exists());self.assertFalse((self.base/str(i)/'validated_result.json').exists())
    def test_old_four_gain_or_different_condition_observations_reject(self):
        provider=Provider();ctx=context();ctx['observations'][0]['expression']={'schema':'pulsemorph.phrase_program.v1','first_left_gain':1.}
        with self.assertRaises(ValueError):p.plan_expression(provider,request(),ctx,self.base/'old4d')
        ctx=context();ctx['observations'][0]['expression']['first_left_pose_gain']=.9
        with self.assertRaises(ValueError):p.plan_expression(provider,request(),ctx,self.base/'condition')
        self.assertEqual(provider.calls,[])
    def test_old_gain_advice_cannot_hide_under_extra_constraints(self):
        ctx=context();ctx['constraints']={'retrieved_context':{'schema':'pulsemorph.conditional_gain_advice.v1','lower':.75,'upper':1.22}}
        provider=Provider()
        with self.assertRaises(ValueError):p.plan_expression(provider,request(),ctx,self.base/'old_advice')
        self.assertEqual(provider.calls,[])
if __name__=='__main__':unittest.main()
