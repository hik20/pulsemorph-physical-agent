"""Persistent fake state, real durable DTOs; no NN, physics, FK or remote calls."""
import copy,json,sys,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.orchestrator_v1 import EmbodiedOrchestrator,CACHE,digest

class Backend:
    def __init__(self):self.revision=0;self.position=0.;self.time=0.;self.executions=[];self.raise_after=False;self.drift=False;self.reads=0
    def capabilities(self):return {'schema':'fixture','physics_step_s':.001,'tools':{'move_joints':{'arguments_schema':{'type':'object','additionalProperties':False,'properties':{'duration_s':{'type':'number','minimum':.02,'maximum':.1},'joint_targets_rad':{'type':'object','additionalProperties':False,'properties':{'shoulder':{'type':'number','minimum':-1,'maximum':1}},'required':['shoulder']}},'required':['duration_s','joint_targets_rad']}}}}
    def observe(self):
        self.reads+=1
        return {'schema':'fixture','session_id':'persistent_fixture','revision':self.revision,'sim_time_s':self.time,'qpos':[self.position+(0.01 if self.drift and self.reads>1 else 0)],'qvel':[0.],'body_poses':{},'contacts':[],'physical_flags':{'finite':True}}
    def execute(self,action):
        before=self.observe();assert action['expected_revision']==self.revision and action['observation_sha256']==digest(before)
        self.executions.append(copy.deepcopy(action));old=self.revision;self.revision+=1;self.time+=action['arguments']['duration_s'];self.position=action['arguments']['joint_targets_rad']['shoulder']
        if self.raise_after:raise RuntimeError('fixture unknown after state advanced')
        return {'action_id':action['action_id'],'before_revision':old,'after_revision':self.revision,'before_observation_sha256':digest(before),'steps':round(action['arguments']['duration_s']/.001),'observation':self.observe()}
class Provider:
    model='gpt-6.1-sol'
    def __init__(self,decisions=None):self.requests=[];self.decisions=iter(decisions or [.5,1.])
    def status(self):return {'enabled':True}
    def generate_json(self,prompt,schema,**kwargs):
        self.requests.append(json.loads(prompt.split('REQUEST_JSON:\n')[1]));value=next(self.decisions)
        response={'decision':value if type(value) is str else 'act','action':None if type(value) is str else {'tool':'move_joints','arguments':{'duration_s':.02,'joint_targets_rad':[{'joint_name':'shoulder','position_rad':value}]}},'summary':'Yerel fixture önerisi','reason':'Gerçek çağrı değildir'}
        return SimpleNamespace(data=response,receipt={'provider':'fixture','model_requested':self.model,'model_returned':None,'live':False,'tool_events':0,'elapsed_seconds':.01,'trust_boundary':'fixture only'})
def verifier(task,obs,state):return {'done':obs['qpos'][0]>=task['target'],'reason':'Actual fixture position','state':{'last_revision':obs['revision']}}

class Tests(unittest.TestCase):
    def run_fixture(self,backend=None,provider=None,**kwargs):
        backend=backend or Backend();provider=provider or Provider();self.temp=tempfile.TemporaryDirectory(dir=CACHE);self.addCleanup(self.temp.cleanup)
        loop=EmbodiedOrchestrator(provider,task_verifier=verifier,allow_fixture=kwargs.pop('allow_fixture',True),**kwargs);out=Path(self.temp.name)/'run'
        result=loop.run('Move shoulder to the fixed target',{'target':1.},backend,out)
        return result,backend,provider,out
    def test_observe_act_revise_same_persistent_state_and_fixed_task(self):
        r,b,p,out=self.run_fixture();self.assertEqual(r['status'],'completed');self.assertTrue(r['fixture_run']);self.assertEqual(r['live_model_calls'],0);self.assertEqual(r['committed_actions'],2);self.assertEqual(r['confirmed_physics_steps'],40)
        self.assertEqual(p.requests[1]['observation']['revision'],1);self.assertEqual(p.requests[1]['observation']['qpos'],[.5]);self.assertEqual(len(p.requests[1]['history']),1);self.assertEqual(p.requests[0]['fixed']['task'],p.requests[1]['fixed']['task']);self.assertTrue((out/'action_01/committed.json').exists());self.assertEqual(len(list(out.glob('assessment_*.json'))),3)
    def test_stale_state_before_execute_stops_zero_actions(self):
        b=Backend();b.drift=True;r,b,p,out=self.run_fixture(b);self.assertEqual(r['status'],'failed');self.assertEqual(b.executions,[])
    def test_unknown_action_completion_durable_no_retry(self):
        b=Backend();b.raise_after=True;r,b,p,out=self.run_fixture(b);self.assertEqual(r['status'],'failed');self.assertTrue(r['action_completion_unconfirmed']);self.assertEqual(len(b.executions),1);self.assertEqual(len(p.requests),1);self.assertTrue((out/'action_00/pending.json').exists());self.assertFalse((out/'action_00/committed.json').exists())
    def test_model_finish_cannot_claim_task_success(self):
        r,b,p,out=self.run_fixture(provider=Provider(['finish']));self.assertFalse(r['task_verified']);self.assertEqual(r['reason'],'model_finish_unverified');self.assertEqual(b.executions,[])
    def test_step_budget_rejects_before_execution(self):
        r,b,p,out=self.run_fixture(max_physics_steps=10);self.assertEqual(r['status'],'failed');self.assertEqual(b.executions,[]);self.assertEqual(r['confirmed_physics_steps'],0)
    def test_nonlive_receipt_denied_in_production(self):
        r,b,p,out=self.run_fixture(allow_fixture=False);self.assertEqual(r['status'],'failed');self.assertEqual(r['live_model_calls'],0);self.assertEqual(b.executions,[])
    def test_finite_model_budget_stops_without_hidden_calls(self):
        r,b,p,out=self.run_fixture(provider=Provider([.5,.5]),max_model_calls=2);self.assertEqual(r['status'],'stopped');self.assertEqual(len(p.requests),2);self.assertEqual(len(b.executions),2);self.assertFalse(r['task_verified'])
    def test_goal_reached_early_does_not_force_fake_extra_action(self):
        r,b,p,out=self.run_fixture(provider=Provider([1.]),min_actions=2);self.assertEqual(r['status'],'stopped');self.assertEqual(r['reason'],'minimum_action_witness_unmet');self.assertTrue(r['task_verified']);self.assertFalse(r['minimum_actions_met']);self.assertEqual(len(b.executions),1)
    def test_proven_zero_work_rejection_can_revise_same_state(self):
        class Rejected(ValueError):
            def __init__(self,report):self.report=report
        class RejectOnce(Backend):
            rejected=False
            def execute(self,action):
                if not self.rejected:
                    self.rejected=True;raise Rejected({'physics_steps':0,'revision_unchanged':self.revision,'observation_sha256':digest(self.observe()),'category':'fixture continuous limit'})
                return super().execute(action)
        r,b,p,out=self.run_fixture(RejectOnce(),Provider([.5,.5,1.]),action_rejection_type=Rejected)
        self.assertEqual(r['status'],'completed');self.assertEqual(r['rejected_actions'],1);self.assertEqual(r['action_attempts'],3);self.assertEqual(r['committed_actions'],2);self.assertEqual(p.requests[1]['observation']['revision'],0);self.assertEqual(p.requests[1]['history'][0]['rejection_report']['physics_steps'],0);self.assertTrue((out/'action_00/rejected.json').exists())
    def test_typed_rejection_after_state_change_is_fatal(self):
        class Rejected(ValueError):
            def __init__(self):self.report={'physics_steps':0}
        class Bad(Backend):
            def execute(self,action):self.revision+=1;raise Rejected()
        r,b,p,out=self.run_fixture(Bad(),Provider([.5]),action_rejection_type=Rejected)
        self.assertEqual(r['status'],'failed');self.assertTrue(r['action_completion_unconfirmed']);self.assertEqual(len(p.requests),1)
    def test_actual_capabilities_method_offline_transport_audit(self):
        from pulsemorph.agentic.embodied.runtime_v1 import PersistentEpisode
        from pulsemorph.agentic.embodied.orchestrator_v1 import response_schema
        from pulsemorph.agentic.phrase_planner import audit_provider_schema
        episode=PersistentEpisode.__new__(PersistentEpisode);episode.backend=SimpleNamespace(joint_names=['joint_'+str(i) for i in range(29)],joint_ranges=[(-1.,1.)]*29);episode.max_control_steps=60
        schema=response_schema(episode.capabilities());audit_provider_schema(schema);action=copy.deepcopy(schema['properties']['action']);action['type']='object';audit_provider_schema(action)
        target=action['properties']['arguments']['properties']['joint_targets_rad'];self.assertEqual(target['maxItems'],29);self.assertEqual(target['items']['required'],['joint_name','position_rad']);self.assertFalse(target['items']['additionalProperties']);self.assertEqual(len(target['items']['properties']['joint_name']['enum']),29)

if __name__=='__main__':unittest.main()
