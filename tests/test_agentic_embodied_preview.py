"""Sandbox/parent distinction using injected fake state; zero NN/FK/physics."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_agentic_embodied_orchestrator import Backend,verifier
from pulsemorph.agentic.embodied.orchestrator_v1 import CACHE,digest
from pulsemorph.agentic.embodied.preview_orchestrator_v1 import PreviewOrchestrator,projected_context

class Fork:
    def __init__(self):self.parent=Backend();self.previews={};self.sandbox_calls=0;self.parity=True;self.drift=False
    def capabilities(self):return self.parent.capabilities()
    def observe(self):return self.parent.observe()
    def preview(self,action):
        before=self.observe();branch=copy.deepcopy(self.parent);transition=branch.execute(action);self.sandbox_calls+=1;pid='fixture_preview_'+str(len(self.previews));row={'preview_id':pid,'parent_observation_sha256':digest(before),'parent_revision':before['revision'],'action':copy.deepcopy(action),'steps':transition['steps'],'hypothetical_observation':transition['observation'],'trajectory_sha256':'a'*64};self.previews[pid]=copy.deepcopy(row)
        if self.drift:self.parent.position+=.01
        return row
    def commit(self,value):
        row=self.previews[value['preview_id']];assert value['observation_sha256']==digest(self.observe());transition=self.parent.execute(row['action']);return {**transition,'preview_id':value['preview_id'],'preview_actual_byte_equal':self.parity}
class Provider:
    model='gpt-6.1-sol'
    def __init__(self,decisions):self.decisions=iter(decisions);self.requests=[]
    def status(self):return {'enabled':True}
    def generate_json(self,prompt,schema,**kwargs):
        req=json.loads(prompt.split('REQUEST_JSON:\n')[1]);self.requests.append(req);decision,value=next(self.decisions);action=None;pid=None
        if decision=='preview':action={'tool':'move_joints','arguments':{'duration_s':.02,'joint_targets_rad':[{'joint_name':'shoulder','position_rad':value}]}}
        if decision=='commit':pid=[p['preview_id'] for p in req['previews'] if p['eligible']][-1]
        data={'decision':decision,'action':action,'preview_id':pid,'summary':'Fixture sanal dalı incele','reason':'Gerçek sağlayıcı çağrısı değil'}
        return SimpleNamespace(data=data,receipt={'provider':'fixture','model_requested':self.model,'model_returned':None,'live':False,'tool_events':0,'elapsed_seconds':.01,'trust_boundary':'fixture only'})
class Tests(unittest.TestCase):
    def run_fixture(self,decisions,fork=None,**kwargs):
        temp=tempfile.TemporaryDirectory(dir=CACHE);self.addCleanup(temp.cleanup);out=Path(temp.name)/'run';fork=fork or Fork();provider=Provider(decisions);loop=PreviewOrchestrator(provider,task_verifier=verifier,allow_fixture=True,**kwargs);result=loop.run('Reach the fixed physical target',{'target':1.},fork,out);return result,fork,provider,out
    def test_two_model_authored_previews_reviewed_and_exact_commits(self):
        r,f,p,out=self.run_fixture([('preview',.5),('commit',None),('preview',1.),('commit',None)])
        self.assertEqual(r['status'],'completed');self.assertEqual(r['live_model_calls'],0);self.assertEqual(r['confirmed_sandbox_steps'],40);self.assertEqual(r['confirmed_actual_steps'],40);self.assertEqual(len(f.parent.executions),2)
        self.assertEqual(p.requests[1]['observation']['qpos'],[0.]);self.assertEqual(p.requests[1]['previews'][0]['hypothetical_observation']['qpos'],[.5]);self.assertIn('HYPOTHETICAL',p.requests[1]['previews'][0]['qualification']);self.assertEqual(p.requests[2]['observation']['qpos'],[.5])
        self.assertTrue((out/'commit_01/committed.json').exists())
    def test_hypothetical_success_is_not_actual_success(self):
        r,f,p,out=self.run_fixture([('preview',1.),('finish',None)]);self.assertFalse(r['task_verified_actual']);self.assertEqual(r['reason'],'unverified_finish');self.assertEqual(f.parent.position,0.);self.assertEqual(r['confirmed_actual_steps'],0)
    def test_preview_budget_no_automatic_search(self):
        r,f,p,out=self.run_fixture([('preview',.5),('preview',1.)],max_previews=1);self.assertEqual(f.sandbox_calls,1);self.assertEqual(r['reason'],'preview_budget');self.assertEqual(f.parent.executions,[])
    def test_parent_change_during_preview_fails_closed(self):
        f=Fork();f.drift=True;r,f,p,out=self.run_fixture([('preview',.5)],f);self.assertEqual(r['status'],'failed');self.assertEqual(r['completion_unconfirmed'],'preview');self.assertEqual(f.parent.executions,[])
    def test_commit_parity_failure_stops_unknown_no_retry(self):
        f=Fork();f.parity=False;r,f,p,out=self.run_fixture([('preview',1.),('commit',None)],f);self.assertEqual(r['status'],'failed');self.assertEqual(r['completion_unconfirmed'],'commit');self.assertEqual(len(p.requests),2);self.assertEqual(len(f.parent.executions),1)
    def test_separate_actual_step_budget_blocks_commit(self):
        r,f,p,out=self.run_fixture([('preview',1.),('commit',None)],max_actual_steps=10);self.assertEqual(r['reason'],'actual_step_budget');self.assertEqual(r['confirmed_sandbox_steps'],20);self.assertEqual(f.parent.executions,[])
    def test_confirmed_failed_sandbox_can_inform_next_model_without_parent_work(self):
        class Failed(RuntimeError):
            def __init__(self,report):self.report=report
        class FailOnce(Fork):
            first=True
            def preview(self,action):
                if self.first:
                    self.first=False;obs=self.observe();raise Failed({'schema':'fixture','parent_unchanged':True,'parent_observation_sha256':digest(obs),'parent_revision':obs['revision'],'hypothetical_only':True,'suffix':{'new_physics_steps':10},'error_type':'fixture physical stop'})
                return super().preview(action)
        r,f,p,out=self.run_fixture([('preview',.5),('preview',1.),('commit',None)],FailOnce(),preview_physical_failure_type=Failed)
        self.assertEqual(r['status'],'completed');self.assertEqual(r['confirmed_sandbox_steps'],30);self.assertEqual(r['confirmed_actual_steps'],20);self.assertEqual(p.requests[1]['history'][0]['kind'],'hypothetical_physical_failure');self.assertEqual(p.requests[1]['observation']['qpos'],[0.]);self.assertTrue((out/'preview_00/physical_failure.json').exists())
    def test_max_history_projection_preserves_raw_data_and_stays_bounded(self):
        obs=Backend().observe();obs['body_poses']={('body_'+str(i)):{'aux':list(range(200))} for i in range(50)};assessment={'done':False,'reason':'fixture','distance_m':.1,'state':{'large':list(range(1000))}}
        previews={};history=[]
        for i in range(8):
            row={'preview_id':'p'+str(i),'parent_observation_sha256':digest(obs),'action':{'tool':'move_joints','arguments':{'duration_s':.4,'joint_targets_rad':{'shoulder':.1}}},'hypothetical_observation':copy.deepcopy(obs),'hypothetical_assessment':copy.deepcopy(assessment),'eligible':True,'qualification':'HYPOTHETICAL'};previews[row['preview_id']]=row;history.append({'kind':'hypothetical_preview','preview':copy.deepcopy(row)});history.append({'kind':'actual_committed','preview_id':row['preview_id'],'transition':{'observation':copy.deepcopy(obs)},'actual_assessment':copy.deepcopy(assessment)})
        raw_before=digest({'history':history,'previews':previews});projected=projected_context({'fixture':True},obs,assessment,history,previews,{})
        self.assertGreater(len(json.dumps({'history':history,'previews':previews})),256*1024);self.assertLess(len(json.dumps(projected).encode()),256*1024);self.assertEqual(raw_before,digest({'history':history,'previews':previews}));self.assertTrue(projected['context_projection']['raw_observations_and_receipts_preserved']);self.assertEqual(projected['previews'][0]['action'],previews['p0']['action'])

if __name__=='__main__':unittest.main()
