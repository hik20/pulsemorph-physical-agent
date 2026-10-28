"""Pure injected fixtures only: no models, native construction, FK or physics."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_agentic_embodied_fork import FixtureBackend
from pulsemorph.agentic.embodied.runtime_v2 import PersistentEpisode,ActionRejected,fingerprint,ObservationMismatch
from pulsemorph.agentic.embodied.fork_v2 import PreviewSession,PreviewIntegrityError,snapshot_episode
from pulsemorph.agentic.embodied.preview_orchestrator_v2 import PreviewOrchestrator
from pulsemorph.agentic.embodied.orchestrator_v1 import CACHE
from pulsemorph.agentic.embodied.tasks_v1 import evaluate_task as evaluate_v1
from pulsemorph.agentic.embodied.tasks_v2 import evaluate_task
from pulsemorph.agentic.embodied.tasks_dual_v2 import evaluate_task as evaluate_dual

class Provider:
    model='gpt-6.1-sol'
    def __init__(self,decisions):self.decisions=iter(decisions);self.requests=[]
    def status(self):return {'enabled':True}
    def generate_json(self,prompt,schema,**kwargs):
        req=json.loads(prompt.split('REQUEST_JSON:\n')[1]);self.requests.append(req)
        decision,target=next(self.decisions);action=None;pid=None
        if decision=='preview':action={'tool':'move_joints','arguments':{'duration_s':.4,'joint_targets_rad':[{'joint_name':'joint_0','position_rad':target}]}}
        if decision=='commit':pid=[p['preview_id'] for p in req['previews'] if p['eligible']][-1]
        return SimpleNamespace(data={'decision':decision,'action':action,'preview_id':pid,'summary':'Fixture only','reason':'No live provider call'},receipt={'provider':'fixture','model_requested':self.model,'model_returned':None,'live':False,'tool_events':0,'elapsed_seconds':.01,'trust_boundary':'fixture only'})

def task(target=0.,hold=.2):return {'schema':'pulsemorph.embodied_wrist_task.v1','task_id':'fixed','body':'right_rubber_hand','frame':'world','target_world_m':[target,0.,0.],'tolerance_m':.012,'hold_s':hold,'max_sample_gap_s':.05,'min_root_height_m':.6}

class Tests(unittest.TestCase):
    def setUp(self):self.b=FixtureBackend();self.e=PersistentEpisode(self.b,'fixture',max_control_steps=100);self.f=PreviewSession(self.e)
    def action(self,target=0.,identity='candidate'):
        o=self.e.observe();return {'action_id':identity,'expected_revision':o['revision'],'observation_sha256':fingerprint(o),'tool':'move_joints','arguments':{'duration_s':.4,'joint_targets_rad':{'joint_0':target}}}
    def run_loop(self,decisions,t):
        CACHE.mkdir(parents=True,exist_ok=True);temp=tempfile.TemporaryDirectory(dir=CACHE);self.addCleanup(temp.cleanup);out=Path(temp.name)/'run';provider=Provider(decisions)
        result=PreviewOrchestrator(provider,task_verifier=evaluate_task,action_rejection_type=ActionRejected,allow_fixture=True).run('Reach and hold the fixed wrist target',t,self.f,out)
        return result,provider,out
    def test_compile_rejection_zero_work_complete_proof_and_no_reservation(self):
        before=snapshot_episode(self.e)
        with self.assertRaises(ActionRejected) as caught:self.f.preview(self.action(.35))
        r=caught.exception.report;self.assertEqual(r['physics_steps'],0);self.assertEqual(r['revision_unchanged'],0);self.assertEqual(r['observation_sha256'],before.observation_sha256);self.assertEqual(r['violations'][0]['joint_name'],'joint_0');self.assertAlmostEqual(r['violations'][0]['max_acceleration_rad_s2'],13.125);self.assertEqual(r['allowed_max_acceleration_rad_s2'],12.)
        self.assertEqual(snapshot_episode(self.e),before);self.assertEqual(self.f.reserved_preview_controls,0);self.assertEqual(self.b.control_index,0)
    def test_compile_feedback_reaches_next_model_request_then_revised_commit(self):
        result,provider,out=self.run_loop([('preview',.35),('preview',.1),('commit',None)],task(.1,0))
        self.assertEqual(result['status'],'completed');self.assertEqual(result['confirmed_sandbox_steps'],400);self.assertEqual(result['confirmed_actual_steps'],400)
        report=provider.requests[1]['history'][0]['report'];self.assertEqual(report['category'],'continuous_joint_constraint');self.assertEqual(report['physics_steps'],0);self.assertEqual(report['violations'][0]['joint_name'],'joint_0');self.assertEqual(provider.requests[1]['observation']['revision'],0)
    def test_positive_hold_uses_twenty_real_ordered_samples_and_keeps_action_revision(self):
        initial=self.e.observe();old=evaluate_v1(task(),initial);preview=self.f.preview(self.action());self.assertEqual(self.e.observe(),initial)
        done=self.f.commit({'preview_id':preview['preview_id'],'expected_revision':0,'observation_sha256':fingerprint(initial),'commit_id':'commit'})
        samples=self.f.committed_samples('commit');self.assertEqual(len(samples),20);self.assertEqual([s['control_index'] for s in samples],list(range(1,21)));self.assertEqual([s['revision'] for s in samples],[0]*19+[1]);self.assertEqual(done['after_revision'],1)
        self.assertFalse(evaluate_v1(task(),done['observation'],old['state'])['done'])
        state=evaluate_task(task(),initial)['state']
        for sample in samples:r=evaluate_task(task(),sample,state);state=r['state']
        self.assertTrue(r['done']);self.assertAlmostEqual(r['sampled_hold_s'],.4)
        again=self.f.commit({'preview_id':preview['preview_id'],'expected_revision':0,'observation_sha256':fingerprint(initial),'commit_id':'commit'});self.assertTrue(again['replay_no_new_work']);self.assertEqual(self.b.control_index,20)
        with self.assertRaises(ObservationMismatch):self.e.execute(self.action(identity='stale')|{'expected_revision':0})
    def test_orchestrator_positive_hold_is_actual_only_after_commit(self):
        result,provider,out=self.run_loop([('preview',0.),('commit',None)],task())
        self.assertEqual(result['status'],'completed');self.assertTrue(result['task_verified_actual']);self.assertEqual(provider.requests[1]['actual_assessment']['done'],False);self.assertTrue(provider.requests[1]['previews'][0]['hypothetical_assessment']['done']);self.assertTrue((out/'commit_00/sampled_observations.json').is_file())
    def test_hypothetical_hold_alone_does_not_satisfy_actual_task(self):
        result,provider,out=self.run_loop([('preview',0.),('finish',None)],task());self.assertFalse(result['task_verified_actual']);self.assertEqual(self.b.control_index,0)
    def test_gap_and_duplicate_changed_sample_do_not_gain_hold(self):
        initial=self.e.observe();prior=evaluate_task(task(),initial)['state'];p=self.f.preview(self.action());samples=self.f.preview_samples(p['preview_id']);late=evaluate_task(task(),samples[-1],prior);self.assertFalse(late['done']);self.assertEqual(late['sampled_hold_s'],0.)
        r=evaluate_task(task(),samples[0],prior);changed=copy.deepcopy(samples[0]);changed['body_poses']['right_rubber_hand']['position_world_m'][0]=.001
        with self.assertRaises(ValueError):evaluate_task(task(),changed,r['state'])
    def test_mutated_sample_journal_fails_closed_without_new_actual_work(self):
        p=self.f.preview(self.action());entry=self.f._previews[p['preview_id']];entry['branch']._sampled_receipts['candidate'][0]['sim_time_s']=999
        with self.assertRaises(PreviewIntegrityError):self.f.preview_samples(p['preview_id'])
        self.assertEqual(self.b.control_index,0)
    def test_mutated_sample_journal_prevents_commit_before_actual_work(self):
        p=self.f.preview(self.action());entry=self.f._previews[p['preview_id']];entry['branch']._sampled_receipts['candidate'][0]['sim_time_s']=999
        with self.assertRaises(PreviewIntegrityError):self.f.commit({'preview_id':p['preview_id'],'expected_revision':0,'observation_sha256':p['parent_observation_sha256'],'commit_id':'commit'})
        self.assertEqual(self.b.control_index,0)
    def test_dual_positive_hold_and_sample_identity(self):
        initial=self.e.observe();p=self.f.preview(self.action());samples=self.f.preview_samples(p['preview_id'])
        goals=[task(),task()];goals[1].update(task_id='left',body='left_rubber_hand')
        dual={'schema':'pulsemorph.embodied_dual_wrist_task.v1','task_id':'both','goals':goals};state=None
        for obs in [initial]+samples:
            obs=copy.deepcopy(obs);obs['body_poses']['left_rubber_hand']=copy.deepcopy(obs['body_poses']['right_rubber_hand'])
            report=evaluate_dual(dual,obs,state);state=report['state']
        self.assertTrue(report['done']);self.assertEqual(report['goals_passed'],2);self.assertAlmostEqual(report['sampled_hold_s'],.4)
    def test_budgets_stale_preview_and_cached_reads_remain_strict(self):
        self.f=PreviewSession(self.e,max_preview_controls=20,max_total_preview_controls=20);p=self.f.preview(self.action());self.assertEqual(self.e.observe(),self.e.observe());self.assertEqual(self.b.control_index,0)
        with self.assertRaises(PreviewIntegrityError):self.f.preview(self.action(identity='budget'))
        self.e.execute(self.action(identity='outside'))
        with self.assertRaises(PreviewIntegrityError):self.f.commit({'preview_id':p['preview_id'],'expected_revision':0,'observation_sha256':p['parent_observation_sha256'],'commit_id':'stale'})
        self.assertEqual(self.b.control_index,20)
    def test_large_full_journal_stays_out_of_model_context_and_receipt_limit(self):
        class LargeBackend(FixtureBackend):
            def observe(self):
                value=super().observe();value['body_poses']['right_rubber_hand']['fixture_padding']=[list(range(700)) for _ in range(10)];return value
        self.b=LargeBackend();self.e=PersistentEpisode(self.b,'fixture',max_control_steps=100);self.f=PreviewSession(self.e)
        result,provider,out=self.run_loop([('preview',0.),('commit',None)],task())
        self.assertEqual(result['status'],'completed');self.assertGreater((out/'commit_00/sampled_observations.json').stat().st_size,512*1024)
        self.assertLess(len(json.dumps(provider.requests[1]).encode()),256*1024)
    def test_orchestrator_rejects_missing_sample_instead_of_inventing_it(self):
        original=self.f.committed_samples
        def bad(commit_id):return original(commit_id)[1:]
        self.f.committed_samples=bad
        result,provider,out=self.run_loop([('preview',0.),('commit',None)],task());self.assertEqual(result['status'],'failed');self.assertFalse(result['task_verified_actual'])

if __name__=='__main__':unittest.main()
