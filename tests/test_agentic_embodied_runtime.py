"""Pure injected state-machine fixtures, NOT physical trials."""
from pathlib import Path
import sys,copy,unittest,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.runtime_v1 import PersistentEpisode,JointSegment,ActionRejected,ObservationMismatch,fingerprint

class FixtureBackend:
 def __init__(self):
  self.joint_names=['joint_%d'%i for i in range(29)];self.joint_ranges=[(-2.,2.)]*29;self.sim_time_s=0.;self.initialization_count=1;self.calls=0;self.prev=np.zeros(29);self.q=np.r_[0.,0.,.8,1.,0.,0.,0.,np.zeros(29)];self.fail_at=None
 def observe(self):return {'sim_time_s':self.sim_time_s,'qpos':self.q.tolist(),'qvel':np.zeros(35).tolist(),'processed_history_sha256':hashlib.sha256(self.prev.tobytes()).hexdigest(),'body_poses':{},'contacts':[],'physical_flags':{'finite':True,'root_height_m':.8,'min_root_height_m':.8,'warnings_unchanged':True}}
 def advance(self,q,dq):
  self.calls+=1
  if self.fail_at==self.calls:raise RuntimeError('fixture partial failure')
  self.sim_time_s+=.02;self.q[7:]=q[0];self.prev=q[0].copy()
 def export(self,directory):pass

class EpisodeTests(unittest.TestCase):
 def setUp(self):self.b=FixtureBackend();self.e=PersistentEpisode(self.b,'fixture',max_control_steps=60)
 def action(self,identity='a',target=.1,duration=.6):
  o=self.e.observe();return {'action_id':identity,'expected_revision':o['revision'],'observation_sha256':fingerprint(o),'tool':'move_joints','arguments':{'duration_s':duration,'joint_targets_rad':{'joint_0':target}}}
 def test_two_chunks_persist_state_clock_history_and_no_read_work(self):
  cap=self.e.capabilities();initial=self.e.observe();self.assertEqual(initial,self.e.observe());self.assertEqual(self.b.calls,0)
  self.e.execute(self.action());middle=self.e.observe();self.assertEqual(middle['revision'],1);self.assertEqual(middle['control_index'],30);self.assertEqual(middle['remaining_control_steps'],30);self.assertAlmostEqual(middle['sim_time_s'],.6)
  self.e.execute(self.action('b',-.05));last=self.e.observe();self.assertEqual(last['revision'],2);self.assertEqual(last['control_index'],60);self.assertAlmostEqual(last['sim_time_s'],1.2);self.assertEqual(self.b.initialization_count,1);self.assertEqual(self.e.capabilities(),cap);self.assertNotEqual(initial['processed_history_sha256'],middle['processed_history_sha256'])
 def test_stale_fatal_duplicate_idempotent(self):
  action=self.action();first=self.e.execute(action);calls=self.b.calls
  self.assertTrue(self.e.execute(copy.deepcopy(action))['replay_no_new_work']);self.assertEqual(self.b.calls,calls)
  with self.assertRaises(ObservationMismatch):self.e.execute({**action,'arguments':{'duration_s':.6,'joint_targets_rad':{'joint_0':.12}}})
  with self.assertRaises(ObservationMismatch):self.e.execute({**action,'action_id':'new'})
 def test_compile_rejection_is_zero_work_with_report(self):
  before=self.e.observe()
  with self.assertRaises(ActionRejected) as caught:self.e.execute(self.action(target=.35,duration=.4))
  self.assertEqual(self.e.observe(),before);self.assertEqual(self.b.calls,0);report=caught.exception.report;self.assertEqual(report['physics_steps'],0);self.assertTrue(report['violations']);self.assertEqual(report['violations'][0]['joint_name'],'joint_0')
  self.e.execute(self.action('valid',.1));self.assertEqual(self.e.revision,1)
 def test_invalid_bool_huge_unknown_and_budget(self):
  for target in (True,10**1000,float('nan')):
   with self.assertRaises(ActionRejected):self.e.execute(self.action(target=target))
  action=self.action();action['arguments']['joint_targets_rad']={'missing':0}
  with self.assertRaises(ActionRejected):self.e.execute(action)
  self.e.execute(self.action());self.e.execute(self.action('b'))
  with self.assertRaises(ActionRejected):self.e.execute(self.action('c'))
 def test_partial_failure_latches_no_restart(self):
  self.b.fail_at=3
  with self.assertRaises(RuntimeError):self.e.execute(self.action())
  self.assertTrue(self.e.stopped);self.assertEqual(self.e.control_index,2);self.assertEqual(self.e.actions[0]['status'],'failed_partial')
  with self.assertRaises(RuntimeError):self.e.execute(self.action('b'))
  self.assertEqual(self.b.calls,3)
 def test_real_shared_reference_live_contract_no_physics(self):
  from pulsemorph.motion.g1 import profile_from_yaml,assemble_reference_inputs
  from pulsemorph.agentic.embodied.reference_v1 import make_reference
  profile=profile_from_yaml();backend=FixtureBackend();backend.profile=profile;backend.joint_names=list(profile['joint_names']);backend.source_pins={'synthetic-fixture-not-physics':'a'*64};backend.anchor_target=np.array([0.,0.,0.,1.]);backend.closure_sha256=lambda:'b'*64
  q=np.zeros(29);target=q.copy();target[18]=.05;segment=JointSegment(q,q,target,.6,[(-2,2)]*29)
  ref=make_reference(backend,segment,'fixture','action',0,1.2,'c'*64,q,q)
  ref.validate();self.assertEqual(ref.metadata['observation']['mode'],'live_causal');self.assertEqual(ref.metadata['validity']['start_time_s'],1.2);self.assertAlmostEqual(ref.metadata['validity']['end_time_s'],1.96)
  self.assertEqual(ref.metadata['producer']['id'],'agentic-embodied-joint-tool')
  for i in range(30):self.assertEqual(assemble_reference_inputs(ref,i*.02)['mimic_future_dof_pos'].shape,(1,4,29))
 def test_continuous_segment_derivative_and_tail(self):
  q=np.zeros(29);goal=q.copy();goal[0]=.1;s=JointSegment(q,q,goal,.6,[(-2,2)]*29)
  t=.23;eps=1e-6;finite=(s.sample(t+eps)[0]-s.sample(t-eps)[0])/(2*eps);np.testing.assert_allclose(s.sample(t)[1],finite,atol=1e-9)
  np.testing.assert_array_equal(s.sample(.76)[0],goal);np.testing.assert_array_equal(s.sample(.76)[1],q)

if __name__=='__main__':unittest.main()
