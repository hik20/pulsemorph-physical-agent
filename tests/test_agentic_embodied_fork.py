"""Fork isolation/parity fixtures. Native toy is memory-copy only, no FK/step/NN."""
from pathlib import Path
import sys,copy,hashlib,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.runtime_v1 import PersistentEpisode,fingerprint,ActionRejected
from pulsemorph.agentic.embodied.fork_v1 import fork_episode,snapshot_episode,PreviewSession,PreviewIntegrityError
class Data:
 def __init__(self):self.qpos=np.r_[0.,0.,.8,1.,0.,0.,0.,np.zeros(29)];self.qvel=np.zeros(35);self.ctrl=np.zeros(29);self.qacc_warmstart=np.zeros(35);self.time=0.
 def __copy__(self):return copy.deepcopy(self)
class FixtureBackend:
 def __init__(self):
  self.data=Data();self.model=object();self.session=object();self.joint_names=['joint_%d'%i for i in range(29)];self.joint_ranges=[(-2.,2.)]*29;self.prev_targets=np.zeros(29,np.float32);self.initialization_count=1;self.control_index=0;self.step_count=0;self.forward_count=1;self.jacobian_calls=2;self.states=[self.data.qpos.copy()];self.velocities=[self.data.qvel.copy()];self.times=[0.];self.targets=[];self.integration_states=[self._integration()];self.substeps=[];self.inputs=[{'warm':np.zeros(1,np.float32)}];self.outputs=[{'warm':np.zeros(1,np.float32)}];self.calls=[{'role':'warmup'}];self.alignment={'offset':np.array([0.,0.,0.,1.])};self.stopped=False
 @property
 def sim_time_s(self):return self.data.time
 def _integration(self):return np.r_[self.data.time,self.data.qpos,self.data.qvel,self.data.ctrl,self.data.qacc_warmstart]
 def closure_sha256(self):return hashlib.sha256(self._integration().tobytes()+self.prev_targets.tobytes()).hexdigest()
 def observe(self):return {'sim_time_s':self.sim_time_s,'qpos':self.data.qpos.tolist(),'qvel':self.data.qvel.tolist(),'processed_history_sha256':hashlib.sha256(self.prev_targets.tobytes()).hexdigest(),'body_poses':{'right_rubber_hand':{'position_world_m':self.data.qpos[7:10].tolist(),'orientation_xyzw':[0.,0.,0.,1.]}},'contacts':[],'physical_flags':{'finite':True,'root_height_m':.8,'min_root_height_m':.8,'warnings_unchanged':True}}
 def advance(self,q,dq):
  self.inputs.append({'q':q.astype('float32').copy(),'history':self.prev_targets.copy()});self.data.qpos[7:]=q[0];self.data.qvel[6:]=dq[0];self.prev_targets=q[0].astype('float32').copy();self.data.ctrl[:]=self.prev_targets;self.data.time+=.02;self.control_index+=1;self.step_count+=20;self.forward_count+=1;self.calls.append({'role':'control'});self.outputs.append({'targets':self.prev_targets.copy()});self.states.append(self.data.qpos.copy());self.velocities.append(self.data.qvel.copy());self.times.append(self.data.time);self.targets.append(self.prev_targets.copy());self.integration_states.append(self._integration());self.substeps.append([self.step_count,self.sim_time_s])
 def export(self,p):pass
class ForkTests(unittest.TestCase):
 def setUp(self):self.backend=FixtureBackend();self.parent=PersistentEpisode(self.backend,'fixture',max_control_steps=100)
 def action(self,identity='move',target=.1):
  o=self.parent.observe();return {'action_id':identity,'expected_revision':o['revision'],'observation_sha256':fingerprint(o),'tool':'move_joints','arguments':{'duration_s':.6,'joint_targets_rad':{'joint_0':target}}}
 def test_copy_full_executor_state_without_alias_or_initialization(self):
  self.parent.execute(self.action('prelude',.03));before=snapshot_episode(self.parent);branch=fork_episode(self.parent,'hypothetical')
  self.assertEqual(snapshot_episode(branch),before);self.assertIs(branch.backend.model,self.backend.model);self.assertIs(branch.backend.session,self.backend.session);self.assertFalse(np.shares_memory(branch.backend.data.qpos,self.backend.data.qpos));self.assertFalse(np.shares_memory(branch.reference_q,self.parent.reference_q));self.assertEqual(branch.backend.initialization_count,1)
  branch.backend.prev_targets[0]=999;branch.actions[0]['status']='changed';branch.backend.alignment['offset'][0]=3
  self.assertEqual(snapshot_episode(self.parent),before);self.assertEqual(self.parent.actions[0]['status'],'completed')
 def test_preview_parent_unchanged_commit_exact_suffix(self):
  previews=PreviewSession(self.parent,max_total_preview_controls=60);cap=previews.capabilities();before=snapshot_episode(self.parent);p=previews.preview(self.action())
  self.assertEqual(snapshot_episode(self.parent),before);self.assertEqual(self.backend.control_index,0);self.assertEqual(p['suffix']['new_online_calls'],30);self.assertEqual(p['suffix']['new_initializations'],0)
  req={'preview_id':p['preview_id'],'expected_revision':0,'observation_sha256':before.observation_sha256,'commit_id':'commit1'};done=previews.commit(req)
  self.assertTrue(done['preview_actual_byte_equal']);self.assertEqual(done['observation'],p['hypothetical_observation']);self.assertEqual(self.backend.control_index,30);self.assertEqual(previews.capabilities(),cap);self.assertTrue(previews.commit(req)['replay_no_new_work']);self.assertEqual(self.backend.control_index,30)
 def test_stale_and_changed_commit_rejected_before_actual_work(self):
  previews=PreviewSession(self.parent);p=previews.preview(self.action());self.parent.execute(self.action('outside',.02));calls=self.backend.control_index
  with self.assertRaises(PreviewIntegrityError):previews.commit({'preview_id':p['preview_id'],'expected_revision':0,'observation_sha256':p['parent_observation_sha256'],'commit_id':'c'})
  self.assertEqual(self.backend.control_index,calls)
 def test_preview_budget_and_invalid_candidate_zero_parent_work(self):
  previews=PreviewSession(self.parent,max_preview_controls=30,max_total_preview_controls=30)
  with self.assertRaises(ActionRejected):previews.preview(self.action(target=10))
  self.assertEqual(previews.reserved_preview_controls,0);previews.preview(self.action())
  with self.assertRaises(PreviewIntegrityError):previews.preview(self.action('another'))
  self.assertEqual(self.backend.control_index,0)
 def test_parity_mutation_is_fatal_after_actual_no_replacement(self):
  previews=PreviewSession(self.parent);p=previews.preview(self.action());previews._previews[p['preview_id']]['branch'].backend.data.qacc_warmstart[0]=4
  with self.assertRaises(PreviewIntegrityError):previews.commit({'preview_id':p['preview_id'],'expected_revision':0,'observation_sha256':p['parent_observation_sha256'],'commit_id':'c'})
  self.assertTrue(previews.stopped);self.assertTrue(self.parent.stopped);self.assertEqual(self.backend.control_index,30)
class NativeMemoryCopyFixture(unittest.TestCase):
 def test_authorized_toy_one_model_two_data_no_steps_fk_nn(self):
  import mujoco
  model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j"/><geom type="sphere" size=".1"/></body></worldbody><actuator><motor joint="j"/></actuator></mujoco>');a=mujoco.MjData(model)
  a.time=2.;a.qpos[:]=.2;a.qvel[:]=.3;a.ctrl[:]=.4;a.qacc_warmstart[:]=.5
  b=copy.copy(a)
  for key in ('qpos','qvel','ctrl','qacc_warmstart'):
   self.assertFalse(np.shares_memory(getattr(a,key),getattr(b,key)));np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
  kind=mujoco.mjtState.mjSTATE_INTEGRATION;x=np.empty(mujoco.mj_stateSize(model,kind));y=np.empty_like(x);mujoco.mj_getState(model,a,x,kind);mujoco.mj_getState(model,b,y,kind);self.assertEqual(x.tobytes(),y.tobytes());b.qpos[0]=99;self.assertEqual(a.qpos[0],.2)
if __name__=='__main__':unittest.main()
