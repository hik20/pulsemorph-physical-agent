"""Synthetic world-wrist observations only; no FK, model or physics."""
import copy
import unittest
from pulsemorph.agentic.embodied.tasks_v1 import validate_task, evaluate_task

def task(hold=0):
    return {'schema':'pulsemorph.embodied_wrist_task.v1','task_id':'fixture',
        'body':'right_rubber_hand','frame':'world','target_world_m':[.2,-.2,1.],
        'tolerance_m':.02,'hold_s':hold,'max_sample_gap_s':.02,'min_root_height_m':.6}

def obs(rev=0,time=0,position=None):
    return {'schema':'pulsemorph.embodied_observation.v1','session_id':'fixture-session',
        'revision':rev,'sim_time_s':time,'body_poses':{'right_rubber_hand':{
        'position_world_m':position or [.2,-.2,1.]}},
        'physical_flags':{'finite':True,'warnings_unchanged':True,'stopped':False,
        'root_height_m':.8,'min_root_height_m':.79}}

class EmbodiedTaskTests(unittest.TestCase):
    def test_actual_world_reach_signed_error(self):
        a=evaluate_task(task(),obs(position=[.1,-.2,1.]))
        self.assertFalse(a['done']);self.assertAlmostEqual(a['signed_error_world_m'][0],.1)
        b=evaluate_task(task(),obs(1,.6),a['state']);self.assertTrue(b['done'])
    def test_no_reference_or_model_success_substitution(self):
        o=obs(position=[0,0,0]);o['claimed_success']=True;o['reference_wrist']=[.2,-.2,1.]
        self.assertFalse(evaluate_task(task(),o)['done'])
        del o['body_poses']['right_rubber_hand']
        with self.assertRaises(ValueError):evaluate_task(task(),o)
    def test_physical_failure_is_not_completion(self):
        for k,value in [('finite',False),('warnings_unchanged',False),('stopped',True),('root_height_m',.59),('min_root_height_m',.59)]:
            o=obs();o['physical_flags'][k]=value;r=evaluate_task(task(),o)
            self.assertFalse(r['done']);self.assertTrue(r['failed'])
    def test_hold_counts_simulation_samples_not_reads(self):
        t=task(.04);r=evaluate_task(t,obs())
        for _ in range(4):
            repeated=evaluate_task(t,obs(),r['state']);self.assertEqual(repeated,r);r=repeated
        r=evaluate_task(t,obs(1,.02),r['state']);self.assertFalse(r['done'])
        r=evaluate_task(t,obs(2,.04),r['state']);self.assertTrue(r['done'])
    def test_no_hold_credit_across_unobserved_chunk(self):
        t=task(.04);a=evaluate_task(t,obs());b=evaluate_task(t,obs(1,.6),a['state'])
        self.assertFalse(b['done']);self.assertEqual(b['sampled_hold_s'],0)
    def test_outside_sample_resets_hold(self):
        t=task(.04);r=evaluate_task(t,obs());r=evaluate_task(t,obs(1,.02,[0,0,0]),r['state'])
        r=evaluate_task(t,obs(2,.04),r['state']);self.assertFalse(r['done']);self.assertEqual(r['sampled_hold_s'],0)
    def test_fixed_task_session_and_monotonic_observation(self):
        r=evaluate_task(task(),obs(2,.2))
        changed=task();changed['target_world_m'][0]=.3
        with self.assertRaises(ValueError):evaluate_task(changed,obs(3,.3),r['state'])
        for o in [obs(1,.3),obs(3,.1),obs(2,.2,[0,0,0])]:
            with self.assertRaises(ValueError):evaluate_task(task(),o,r['state'])
        o=obs(3,.3);o['session_id']='different'
        with self.assertRaises(ValueError):evaluate_task(task(),o,r['state'])
    def test_exact_dto_finite_types_and_frame(self):
        bad=[('hold_s',True),('tolerance_m',0),('max_sample_gap_s',.6),('min_root_height_m',.5),('frame','torso'),('target_world_m',[float('nan'),0,0])]
        for k,v in bad:
            t=task();t[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):validate_task(t)
        t=task();t['model_can_revise']=True
        with self.assertRaises(ValueError):validate_task(t)
        o=obs();o['physical_flags']['finite']=1
        with self.assertRaises(ValueError):evaluate_task(task(),o)
    def test_input_objects_unchanged_and_output_json_finite(self):
        import json
        t=task();o=obs();before=copy.deepcopy((t,o));r=evaluate_task(t,o)
        self.assertEqual((t,o),before);json.dumps(r,allow_nan=False)

if __name__=='__main__':unittest.main()
