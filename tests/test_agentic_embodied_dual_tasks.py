"""Two fixed world goals on pure synthetic observations; zero runtime work."""
import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.tasks_dual_v1 import SCHEMA,validate_task,evaluate_task

def goal(body):return {'schema':'pulsemorph.embodied_wrist_task.v1','task_id':body,'body':body,'frame':'world','target_world_m':[.16,-.14 if body.startswith('right') else .14,.61],'tolerance_m':.012,'hold_s':0.,'max_sample_gap_s':.05,'min_root_height_m':.6}
def task():return {'schema':SCHEMA,'task_id':'fixed_dual','goals':[goal('right_rubber_hand'),goal('left_rubber_hand')]}
def obs():return {'schema':'pulsemorph.embodied_observation.v1','session_id':'same_body','revision':2,'sim_time_s':2.,'body_poses':{g['body']:{'position_world_m':g['target_world_m'][:]} for g in task()['goals']},'physical_flags':{'finite':True,'warnings_unchanged':True,'stopped':False,'root_height_m':.8,'min_root_height_m':.79}}

class Tests(unittest.TestCase):
    def test_both_inside_same_actual_observation(self):
        result=evaluate_task(task(),obs());self.assertTrue(result['done']);self.assertFalse(result['failed']);self.assertEqual(result['goals_passed'],2);self.assertEqual(result['distance_m'],0.)
    def test_one_wrist_not_enough_and_compact_reason_preserves_both(self):
        o=obs();o['body_poses']['left_rubber_hand']['position_world_m'][0]+=.03;r=evaluate_task(task(),o);self.assertFalse(r['done']);self.assertEqual(r['goals_passed'],1);self.assertAlmostEqual(r['distance_m'],.03);self.assertAlmostEqual(r['signed_error_world_m'][0],-.03);self.assertIn('right_rubber_hand',r['reason']);self.assertIn('left_rubber_hand',r['reason']);self.assertIn('30.000000mm',r['reason'])
    def test_right_drift_after_left_reach_is_not_success(self):
        o=obs();o['body_poses']['left_rubber_hand']['position_world_m'][0]+=.03;first=evaluate_task(task(),o);n=obs();n['revision']=3;n['sim_time_s']=2.8;n['body_poses']['right_rubber_hand']['position_world_m'][0]+=.02;r=evaluate_task(task(),n,first['state']);self.assertFalse(r['done']);self.assertEqual(r['goals_passed'],1)
    def test_repeated_observation_idempotent_and_inputs_unchanged(self):
        t,o=task(),obs();original=copy.deepcopy((t,o));a=evaluate_task(t,o);b=evaluate_task(t,o,a['state']);self.assertEqual(a,b);self.assertEqual((t,o),original)
    def test_goal_session_same_revision_and_clock_changes_fail(self):
        t,o=task(),obs();a=evaluate_task(t,o)
        for change in ('goal','session','sameobs','clock'):
            tt,oo=copy.deepcopy(t),copy.deepcopy(o)
            if change=='goal':tt['goals'][0]['target_world_m'][0]+=.001
            elif change=='session':oo['session_id']='other'
            elif change=='sameobs':oo['body_poses']['left_rubber_hand']['position_world_m'][0]+=.001
            else:oo.update(revision=3,sim_time_s=1.)
            with self.assertRaises(ValueError):evaluate_task(tt,oo,a['state'])
    def test_physical_failure_overrides_both_reached(self):
        o=obs();o['physical_flags']['stopped']=True;r=evaluate_task(task(),o);self.assertTrue(r['failed']);self.assertFalse(r['done']);self.assertFalse(r['physical_gate_passed'])
    def test_wrapper_state_cannot_disagree_with_goal_states(self):
        r=evaluate_task(task(),obs());state=copy.deepcopy(r['state']);state['revision']+=1
        with self.assertRaises(ValueError):evaluate_task(task(),obs(),state)
    def test_strict_schema_duplicate_body_and_nonfinite(self):
        for mutation in ('unknown','duplicate','bool','nan'):
            t=task()
            if mutation=='unknown':t['extra']=1
            elif mutation=='duplicate':t['goals'][1]=copy.deepcopy(t['goals'][0])
            elif mutation=='bool':t['goals'][0]['tolerance_m']=True
            else:t['goals'][0]['target_world_m'][0]=float('nan')
            with self.assertRaises(ValueError):validate_task(t)

if __name__=='__main__':unittest.main()
