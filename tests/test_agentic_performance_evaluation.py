"""Pure-array tests; synthetic signals are not physical dance trials."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.performance.evaluation_v1 import evaluate_performance

def fixture():
    arms=[s+'_'+n+'_joint' for s in ('left','right') for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')];trunk=['waist_'+n+'_joint' for n in ('yaw','roll','pitch')];hips=[s+'_hip_'+n+'_joint' for s in ('left','right') for n in ('pitch','roll','yaw')];knees=[s+'_knee_joint' for s in ('left','right')];ankles=[s+'_ankle_'+n+'_joint' for s in ('left','right') for n in ('pitch','roll')];names=arms+trunk+hips+knees+ankles
    times=np.arange(201)*.02;wave=np.sin(4*np.pi*times);amps=np.asarray([.35]*14+[.05]*3+[.07]*6+[.1]*2+[.03]*4);q=np.zeros((201,36));q[:,3]=1;q[:,2]=.8;q[:,7:]=wave[:,None]*amps;v=np.zeros((201,35));v[:,6:]=(4*np.pi*np.cos(4*np.pi*times))[:,None]*amps;rows=[]
    for t in times:
        y=.02*np.sin(4*np.pi*t);z=.8+.02*np.cos(4*np.pi*t);positions={'pelvis':[0.,y,z],'torso_link':[0.,y,z+.3],**{side+'_rubber_hand':[.15*np.sin(4*np.pi*t),sign*.2,.9+.03*np.cos(4*np.pi*t)] for side,sign in [('left',1),('right',-1)]},**{side+'_ankle_roll_link':[.1*t,sign*.1,.05] for side,sign in [('left',1),('right',-1)]}}
        rows.append({'sim_time_s':float(t),'root_height_m':float(z),'root_up_z':.98,'com_world_m':[0.,float(y),float(z+.1)],'positions_world_m':{k:[float(x) for x in p] for k,p in positions.items()},'foot_contacts':{'left':not 2.9<=t<=3.1,'right':True},'ncon':2})
    music={'duration_s':2.,'beat_times_s':[i*.25 for i in range(9)]};flags={'finite':True,'warnings_unchanged':True,'stopped':False,'physical_gate_passed':True,'force_caps_passed':True}
    return q,v,times,rows,names,q[:,7:].copy(),music,flags
def run(data):return evaluate_performance(*data[:7],start_s=2.,physics_flags=data[7])

class Tests(unittest.TestCase):
    def test_all_fixed_minimums_and_observed_beat_alignment(self):
        d=fixture();r=run(d);self.assertTrue(r['demo_minimum_passed']);self.assertTrue(r['performance_screen_passed']);self.assertEqual(r['sample_count'],101);self.assertAlmostEqual(r['actual_groups']['arms']['span_rad'],.6986187099,8);self.assertEqual(set(r['actual_groups']['hips']['side_max_span_rad']),{'left','right'});self.assertLess(r['music_alignment']['mean_peak_to_nearest_beat_s'],.02)
    def test_reference_movement_cannot_substitute_actual_static_body(self):
        d=list(fixture());d[0]=d[0].copy();d[0][:,7:]=0;r=run(d);self.assertFalse(r['demo_minimum_passed']);self.assertGreater(r['reference']['groups']['arms']['span_rad'],.6);self.assertIn('arms',r['feedback']['below_minimums'])
    def test_static_distance_between_hands_is_not_workspace(self):
        d=list(fixture())
        for row in d[3]:row['positions_world_m']['left_rubber_hand']=[0.,.5,1.]
        r=run(d);self.assertEqual(r['hand_world_motion']['left']['temporal_diameter_m'],0.);self.assertIn('left_hand_workspace',r['feedback']['below_minimums']);self.assertFalse(r['demo_minimum_passed'])
    def test_partial_window_not_complete_or_relaxed(self):
        d=list(fixture());d[0]=d[0][:151];d[1]=d[1][:151];d[2]=d[2][:151];d[3]=d[3][:151];d[5]=d[5][:151];r=run(d);self.assertEqual(r['status'],'partial_window');self.assertFalse(r['demo_minimum_passed']);self.assertEqual(r['actual_coverage_s'],[2.,3.])
    def test_physical_screen_separate_and_stricter_than_backend_stop(self):
        d=list(fixture())
        for row in d[3]:row['root_up_z']=.89
        r=run(d);self.assertTrue(r['demo_minimum_passed']);self.assertFalse(r['physical_gate']['passed']);self.assertFalse(r['performance_screen_passed'])
    def test_contact_adjacent_proxy_uses_only_continuous_contact_pairs(self):
        r=run(fixture());left=r['contact_adjacent_foot_motion']['left'];right=r['contact_adjacent_foot_motion']['right'];self.assertLess(left['eligible_adjacent_pairs'],right['eligible_adjacent_pairs']);self.assertAlmostEqual(left['mean_speed_m_s'],.1);self.assertIn('not sole',left['qualification'])
    def test_nonfinite_layout_timestamp_and_reference_alignment_rejected(self):
        for change in ('nan','joint','time','reference'):
            d=list(fixture())
            if change=='nan':d[0][110,7]=np.nan
            elif change=='joint':d[4][0]='unknown_joint'
            elif change=='time':d[3][110]['sim_time_s']+=.001
            else:d[5]=d[5][1:]
            with self.assertRaises(ValueError):run(d)
    def test_reference_independent_clock_and_missing_physics_honest(self):
        d=list(fixture());d[5]={'times':d[2][::2],'q':d[5][::2]};r=evaluate_performance(*d[:7],start_s=2.);self.assertEqual(r['reference']['sample_count'],51);self.assertIsNone(r['physical_gate']['passed']);self.assertIsNone(r['performance_screen_passed'])
    def test_root_music_beats_s_contract_and_conflicting_alias_rejected(self):
        d=list(fixture());d[6]['beats_s']=d[6].pop('beat_times_s');d[6]['sha256']='a'*64;r=run(d);self.assertEqual(r['music_alignment']['beat_count_in_observed_window'],9);self.assertEqual(r['music_alignment']['music_sha256'],'a'*64);d[6]['beat_times_s']=[0.]
        with self.assertRaises(ValueError):run(d)

if __name__=='__main__':unittest.main()
