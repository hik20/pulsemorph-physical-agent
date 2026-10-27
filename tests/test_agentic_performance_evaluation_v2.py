"""Prospective stage/relative-motion falsification using arrays only."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_agentic_performance_evaluation as fixtures
from pulsemorph.agentic.performance.evaluation_v1 import evaluate_performance as old
from pulsemorph.agentic.performance.evaluation_v2 import evaluate_performance
def run(d):return evaluate_performance(*d[:7],start_s=2.,physics_flags=d[7])
def translate(d,dx):
    for row in d[3]:
        delta=float(dx(max(0.,row['sim_time_s']-2.)))
        for p in row['positions_world_m'].values():p[0]+=delta
        row['com_world_m'][0]+=delta

class Tests(unittest.TestCase):
    def test_inplace_signal_pass_and_old_metrics_preserved(self):
        d=fixtures.fixture();before=old(*d[:7],start_s=2.,physics_flags=d[7]);r=run(d);self.assertTrue(r['performance_screen_passed']);self.assertTrue(r['legacy_performance_screen_passed']);self.assertEqual(r['actual_groups'],before['actual_groups']);self.assertEqual(r['hand_world_motion'],before['hand_world_motion']);self.assertEqual(r['music_alignment'],before['music_alignment']);self.assertTrue(r['stage_bound']['origin_available'])
    def test_translation_cannot_masquerade_as_relative_hand_expression(self):
        d=list(fixtures.fixture())
        for row in d[3]:
            pelvis=row['positions_world_m']['pelvis'];row['positions_world_m']['left_rubber_hand']=[pelvis[0],pelvis[1]+.3,pelvis[2]+.3];row['positions_world_m']['right_rubber_hand']=[pelvis[0],pelvis[1]-.3,pelvis[2]+.3]
        translate(d,lambda t:.2*t);r=run(d);self.assertGreater(r['hand_world_motion']['left']['temporal_diameter_m'],.2);self.assertLess(r['hand_pelvis_relative_motion']['left']['temporal_diameter_m'],1e-10);self.assertFalse(r['performance_screen_passed'])
    def test_excursion_fails_even_if_body_returns_to_start(self):
        d=list(fixtures.fixture());translate(d,lambda t:np.sin(np.pi*t/2));r=run(d);self.assertGreater(r['stage_bound']['max_planar_radius_from_start_m'],.75);self.assertLess(r['stage_bound']['end_planar_offset_from_start_m'],.35);self.assertFalse(r['performance_screen_passed']);self.assertTrue(r['legacy_performance_screen_passed'])
    def test_end_offset_fails_inside_radius(self):
        d=list(fixtures.fixture());translate(d,lambda t:.25*t);r=run(d);self.assertLess(r['stage_bound']['max_planar_radius_from_start_m'],.75);self.assertGreater(r['stage_bound']['end_planar_offset_from_start_m'],.35);self.assertFalse(r['performance_screen_passed'])
    def test_missing_exact_start_origin_not_silently_rebased(self):
        d=list(fixtures.fixture());keep=d[2]!=2.;d[0]=d[0][keep];d[1]=d[1][keep];d[3]=[row for i,row in enumerate(d[3]) if keep[i]];d[5]=d[5][keep];d[2]=d[2][keep];r=run(d);self.assertFalse(r['stage_bound']['origin_available']);self.assertIsNone(r['stage_bound']['max_planar_radius_from_start_m']);self.assertFalse(r['performance_screen_passed'])
    def test_partial_and_physical_failure_cannot_be_promoted(self):
        d=list(fixtures.fixture());d[7]['force_caps_passed']=False;r=run(d);self.assertFalse(r['performance_screen_passed']);self.assertTrue(r['stage_bound']['passed_so_far'])

if __name__=='__main__':unittest.main()
