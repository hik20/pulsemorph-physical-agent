"""Synthetic sampled positions only, zero FK or controller execution."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
path=Path(__file__).resolve().parents[1]/'experiments/agentic/embodied/audit_goal_v1.py'
spec=importlib.util.spec_from_file_location('goal_trajectory_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class GoalTrajectoryTests(unittest.TestCase):
    def test_transient_inside_does_not_change_final_assessment(self):
        d,r=module.summarize_trajectory(np.array([0,.02,.04]),np.array([[.1,0,0],[0,0,0],[.05,0,0]]),[0,0,0],.012,0)
        self.assertEqual(r['first_in_tolerance_time_s'],.02);self.assertFalse(r['final_sample_inside_tolerance']);self.assertNotIn('task_verified',r)
    def test_prelude_excluded_and_nearest_frame_chosen(self):
        d,r=module.summarize_trajectory(np.array([0,.02,.04]),np.array([[0,0,0],[.1,0,0],[.05,0,0]]),[0,0,0],.012,.02)
        self.assertIsNone(r['first_in_tolerance_time_s']);self.assertEqual(r['sample_count'],2);self.assertEqual(r['minimum_time_s'],.04)
    def test_bad_clock_nonfinite_or_missing_active_samples_rejected(self):
        for times,positions,start in [([0,0],[[0,0,0],[0,0,0]],0),([0,.02],[[float('nan'),0,0],[0,0,0]],0),([0,.02],[[0,0,0],[0,0,0]],1)]:
            with self.assertRaises(ValueError):module.summarize_trajectory(times,positions,[0,0,0],.012,start)
if __name__=='__main__':unittest.main()
