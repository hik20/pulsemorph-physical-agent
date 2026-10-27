"""Two-phase replay selection only, no robot render or FK."""
import ast
import importlib.util
from pathlib import Path
import unittest
import numpy as np
p=Path(__file__).resolve().parents[1]/'experiments/agentic/embodied/render_episode_v3.py'
spec=importlib.util.spec_from_file_location('two_goal_render_fixture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TwoGoalRenderTests(unittest.TestCase):
    def test_actual_3p2_second_frame_budget(self):self.assertEqual(len(m.frames_for(np.arange(161)*.02)),80)
    def test_phase_boundary_belongs_to_completed_previous_action(self):
        a=[{'start_sim_time_s':2,'planned_control_steps':30},{'start_sim_time_s':2.6,'planned_control_steps':30}]
        self.assertIs(m.active_action(a,2.6),a[0]);self.assertIs(m.active_action(a,2.62),a[1])
    def test_one_fk_two_cameras_and_actual_only_sources(self):
        s=p.read_text();t=ast.parse(s);calls=[n.func.attr for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(calls.count('mj_forward'),1);self.assertNotIn('mj_step',calls);self.assertNotIn('InferenceSession',calls)
        self.assertIn('task_verified_actual',s);self.assertIn('phase_00_right',s);self.assertIn('phase_01_both',s);self.assertNotIn("execution/'previews'",s)
if __name__=='__main__':unittest.main()
