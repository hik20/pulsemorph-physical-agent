"""Pure replay scheduling/authority tests, no renderer or FK execution."""
import ast
import importlib.util
from pathlib import Path
import unittest
import numpy as np
path=Path(__file__).resolve().parents[1]/'experiments/agentic/embodied/render_episode_v1.py'
spec=importlib.util.spec_from_file_location('embodied_render_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class EmbodiedRenderTests(unittest.TestCase):
    def test_5p6_seconds_140_frames_one_fk_per_frame(self):
        frames=module.frames_for(np.arange(281)*.02)
        self.assertEqual(len(frames),140);self.assertEqual(frames[0],0);self.assertEqual(frames[-1],278)
    def test_reset_clock_and_odd_half_frame_rejected(self):
        for times in [np.array([0,.02,0]),np.arange(4)*.02,np.array([0,.02,float('nan')])]:
            with self.assertRaises(ValueError):module.frames_for(times)
    def test_action_boundary_uses_actual_next_action(self):
        actions=[{'start_sim_time_s':0,'planned_control_steps':30},{'start_sim_time_s':.6,'planned_control_steps':30}]
        self.assertIs(module.active_action(actions,.58),actions[0]);self.assertIs(module.active_action(actions,.6),actions[1])
        with self.assertRaises(ValueError):module.active_action(actions,1.2)
        with self.assertRaises(ValueError):module.active_action(actions+[actions[0]],.1)
    def test_no_step_session_or_audio_and_lazy_render(self):
        source=path.read_text(encoding='utf-8-sig');tree=ast.parse(source)
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('mj_step',attrs);self.assertNotIn('InferenceSession',attrs);self.assertEqual(attrs.count('mj_forward'),1)
        self.assertIn("'-threads','1'",source);self.assertIn("'audio':'none'",source)
        self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=='mujoco' for a in n.names) for n in tree.body))
if __name__=='__main__':unittest.main()
