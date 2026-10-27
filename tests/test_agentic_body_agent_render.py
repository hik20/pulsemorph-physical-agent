"""Body Agent renderer admission/scheduling only; no native renderer or FK."""
import ast
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from pulsemorph.agentic.embodied.app import renderer_v1 as r

class BodyAgentRenderTests(unittest.TestCase):
    def test_full_budget_250_frames_and_odd_trace(self):
        self.assertEqual(len(r.frames_for(np.arange(501)*.02)),250)
        self.assertEqual(len(r.frames_for(np.arange(500)*.02)),250)
        with self.assertRaises(ValueError):r.frames_for(np.arange(503)*.02)
    def test_boundary_completed_previous_not_next(self):
        a=[{'start_sim_time_s':0,'planned_control_steps':30},{'start_sim_time_s':.6,'planned_control_steps':30}]
        self.assertIs(r.active_action(a,.6),a[0]);self.assertIs(r.active_action(a,.62),a[1])
    def test_precancel_no_admission_or_fk(self):
        event=threading.Event();event.set()
        with patch.object(r,'prepare_actual',side_effect=AssertionError('must not admit')):
            with self.assertRaisesRegex(ValueError,'cancelled'):r.render_actual_trace('trace',{},[],'output',cancel_event=event,max_fk=250)
    def test_entry_returns_separate_display_fk_count(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'media';plan={'frames':65}
            with patch.object(r,'prepare_actual',return_value=(plan,('fixture',))),patch.object(r,'render') as fake:
                result=r.render_actual_trace('trace',{},[],output,max_fk=250)
            self.assertEqual(result['render_FK'],65);self.assertEqual(result['video'],str(output/'episode.mp4'));fake.assert_called_once()
    def test_actual_only_no_model_steps_one_fk_callsite(self):
        text=Path(r.__file__).read_text();tree=ast.parse(text)
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(attrs.count('mj_forward'),1);self.assertNotIn('mj_step',attrs);self.assertNotIn('InferenceSession',attrs)
        self.assertIn("indices[-1]=len(times)-1",text);self.assertIn("'audio':'none'",text);self.assertIn("'-threads','1'",text)
if __name__=='__main__':unittest.main()
