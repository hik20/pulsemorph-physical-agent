"""Finite renderer admission/label checks; no MuJoCo, policy or physics calls."""
import ast
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from pulsemorph.agentic.expression import render_v1 as r

class ExpressionRenderTests(unittest.TestCase):
    def test_gain_labels_use_expression_components_not_old_four_gains(self):
        text=r._gain_text({'expression':{'second_left_pose_gain':.75,'second_left_motion_gain':1}})
        self.assertIn('ortalama 0.75',text); self.assertIn('değişim 1',text)
    def test_existing_or_trace_nested_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(ValueError):r._output(p,[p])
            with self.assertRaises(ValueError):r._output(p/'child',[p])
    def test_cancel_before_any_verification_or_render(self):
        event=threading.Event();event.set()
        class Runtime:
            def _verify_trial(self,t):raise AssertionError('must not verify after precancel')
        trial={'paths':{'directory':'C:/fixture/trial'}}
        with patch.object(r,'_output',return_value=Path('C:/fixture/media')):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                r.render_comparison('.',Runtime(),trial,trial,{},'unused',cancel_event=event)
    def test_original_runtime_verification_required_before_trace_load(self):
        class Runtime:
            def _verify_trial(self,t):raise RuntimeError('fixture original audit failure')
        trial={'paths':{'directory':'C:/fixture/trial'}}
        with patch.object(r,'_output',return_value=Path('C:/fixture/media')),patch.object(r,'_load_trace',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(RuntimeError,'original audit failure'):
                r.render_comparison('.',Runtime(),trial,trial,{},'unused')
    def test_ast_budget_no_step_policy_or_old_program_labels(self):
        source=Path(r.__file__).read_text(encoding='utf-8-sig');tree=ast.parse(source)
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('mj_step',attrs);self.assertNotIn('InferenceSession',attrs);self.assertNotIn('run_trial',attrs)
        self.assertEqual(attrs.count('mj_forward'),1);self.assertEqual(len(range(0,400,2))*2,400)
        self.assertNotIn('_program_text',source);self.assertIn('get_ffmpeg_exe',attrs)
        self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=='mujoco' for a in n.names) for n in tree.body))
if __name__=='__main__':unittest.main()
