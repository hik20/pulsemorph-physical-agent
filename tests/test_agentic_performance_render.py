"""Saved scheduling/music admission tests, no native FK/render/controller."""
import ast
import io
import hashlib
import tempfile
from pathlib import Path
import unittest
import wave
import numpy as np
from pulsemorph.agentic.performance.render_v1 import sample_indices,music_binding,render_performance,configure_display_buffer

class PerformanceRenderTests(unittest.TestCase):
    def test_display_framebuffer_grows_only_when_needed(self):
        from types import SimpleNamespace
        for width,height,expected in [(640,480,(1280,600)),(2048,1024,(2048,1024))]:
            display=SimpleNamespace(offwidth=width,offheight=height);model=SimpleNamespace(vis=SimpleNamespace(global_=display))
            configure_display_buffer(model);self.assertEqual((display.offwidth,display.offheight),expected)
    def test_pilot115_and_full420_endpoint_within_budget(self):
        indices,duration=sample_indices(np.arange(231)*.02)
        self.assertEqual(len(indices),115);self.assertEqual(indices[-1],230)
        indices,duration=sample_indices(np.arange(941)*.02,2.,16.8)
        self.assertEqual(len(indices),420);self.assertEqual(indices[0],100);self.assertEqual(indices[-1],940);self.assertEqual(duration,16.8)
    def test_clip_bounds_and_clock_no_hidden_clamp(self):
        for start,duration in [(-.02,1),(0,5),(0,.015),(.01,1)]:
            with self.subTest(start=start),self.assertRaises(ValueError):sample_indices(np.arange(231)*.02,start,duration)
        with self.assertRaises(ValueError):sample_indices([0,.02,0])
    def test_music_exact_bytes_beat_grid_and_offset(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'score.wav'
            with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(48000);w.writeframes(b'\0\0'*48000)
            music={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'duration_s':1.,'beats_s':[0.,.6]}
            binding=music_binding(music);self.assertEqual(binding['start_sim_time_s'],2.)
            music['beats_s']=[.6,0.]
            with self.assertRaises(ValueError):music_binding(music)
            music['beats_s']=[0.,.6];path.write_bytes(b'changed')
            with self.assertRaises(ValueError):music_binding(music)
    def test_ast_single_fk_no_nn_or_step_one_encoder_thread(self):
        import inspect
        text=inspect.getsource(__import__('pulsemorph.agentic.performance.render_v1',fromlist=['x']));tree=ast.parse(text)
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(calls.count('mj_forward'),1);self.assertNotIn('mj_step',calls);self.assertNotIn('InferenceSession',calls);self.assertIn("'-threads','1'",text)
        self.assertIn("'label_caller_supplied_not_attested'",text)
if __name__=='__main__':unittest.main()
