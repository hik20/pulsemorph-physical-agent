"""Observed-audio binding and real subprocess cancellation; no model/physics jobs."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave
from pulsemorph.agentic.app.renderer import _audio_binding,_run_mux

class RendererTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'observed.wav'
        with wave.open(str(self.path),'wb') as pcm:
            pcm.setparams((1,2,8000,32000,'NONE','not compressed'))
            pcm.writeframes(b'\0\0'*32000)
        obs={'available':True,'path':str(self.path),'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
             'sample_rate':8000,'observed_through_sample':32000,'duration_s':4.}
        self.rows=[{'manifest':{'audio_observation':copy.deepcopy(obs)}} for _ in range(2)]

    def tearDown(self): self.temp.cleanup()

    def test_accepts_exact_observed_pcm(self):
        binding=_audio_binding(self.rows,self.path)
        self.assertEqual(binding['observed_through_sample'],32000)
        self.assertEqual(binding['sha256'],self.rows[0]['manifest']['audio_observation']['sha256'])

    def test_rejects_selected_trial_audio_mismatch(self):
        self.rows[1]['manifest']['audio_observation']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'path/hash'):_audio_binding(self.rows,self.path)

    def test_rejects_identical_audio_under_unobserved_path(self):
        other=self.path.with_name('posthoc.wav');other.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(ValueError,'path/hash'):_audio_binding(self.rows,other)

    def test_rejects_pcm_timing_mismatch(self):
        self.rows[1]['manifest']['audio_observation']['observed_through_sample']=88200
        with self.assertRaisesRegex(ValueError,'timing'):_audio_binding(self.rows,self.path)

    def test_rejects_mutation_on_post_mux_recheck(self):
        _audio_binding(self.rows,self.path)
        content=bytearray(self.path.read_bytes());content[-1]=1;self.path.write_bytes(content)
        with self.assertRaisesRegex(ValueError,'path/hash'):_audio_binding(self.rows,self.path)

    def test_no_audio_explicit_and_no_posthoc_track(self):
        rows=[{'manifest':{'audio_observation':{'available':False}}}]*2
        self.assertIsNone(_audio_binding(rows,None))
        with self.assertRaisesRegex(ValueError,'posthoc'):_audio_binding(rows,self.path)
        with self.assertRaisesRegex(ValueError,'both trials'):_audio_binding(self.rows,None)

    def test_cancel_kills_real_subprocess_promptly(self):
        event=threading.Event();processes=[];real_popen=subprocess.Popen
        def capture(*args,**kwargs):
            process=real_popen(*args,**kwargs);processes.append(process);return process
        timer=threading.Timer(.15,event.set);started=time.monotonic();timer.start()
        try:
            with patch('pulsemorph.agentic.app.renderer.subprocess.Popen',capture),self.assertRaisesRegex(RuntimeError,'cancelled'):
                _run_mux([sys.executable,'-c','import time; time.sleep(30)'],event)
        finally:timer.cancel()
        self.assertLess(time.monotonic()-started,3)
        self.assertIsNotNone(processes[0].poll())

    def test_timeout_kills_real_subprocess(self):
        started=time.monotonic()
        with self.assertRaisesRegex(RuntimeError,'timed out'):
            _run_mux([sys.executable,'-c','import time; time.sleep(30)'],timeout=.1)
        self.assertLess(time.monotonic()-started,3)
