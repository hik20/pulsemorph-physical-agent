"""Owned audio synthesis and measurements only, no model/robot calls."""
import hashlib
import io
import tempfile
from pathlib import Path
import unittest
import wave
import numpy as np
from pulsemorph.agentic.performance.music_v1 import synthesize_music,create_demo_music

class PerformanceMusicTests(unittest.TestCase):
    def test_exact_score_grid_and_actual_pcm_measurements(self):
        raw,d=synthesize_music()
        with wave.open(io.BytesIO(raw),'rb') as w:
            self.assertEqual((w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes()),(48000,1,2,806400));pcm=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
        self.assertEqual(d['sha256'],hashlib.sha256(raw).hexdigest());self.assertEqual(len(d['beats_s']),28);self.assertEqual(d['beats_s'][-1],16.2)
        self.assertAlmostEqual(d['rms'],float(np.sqrt(np.mean(pcm*pcm))),places=14);self.assertGreater(np.sqrt(np.mean(pcm[:4800]**2)),.01);self.assertLessEqual(d['peak_amplitude'],.921)
        self.assertEqual(len(d['waveform128']),128);self.assertFalse(d['source']['external_audio_assets'])
    def test_deterministic_score_and_exclusive_output(self):
        a,da=synthesize_music();b,db=synthesize_music();self.assertEqual(a,b);self.assertEqual(da,db)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'owned';d=create_demo_music(out);self.assertEqual(Path(d['path']).read_bytes(),a)
            with self.assertRaises(FileExistsError):create_demo_music(out)
if __name__=='__main__':unittest.main()
