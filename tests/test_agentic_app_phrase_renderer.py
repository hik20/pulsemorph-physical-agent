"""Synthetic saved-state/audio fixtures; no rendering, inference or physics."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from pulsemorph.agentic.app.phrase.renderer import _load_trace,_verify_trace,_audio_binding,_frame_indices,_directory

class PhraseRendererTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.trial=self.root/'trial';self.trial.mkdir();self.pin=self.root/'pinned_source.py';self.pin.write_bytes(b'fixture')
        np.save(self.trial/'qpos.npy',np.zeros((401,36)));np.save(self.trial/'qvel.npy',np.zeros((401,35)));np.save(self.trial/'state_times.npy',np.arange(401)*.02)
        self.result={'status':'completed','counts':{'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1},'source_pins':{str(self.pin):hashlib.sha256(self.pin.read_bytes()).hexdigest()}}
        self.save_result()
    def save_result(self):(self.trial/'result.json').write_text(json.dumps(self.result))
    def test_accepts_finite_saved_trace_and_exact_frame_budget(self):
        row=_load_trace(self.trial);_verify_trace(row);self.assertEqual(row['qpos'].shape,(401,36));self.assertEqual(list(_frame_indices()),list(range(0,400,2)));self.assertEqual(len(_frame_indices()),200)
    def test_wrong_counts_time_and_nonfinite_rejected(self):
        self.result['counts']['steps']=7999;self.save_result()
        with self.assertRaises(ValueError):_load_trace(self.trial)
        self.result['counts']['steps']=8000;self.save_result();np.save(self.trial/'state_times.npy',np.arange(401)*.03)
        with self.assertRaises(ValueError):_load_trace(self.trial)
        np.save(self.trial/'state_times.npy',np.arange(401)*.02);q=np.zeros((401,36));q[2,0]=np.nan;np.save(self.trial/'qpos.npy',q)
        with self.assertRaises(ValueError):_load_trace(self.trial)
    def test_snapshot_and_pinned_code_mutation_rejected(self):
        row=_load_trace(self.trial);self.pin.write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError,'Pinned'):_verify_trace(row)
        self.pin.write_bytes(b'fixture');(self.trial/'extra').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError,'trace changed'):_verify_trace(row)
    def audio(self,seconds=8):
        path=self.root/'score.wav'
        with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000);w.writeframes(b'\0\0'*(8000*seconds))
        digest=hashlib.sha256(path.read_bytes()).hexdigest();music={'path':str(path),'sha256':digest};obs={'available':True,'path':str(path),'sha256':digest,'sample_rate':8000,'observed_through_sample':8000*seconds,'duration_s':seconds};return music,obs
    def test_audio_requires_same_observed_eight_second_PCM(self):
        music,obs=self.audio();rows=[{'result':{'music_observation':dict(obs)}} for _ in range(2)];self.assertEqual(_audio_binding(rows,music)['duration_s'],8)
        rows[1]['result']['music_observation']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'lineage'):_audio_binding(rows,music)
    def test_audio_tamper_and_four_second_input_rejected(self):
        music,obs=self.audio();rows=[{'result':{'music_observation':obs}}];Path(music['path']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'hash'):_audio_binding(rows,music)
        music,obs=self.audio(4)
        with self.assertRaisesRegex(ValueError,'exactly8'):_audio_binding([{'result':{'music_observation':obs}}],music)
    def test_no_completed_trace_or_directory_no_replay(self):
        self.result['status']='failed';self.save_result()
        with self.assertRaises(ValueError):_load_trace(self.trial)
        with self.assertRaises(ValueError):_directory({'paths':{'directory':str(self.root/'absent')}})

if __name__=='__main__':unittest.main()
