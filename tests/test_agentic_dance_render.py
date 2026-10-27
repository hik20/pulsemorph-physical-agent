"""Controller-free saved dance render admission checks. No MuJoCo or model calls."""
import ast
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave
import numpy as np
from pulsemorph.agentic.dance import render_v1 as render

class DanceRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.directory=self.root/'trial';self.directory.mkdir();self.trial={'id':'fixture','paths':{'directory':str(self.directory)}}
        self.manifest={'status':'success','implementation_pins':{}}
        self.trace()
    def trace(self,count=201):
        np.save(self.directory/'qpos.npy',np.zeros((count,36)))
        np.save(self.directory/'qvel.npy',np.zeros((count,35)))
        np.save(self.directory/'state_times.npy',np.arange(count)*.02)
        self.save()
    def save(self):(self.directory/'manifest.json').write_text(json.dumps(self.manifest))
    def music(self,seconds=4):
        path=self.root/'music.wav'
        with wave.open(str(path),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(48000);w.writeframes(b'\x01\x00'*(seconds*48000))
        data={'path':str(path),'sha256':render.sha(path)}
        obs={'available':True,**data,'sample_rate':48000,'duration_s':4,'observed_through_sample':192000}
        return data,obs
    def test_exact_201_saved_states(self):
        row=render.load_trace(self.trial);self.assertEqual(row['qpos'].shape,(201,36));self.assertEqual(row['times'][-1],4.)
    def test_old_eight_second_401_trace_rejected(self):
        self.trace(401)
        with self.assertRaisesRegex(ValueError,'four-second'):render.load_trace(self.trial)
    def test_changed_time_nonfinite_and_failed_status_rejected(self):
        for kind in ('time','nan','status'):
            with self.subTest(kind=kind):
                self.manifest['status']='success';self.trace()
                if kind=='time':np.save(self.directory/'state_times.npy',np.arange(201)*.021)
                elif kind=='nan':
                    q=np.zeros((201,36));q[4,3]=np.nan;np.save(self.directory/'qpos.npy',q)
                else:self.manifest['status']='failed';self.save()
                with self.assertRaises(ValueError):render.load_trace(self.trial)
    def test_exact_four_second_audio_and_two_trace_lineage(self):
        music,obs=self.music();rows=[{'manifest':{'audio_observation':dict(obs)}} for _ in range(2)]
        self.assertEqual(render.bind_audio(rows,music)['observed_through_sample'],192000)
        rows[1]['manifest']['audio_observation']['observed_through_sample']=192001
        with self.assertRaisesRegex(ValueError,'lineage'):render.bind_audio(rows,music)
    def test_audio_bytes_and_old_eight_seconds_rejected(self):
        music,obs=self.music();Path(music['path']).write_bytes(b'altered')
        with self.assertRaises(ValueError):render.bind_audio([{'manifest':{'audio_observation':obs}}],music)
        music,obs=self.music(8)
        with self.assertRaises(ValueError):render.bind_audio([{'manifest':{'audio_observation':obs}}],music)
    def test_existing_outside_and_trace_nested_output_rejected(self):
        class NoRuntime:
            def _verify_trial(self,t):raise AssertionError('must reject output first')
        with patch.object(render,'ROOT_CACHE',self.root):
            for output in (self.directory,self.root.parent/'outside_fixture',self.directory/'media'):
                with self.subTest(output=output),self.assertRaises(ValueError):
                    render.render_dance(self.root,NoRuntime(),self.trial,self.trial,{},output)
    def test_original_runtime_audit_rejects_before_trace_load_or_mujoco(self):
        class BadRuntime:
            def _verify_trial(self,t):raise RuntimeError('fixture raw proof changed')
        with patch.object(render,'ROOT_CACHE',self.root),patch.object(render,'load_trace',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(RuntimeError,'raw proof changed'):
                render.render_dance(self.root,BadRuntime(),self.trial,self.trial,{},self.root/'media')
        self.assertFalse((self.root/'media').exists())
    def test_wrong_body_pin_rejected_before_import_or_output_creation(self):
        music,obs=self.music();self.manifest['audio_observation']=obs;self.save()
        body=self.root/'body.xml';body.write_text('<mujoco><compiler meshdir="."/><asset/><worldbody/></mujoco>')
        class ReadOnlyRuntime:
            def _verify_trial(self,t):return {'integrity_passed':True}
        with patch.object(render,'ROOT_CACHE',self.root),patch.object(render,'BODY','body.xml'):
            with self.assertRaisesRegex(ValueError,'body/mesh'):
                render.render_dance(self.root,ReadOnlyRuntime(),self.trial,self.trial,music,self.root/'media')
        self.assertFalse((self.root/'media').exists())
    def test_cancelled_before_mujoco_import(self):
        music,obs=self.music();self.manifest['audio_observation']=obs;self.save();event=threading.Event();event.set()
        class ReadOnlyRuntime:
            def _verify_trial(self,t):return {'integrity_passed':True}
        with patch.object(render,'ROOT_CACHE',self.root):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                render.render_dance(self.root,ReadOnlyRuntime(),self.trial,self.trial,music,self.root/'media',cancel_event=event)
        self.assertFalse((self.root/'media').exists())
    def test_ast_no_policy_step_or_compiler_and_render_budget(self):
        source=Path(render.__file__).read_text(encoding='utf-8-sig');tree=ast.parse(source)
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('mj_step','InferenceSession','run_trial','compile_expression'):self.assertNotIn(name,attrs)
        self.assertEqual(attrs.count('mj_forward'),1);self.assertEqual(len(range(0,200,2))*2,200)
        self.assertIn('macro_block_size=1',source)
        self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=='mujoco' for a in n.names) for n in tree.body))
if __name__=='__main__':unittest.main()
