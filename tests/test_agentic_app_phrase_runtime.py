"""Private phrase cache/runtime fixtures; no physical or ONNX calls."""
from pathlib import Path
import json
import copy
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch,MagicMock
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions')]
from pulsemorph.agentic.app.phrase.assets import PhraseAssets,ASSET_ROOT,PUBLIC_PYTHON,sha
from pulsemorph.agentic.app.phrase.runtime import PhraseRuntime
from pulsemorph.agentic.app.phrase.worker import make_gate,call_budget,step_budget,interface_guard,check_current_torso
from pulsemorph.agentic.phrase_program import ProgramCompileError
from pulsemorph.agentic.phrase_reference_v2 import compose_ordered_pair


def program(**changes):return {'schema':'pulsemorph.phrase_program.v1','order':['wave','high5'],'heading_policy':'recorded_world','first_left_gain':1.,'first_right_gain':1.,'second_left_gain':1.,'second_right_gain':1.,'rationale':'fixture',**changes}


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='phrase-runtime-fixtures-');cls.path=Path(cls.temp.name)
        cls.runtime=PhraseRuntime(ROOT,PUBLIC_PYTHON,cls.path/'jobs')
        cls.ref,_=compose_ordered_pair(cls.runtime.assets.references(),order=('wave','high5'),heading_policy='recorded_world',reference_id='fixture_reference',sources=[])
        cls.music=cls.path/'fixture.wav'
        with wave.open(str(cls.music),'wb') as w:w.setparams((1,2,48000,0,'NONE','not compressed'));w.writeframes(bytes(384000*2))
        cls.observation={'available':True,'path':str(cls.music),'sha256':sha(cls.music),'sample_rate':48000,'observed_through_sample':384000,'duration_s':8}
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def output(self,name):return self.path/'jobs'/name
    def test_cache_only_reader_does_not_open_historical_paths(self):
        original=Path.open
        def guarded(p,*a,**kw):
            self.assertNotIn('artifacts/runs/agentic/app/jobs',p.as_posix())
            return original(p,*a,**kw)
        with patch.object(Path,'open',guarded):
            assets=PhraseAssets(ROOT);self.assertEqual(set(assets.references()),{'wave','high5'});self.assertEqual(assets.arrays('golden_prefix')['qpos'].shape,(19,36))
    def test_cache_tamper_preserved_and_rejected(self):
        dest=self.path/'tampered-assets';shutil.copytree(ASSET_ROOT,dest)
        p=dest/'golden_prefix.npz';p.write_bytes(b'corrupt fixture')
        with self.assertRaises(ValueError):PhraseAssets(ROOT,dest)
        self.assertEqual(p.read_bytes(),b'corrupt fixture')
    def test_path_boundary_reverse_and_baseline_domain_before_worker(self):
        with self.assertRaises(ValueError):self.runtime._directory(self.path/'outside')
        with patch('pulsemorph.agentic.app.phrase.runtime.subprocess.Popen') as popen:
            for p,index in ((program(order=['high5','wave']),0),(program(first_left_gain=1.1),-1)):
                with self.assertRaises(ValueError):self.runtime.run_trial(p,self.output('unsupported'),'fixture',index,self.observation)
            popen.assert_not_called()
    def test_music_hash_duration_header_and_truncation_reject(self):
        for changes in ({'sha256':'0'*64},{'duration_s':4},{'observed_through_sample':1},{'available':False}):
            with self.assertRaises(ValueError):self.runtime._music({**self.observation,**changes})
        p=self.path/'truncated.wav';p.write_bytes(self.music.read_bytes()[:-100])
        with self.assertRaises(ValueError):self.runtime._music({**self.observation,'path':str(p),'sha256':sha(p)})
    def test_cancel_before_execution_and_existing_output_no_worker(self):
        event=threading.Event();event.set()
        with patch('pulsemorph.agentic.app.phrase.runtime.subprocess.Popen') as popen:
            with self.assertRaises(RuntimeError):self.runtime.run_trial(program(),self.output('cancel'),'fixture',0,self.observation,event)
            self.assertFalse(self.output('cancel').exists())
            self.output('used').mkdir(parents=True)
            with self.assertRaises(FileExistsError):self.runtime.run_trial(program(),self.output('used'),'fixture',0,self.observation)
            popen.assert_not_called()
    def test_compile_rejection_preserved_no_worker(self):
        rejection=ProgramCompileError('fixture continuous bound',{'status':'rejected_joint_ranges','violations':['left_elbow_joint']})
        with patch('pulsemorph.agentic.app.phrase.runtime.compile_program',side_effect=rejection),patch('pulsemorph.agentic.app.phrase.runtime.subprocess.Popen') as popen:
            with self.assertRaises(ProgramCompileError):self.runtime.run_trial(program(),self.output('reject'),'fixture',0,self.observation)
            self.assertEqual(json.loads((self.output('reject')/'rejection.json').read_text())['new_onnx_calls'],0);popen.assert_not_called()
    def test_worker_failure_preserves_request_no_retry(self):
        process=MagicMock();process.poll.return_value=1;process.returncode=1
        with patch('pulsemorph.agentic.app.phrase.runtime.compile_program',return_value=(self.ref,{'fixture':True})),patch('pulsemorph.agentic.app.phrase.runtime.subprocess.Popen',return_value=process) as popen:
            with self.assertRaises(RuntimeError):self.runtime.run_trial(program(),self.output('failed'),'fixture',0,self.observation)
            self.assertEqual(popen.call_count,1);self.assertTrue((self.output('failed')/'request.json').exists())
            with self.assertRaises(FileExistsError):self.runtime.run_trial(program(),self.output('failed'),'fixture',0,self.observation)
            self.assertEqual(popen.call_count,1)
    def test_postlaunch_cancel_kills_only_owned_process(self):
        process=MagicMock();process.poll.return_value=None;event=MagicMock();event.is_set.side_effect=[False,False,True,True]
        with patch('pulsemorph.agentic.app.phrase.runtime.compile_program',return_value=(self.ref,{'fixture':True})),patch('pulsemorph.agentic.app.phrase.runtime.subprocess.Popen',return_value=process):
            with self.assertRaises(RuntimeError):self.runtime.run_trial(program(),self.output('midcancel'),'fixture',0,self.observation,event)
            process.kill.assert_called_once();self.assertTrue((self.output('midcancel')/'interrupted.json').exists())
    def test_exact_safe_golden_gate_and_caps(self):
        assets=self.runtime.assets;values=assets.arrays('golden_prefix');gate=make_gate(assets);gate.state(0,values['qpos'][0],values['qvel'][0],values['time'][0])
        from pulsemorph.agentic.app.phrase.assets import INPUTS,OUTPUTS
        for c in range(19):
            gate.feed(c,{n:values[f'inputs__call_{c:04d}__'+n] for n in INPUTS});gate.output(c,OUTPUTS,[values[f'outputs__call_{c:04d}__'+n] for n in OUTPUTS])
            if c:gate.target(c-1,values['PD'][c-1]);gate.state(c,values['qpos'][c],values['qvel'][c],values['time'][c])
        self.assertTrue(gate.passed)
        with self.assertRaises(RuntimeError):call_budget(19,False)
        call_budget(400,True);step_budget(7999)
        with self.assertRaises(RuntimeError):call_budget(401,True)
        with self.assertRaises(RuntimeError):step_budget(8000)
    def test_evaluation_same_heading_baseline_and_own_measured_ratios(self):
        base={'program':program(),'metrics':{'arm_rms_rad':[2,3,4,5],'physical_gate_passed':True}}
        trial={'program':program(first_left_gain=.85),'metrics':{'arm_rms_rad':[1.7,3,4.6,5.75],'physical_gate_passed':True}}
        with patch.object(self.runtime,'_verify_trial',return_value={}):
            result=self.runtime.evaluate_trial(trial,base,[.85,1,1.15,1.15]);self.assertEqual(result['goals_passed'],4);self.assertTrue(result['integrity_passed']);self.assertEqual(result['goal_total'],4)
            base['program']['heading_policy']='align_join_heading'
            with self.assertRaises(ValueError):self.runtime.evaluate_trial(trial,base,[1]*4)
    def test_evaluation_integrity_failure_never_becomes_metrics(self):
        with patch.object(self.runtime,'_verify_trial',side_effect=RuntimeError('fixture corruption')):
            with self.assertRaises(RuntimeError):self.runtime.evaluate_trial({}, {},[1]*4)
    def test_source_mutation_blocks_before_tools(self):
        with patch.object(self.runtime,'implementation_pins',return_value={'changed':'pin'}),patch('pulsemorph.agentic.app.phrase.runtime.subprocess.Popen') as popen:
            with self.assertRaises(RuntimeError):self.runtime.run_trial(program(),self.output('changed'),'fixture',0,self.observation)
            popen.assert_not_called()
    def test_current_torso_sign_or_one_float32_step_never_passes_exact_audit(self):
        rotation=np.array([0.,0.,0.,1.],dtype=np.float32)
        self.assertEqual(check_current_torso(rotation,rotation[None].copy()),0)
        for recorded in (-rotation[None],np.nextafter(rotation[None],np.float32(2))):
            with self.assertRaisesRegex(ValueError,'current_anchor_rot'):check_current_torso(rotation,recorded)
    def test_native_numerical_module_pins_captured_in_trial_implementation(self):
        pins=self.runtime.environment['module_pins'];names=[Path(p).name.lower() for p in pins]
        self.assertTrue(any('_multiarray_umath' in p and p.endswith('.pyd') for p in names))
        self.assertTrue(any('onnxruntime_pybind11_state' in p and p.endswith('.pyd') for p in names))
        self.assertTrue(any('_functions' in p and p.endswith('.pyd') for p in names))
        self.assertTrue(all(self.runtime._implementation[p]==h for p,h in pins.items()))
    def test_evaluated_wrapper_keeps_exact_raw_proof_and_rejects_tamper(self):
        # Deliberate fake proof files isolate the binding layer, never physics.
        directory=self.output('binding-fixture');directory.mkdir(parents=True)
        required={'qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy','policy_inputs.npz','policy_outputs.npz','policy_calls.json','substep_telemetry.npz','request.json','common_reference/manifest.json','common_reference/arrays.npz'}
        for name in required:
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture-only binding payload')
        raw={'arm_rms_rad':[1.,2.,3.,4.],'physical_gate_passed':True}
        audit={'schema':'pulsemorph.app_phrase_audit.v1','integrity_passed':True,'fresh_mj_kinematics_calls':400,'metrics':raw,'inventory':{name:sha(directory/name) for name in required}}
        (directory/'audit.json').write_text(json.dumps(audit))
        result={'schema':'pulsemorph.app_phrase_physical_trial.v1','status':'completed','integrity_passed':True,'program':program(),'metrics':raw,'audit_sha256':sha(directory/'audit.json'),'counts':{'calls_attempted':401,'calls_completed':401,'steps':8000,'forward':400,'initial':1},'source_pins':self.runtime._implementation,'request_sha256':sha(directory/'request.json')}
        (directory/'result.json').write_text(json.dumps(result))
        trial={'program':program(),'paths':{'directory':str(directory)},'physical_metrics':copy.deepcopy(raw),'metrics':{**raw,'features':[1]*4,'goals_passed':4}}
        self.runtime._verify_trial(trial)
        trial['physical_metrics']['arm_rms_rad'][0]=99
        with self.assertRaises(RuntimeError):self.runtime._verify_trial(trial)
    def test_baseline_evaluated_before_candidate_uses_physical_denominators(self):
        baseline={'program':program(),'physical_metrics':{'arm_rms_rad':[2.,3.,4.,5.],'physical_gate_passed':True},'metrics':{}}
        trial={'program':program(first_left_gain=.85),'physical_metrics':{'arm_rms_rad':[1.7,3.,4.6,5.75],'physical_gate_passed':True},'metrics':{'arm_rms_rad':[999]*4,'physical_gate_passed':False}}
        with patch.object(self.runtime,'_verify_trial',return_value={}):
            baseline['metrics']=self.runtime.evaluate_trial(baseline,baseline,[.85,1,1.15,1.15])
            evaluation=self.runtime.evaluate_trial(trial,baseline,[.85,1,1.15,1.15])
        np.testing.assert_allclose(evaluation['features'],[.85,1,1.15,1.15]);self.assertTrue(evaluation['physical_gate_passed'])
    def test_status_readonly_resource_gate_and_no_global_execution_claim(self):
        old=self.runtime.artifact_root;path=self.path/'status-only'/'not-created';self.runtime.artifact_root=path
        try:
            with patch('pulsemorph.agentic.app.phrase.runtime.shutil.disk_usage',return_value=type('Disk',(),{'free':255*1024**2})()) as disk:
                status=self.runtime.status();self.assertFalse(status['available']);self.assertEqual(status['state'],'insufficient_disk');self.assertEqual(disk.call_args.args[0],self.path)
            with patch('pulsemorph.agentic.app.phrase.runtime.shutil.disk_usage',return_value=type('Disk',(),{'free':512*1024**2})()):
                status=self.runtime.status();self.assertTrue(status['available']);self.assertEqual(status['state'],'profile_verified_no_controller_execution');self.assertFalse(status['execution_verified'])
            self.assertFalse(path.parent.exists())
        finally:self.runtime.artifact_root=old
    def test_status_integrity_failure_never_reaches_resource_or_tools(self):
        with patch.object(self.runtime,'_unchanged',side_effect=RuntimeError('fixture changed')),patch('pulsemorph.agentic.app.phrase.runtime.shutil.disk_usage') as disk:
            status=self.runtime.status();self.assertFalse(status['available']);self.assertEqual(status['state'],'integrity_changed');self.assertFalse(status['execution_verified']);disk.assert_not_called()


if __name__=='__main__':unittest.main()
