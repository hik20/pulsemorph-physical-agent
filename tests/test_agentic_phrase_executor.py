"""Prospective adapter/gate/budget fixtures: zero real session/physics calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'experiments/agentic/phrase'
sys.path[:0]=[str(ROOT/'src'),str(HERE)]
import adapter_v1 as adapter
spec=importlib.util.spec_from_file_location('phrase_executor',HERE/'execute_phrase_v1.py')
executor=importlib.util.module_from_spec(spec);spec.loader.exec_module(executor)
from pulsemorph.motion.contracts import load_reference,MotionReference
from pulsemorph.agentic.phrase_reference import plain


class PhraseExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=load_reference(executor.REFERENCE_PLAN/'common_reference')
        with np.load(executor.BASE/'reference_cache.npz') as c:
            cls.carrier={k:c['committed__'+k][0].copy() for k in ('dof_pos','dof_vel','body_pos','body_rot','body_vel','body_ang_vel')}
        with np.load(executor.BASE/'policy_inputs.npz') as c:cls.inputs={k:c[k].copy() for k in c.files}
        with np.load(executor.BASE/'policy_outputs.npz') as c:cls.outputs={k:c[k].copy() for k in c.files}
        cls.arrays=[np.load(executor.BASE/n) for n in ('qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy')]
    def gate(self):return adapter.PrefixGate(self.inputs,self.outputs,*self.arrays)
    def test_full_finite_horizon_and_no_root_carrier_after_initial(self):
        p=adapter.PhrasePlayer(self.reference,self.carrier,16)
        self.assertEqual(p.total_frames,400)
        for i in (0,167,168,399):self.assertEqual(p.get_future_references(i,[1,2,4,8])['dof_pos'].shape,(4,29))
        self.assertEqual(set(p.get_state_at_frame(400)),{'dof_pos','dof_vel'})
        for i in (-1,400):
            with self.assertRaises(ValueError):p.get_future_references(i,[1,2,4,8])
        with self.assertRaises(ValueError):p.get_future_references(0,[1,2,4,9])
    def test_reject_altered_initial_and_producer(self):
        bad={k:v.copy() for k,v in self.carrier.items()};bad['dof_pos'][0]+=.01
        with self.assertRaises(ValueError):adapter.PhrasePlayer(self.reference,bad,16)
        m=plain(self.reference.metadata);m['producer']['id']='unreviewed'
        badref=MotionReference(m,self.reference.time_offsets_s,self.reference.joint_position_rad,self.reference.joint_velocity_rad_s,self.reference.anchor_orientation_xyzw)
        with self.assertRaises(ValueError):adapter.PhrasePlayer(badref,self.carrier,16)
    def test_exact_gate_all_rows_then_continuation(self):
        g=self.gate();g.state(0,*[a[0] for a in self.arrays[:3]])
        for call in range(169):
            prefix=f'call_{call:04d}__';g.feed(call,{n:self.inputs[prefix+n] for n in adapter.INPUTS});g.output(call,adapter.OUTPUTS,[self.outputs[prefix+n] for n in adapter.OUTPUTS])
            if call:
                g.target(call-1,self.arrays[3][call-1]);g.state(call,*[a[call] for a in self.arrays[:3]])
        self.assertTrue(g.passed);g.feed(169,{})
    def test_stop_each_mismatch_and_incomplete_boundary(self):
        g=self.gate();feed={n:self.inputs['call_0001__'+n].copy() for n in adapter.INPUTS};feed['current_dof_pos'][0,0]+=.01
        with self.assertRaisesRegex(ValueError,'input'):g.feed(1,feed)
        values=[self.outputs['call_0001__'+n].copy() for n in adapter.OUTPUTS];values[0][0,0]+=.01
        with self.assertRaisesRegex(ValueError,'output'):g.output(1,adapter.OUTPUTS,values)
        with self.assertRaisesRegex(ValueError,'qpos'):g.state(0,self.arrays[0][0]+.01,self.arrays[1][0],self.arrays[2][0])
        with self.assertRaisesRegex(ValueError,'appliedPD'):g.target(0,self.arrays[3][0]+.01)
        with self.assertRaisesRegex(ValueError,'incomplete'):g.state(168,*[a[168] for a in self.arrays[:3]])
        with self.assertRaisesRegex(ValueError,'not passed'):g.feed(169,{})
    def test_call_step_caps_and_prefix_stop(self):
        for i in range(169):adapter.call_budget(i,False)
        with self.assertRaises(RuntimeError):adapter.call_budget(169,False)
        for i in range(169,401):adapter.call_budget(i,True)
        for i in (-1,401,True):
            with self.assertRaises(RuntimeError):adapter.call_budget(i,True)
        adapter.step_budget(7999)
        for i in (-1,8000,True):
            with self.assertRaises(RuntimeError):adapter.step_budget(i)
    def test_actual_graph_contract_and_reject_added_body_feed(self):
        d=json.loads((ROOT/'artifacts/runs/agentic/app_future_joint_probe_v1/metadata_failure_review_v1.json').read_text())['actual']
        def session(v):return SimpleNamespace(get_inputs=lambda:[SimpleNamespace(**x) for x in v['inputs']],get_outputs=lambda:[SimpleNamespace(**x) for x in v['outputs']])
        adapter.interface_guard(session(d))
        for group,key,value in (('inputs','name','mimic_future_rot'),('inputs','shape',['batch_size',33,4]),('outputs','type','tensor(double)')):
            bad=copy.deepcopy(d);bad[group][0][key]=value
            with self.assertRaises(ValueError):adapter.interface_guard(session(bad))
    def test_pin_mutation_fails_without_calls(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'input';p.write_text('a');pins={str(p):executor.sha(p)};executor.verify(pins);p.write_text('b')
            with self.assertRaises(ValueError):executor.verify(pins)
    def test_output_exclusive_no_second_launch(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'used';p.mkdir()
            with patch.object(executor,'OUT',p),patch.object(executor,'approval_check',return_value={'minimum_free_bytes':0}),patch.object(executor.subprocess,'Popen') as process:
                with self.assertRaises(FileExistsError):executor.launch('fixture')
                process.assert_not_called()


if __name__=='__main__':unittest.main()
