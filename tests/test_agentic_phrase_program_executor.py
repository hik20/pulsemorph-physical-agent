"""Program executor boundaries with synthetic metadata/cached arrays only.

No InferenceSession, policy run, mj_forward, mj_step or physical output creation.
"""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'experiments/agentic/phrase'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions'),str(HERE)]
import adapter_program_v1 as adapter
spec=importlib.util.spec_from_file_location('program_executor_boundary_fixture',HERE/'execute_program_v1.py')
executor=importlib.util.module_from_spec(spec);spec.loader.exec_module(executor)
from pulsemorph.motion.contracts import MotionReference,load_reference
from pulsemorph.agentic.phrase_reference import plain


class ProgramExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref=load_reference(ROOT/'artifacts/runs/agentic/phrase_reference_plan_v1/common_reference')
        metadata=plain(ref.metadata);metadata['producer']={'kind':'agentic','id':'phrase-program-reference','version':'1'}
        metadata['provenance']['phrase_program']={'order':['wave','high5']}
        cls.reference=MotionReference(metadata,ref.time_offsets_s,ref.joint_position_rad,ref.joint_velocity_rad_s,ref.anchor_orientation_xyzw)
        base=ROOT/'artifacts/runs/agentic/app/jobs/20261027T023422_8efe4bc4/trials/baseline'
        with np.load(base/'reference_cache.npz',allow_pickle=False) as cache:cls.carrier={k:cache['committed__'+k][0].copy() for k in ('dof_pos','dof_vel','body_pos','body_rot','body_vel','body_ang_vel')}
        with np.load(base/'policy_inputs.npz',allow_pickle=False) as cache:cls.inputs={k:cache[k].copy() for k in cache.files}
        with np.load(base/'policy_outputs.npz',allow_pickle=False) as cache:cls.outputs={k:cache[k].copy() for k in cache.files}
        cls.arrays=[np.load(base/n,allow_pickle=False) for n in ('qpos.npy','qvel.npy','state_times.npy','applied_joint_targets.npy')]

    def gate(self):return adapter.PrefixGate(self.inputs,self.outputs,*self.arrays)

    def test_finite_horizon_and_no_postinitial_root_carrier(self):
        player=adapter.PhrasePlayer(self.reference,self.carrier,16)
        for index in (0,17,18,399):
            refs=player.get_future_references(index,[1,2,4,8]);self.assertEqual(refs['dof_pos'].shape,(4,29));self.assertTrue(np.isfinite(refs['dof_pos']).all());self.assertNotIn('body_pos',refs)
        self.assertEqual(set(player.get_state_at_frame(400)),{'dof_pos','dof_vel'})
        for index in (-1,400,True,1.5):
            with self.subTest(index=index),self.assertRaises(ValueError):player.get_future_references(index,[1,2,4,8])
        with self.assertRaises(ValueError):player.get_future_references(0,[1,2,4,9])

    def test_authentic_initial_producer_and_horizon_guards(self):
        for field in ('dof_pos','body_rot'):
            carrier={k:v.copy() for k,v in self.carrier.items()}
            if field=='body_rot':carrier[field][16,0]+=.01
            else:carrier[field].flat[0]+=.01
            with self.subTest(field=field),self.assertRaises(ValueError):adapter.PhrasePlayer(self.reference,carrier,16)
        metadata=plain(self.reference.metadata);metadata['producer']['id']='unreviewed'
        altered=MotionReference(metadata,self.reference.time_offsets_s,self.reference.joint_position_rad,self.reference.joint_velocity_rad_s,self.reference.anchor_orientation_xyzw)
        with self.assertRaises(ValueError):adapter.PhrasePlayer(altered,self.carrier,16)
        metadata=plain(self.reference.metadata);metadata['validity']['end_time_s']=8.12
        short=MotionReference(metadata,self.reference.time_offsets_s[:-1],self.reference.joint_position_rad[:-1],self.reference.joint_velocity_rad_s[:-1],self.reference.anchor_orientation_xyzw[:-1])
        with self.assertRaises(ValueError):adapter.PhrasePlayer(short,self.carrier,16)

    def test_exact_eighteen_control_gate_then_continuation(self):
        gate=self.gate();gate.state(0,*[a[0] for a in self.arrays[:3]])
        for call in range(19):
            prefix=f'call_{call:04d}__';gate.feed(call,{n:self.inputs[prefix+n] for n in adapter.INPUTS});gate.output(call,adapter.OUTPUTS,[self.outputs[prefix+n] for n in adapter.OUTPUTS])
            if call:gate.target(call-1,self.arrays[3][call-1]);gate.state(call,*[a[call] for a in self.arrays[:3]])
        self.assertTrue(gate.passed);self.assertEqual((gate.inputs_checked,gate.outputs_checked,gate.states_checked,gate.targets_checked),(19,19,19,18));gate.feed(19,{})

    def test_each_prefix_mismatch_and_incomplete_boundary_rejects(self):
        feed={n:self.inputs['call_0001__'+n].copy() for n in adapter.INPUTS};feed['current_dof_pos'][0,0]+=.01
        with self.assertRaisesRegex(ValueError,'input'):self.gate().feed(1,feed)
        values=[self.outputs['call_0001__'+n].copy() for n in adapter.OUTPUTS];values[0][0,0]+=.01
        with self.assertRaisesRegex(ValueError,'output'):self.gate().output(1,adapter.OUTPUTS,values)
        for kind in ('qpos','qvel','time'):
            q,v,t=[a[0].copy() for a in self.arrays[:3]]
            if kind=='qpos':q[0]+=.01
            elif kind=='qvel':v[0]+=.01
            else:t+=.001
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.gate().state(0,q,v,t)
        with self.assertRaisesRegex(ValueError,'appliedPD'):self.gate().target(0,self.arrays[3][0]+.01)
        with self.assertRaisesRegex(ValueError,'incomplete'):self.gate().state(18,*[a[18] for a in self.arrays[:3]])
        with self.assertRaisesRegex(ValueError,'not passed'):self.gate().feed(19,{})

    def test_call_step_caps_and_short_prefix(self):
        for i in range(19):adapter.call_budget(i,False)
        with self.assertRaises(RuntimeError):adapter.call_budget(19,False)
        for i in (19,400):adapter.call_budget(i,True)
        for i in (-1,401,True,1.5):
            with self.subTest(call=i),self.assertRaises(RuntimeError):adapter.call_budget(i,True)
        adapter.step_budget(7999)
        for i in (-1,8000,True,1.5):
            with self.subTest(step=i),self.assertRaises(RuntimeError):adapter.step_budget(i)

    def test_symbolic_graph_interface_excludes_body_carriers(self):
        shapes=[["batch_size",4],["batch_size",29],["batch_size",29],["batch_size",3],["batch_size",1,29],["batch_size",4,4],["batch_size",4,29],["batch_size",4,29]]
        def session(extra=False,badtype=False):return SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name=n,shape=s,type='tensor(float)') for n,s in zip(adapter.INPUTS,shapes)]+([SimpleNamespace(name='mimic_future_pos',shape=['batch_size',4,33,3],type='tensor(float)')] if extra else []),get_outputs=lambda:[SimpleNamespace(name=n,shape=['batch_size',29],type='tensor(double)' if badtype else 'tensor(float)') for n in adapter.OUTPUTS])
        adapter.interface_guard(session())
        for candidate in (session(extra=True),session(badtype=True)):
            with self.assertRaises(ValueError):adapter.interface_guard(candidate)

    def test_exclusive_output_and_reverse_order_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            used=Path(directory)/'used';used.mkdir()
            with patch.object(executor,'OUT',used),patch.object(executor,'approval_check',return_value={'minimum_free_bytes':0}),patch.object(executor.subprocess,'Popen') as process:
                with self.assertRaises(FileExistsError):executor.launch('fixture')
                process.assert_not_called()
        metadata=plain(self.reference.metadata);metadata['provenance']['phrase_program']['order']=['high5','wave']
        reversed_ref=MotionReference(metadata,self.reference.time_offsets_s,self.reference.joint_position_rad,self.reference.joint_velocity_rad_s,self.reference.anchor_orientation_xyzw)
        with patch('pulsemorph.motion.contracts.load_reference',return_value=reversed_ref),self.assertRaisesRegex(ValueError,'Wave-first'):executor.preflight()

    def test_worker_numeric_AST_matches_frozen_executor_except_provenance(self):
        old=ast.parse((HERE/'execute_phrase_v1.py').read_text());new=ast.parse((HERE/'execute_program_v1.py').read_text())
        class Normalize(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value in ('pulsemorph.phrase_physical_trial.v1','pulsemorph.phrase_program_physical_trial.v1'):node.value='TRIAL_PROVENANCE'
                return node
            def visit_Dict(self,node):
                self.generic_visit(node)
                for i,key in enumerate(node.keys):
                    if isinstance(key,ast.Constant) and key.value=='audio':node.values[i]=ast.Constant(value='RECORDED_AUDIO_CONTEXT')
                return node
        for name in ('worker','launch','approval_check','environment'):
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(Normalize().visit(a),include_attributes=False),ast.dump(Normalize().visit(b),include_attributes=False),name)

if __name__=='__main__':unittest.main()
