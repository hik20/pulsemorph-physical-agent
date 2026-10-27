"""Strict program/compiler tests, with no physics, ONNX or LLM operations."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions')]
from pulsemorph.agentic.phrase_program import parse_program,compile_program,ProgramCompileError,_gain_source,_arm_indices,GAIN_FIELDS
from pulsemorph.agentic.phrase_reference_v2 import compose_ordered_pair
from pulsemorph.motion.contracts import load_reference
from pulsemorph.motion.g1 import sample_reference


def program(**changes):
    return {'schema':'pulsemorph.phrase_program.v1','order':['wave','high5'],'heading_policy':'recorded_world',**{k:1. for k in GAIN_FIELDS},'rationale':'bounded reference fixture',**changes}


class ProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs={n:load_reference(ROOT/'artifacts/runs/agentic/app/jobs'/j/'trials/baseline/common_reference') for n,j in {'wave':'20261027T023422_8efe4bc4','high5':'20261027T022322_76db8965'}.items()}
    def test_parser_owns_values_and_accepts_domain_edges(self):
        value=program(first_left_gain=.75,second_right_gain=1.25,rationale='')
        parsed=parse_program(value);parsed['order'].reverse()
        self.assertEqual(value['order'],['wave','high5'])
        self.assertEqual(parsed['first_left_gain'],.75)
    def test_parser_rejects_unknown_missing_bool_nonfinite_and_bad_text(self):
        cases=[{},program(extra=1),program(schema='other'),program(order=('wave','high5')),program(order=['wave','wave']),program(heading_policy='auto'),program(rationale=True),program(rationale=None),program(rationale='x'*1201)]
        for gain in (True,False,None,float('nan'),float('inf'),10**1000,-.1,.7499,1.2501,'1'):
            cases.append(program(first_left_gain=gain))
        for value in cases:
            with self.subTest(value=str(value)[:80]):
                with self.assertRaises(ValueError):parse_program(value)
    def test_product_rule_and_exact_authentic_prefix_and_other_joints(self):
        source=self.refs['wave'];edited=_gain_source(source,1.1,.9);names=tuple(source.metadata['body']['joint_names']);groups=_arm_indices(names)
        prefix=source.time_offsets_s<=.5
        for name in ('joint_position_rad','joint_velocity_rad_s'):
            self.assertEqual(getattr(source,name)[prefix].tobytes(),getattr(edited,name)[prefix].tobytes())
        self.assertEqual(source.anchor_orientation_xyzw.tobytes(),edited.anchor_orientation_xyzw.tobytes())
        other=np.array([i for i in range(29) if i not in np.r_[*groups]])
        for name in ('joint_position_rad','joint_velocity_rad_s'):
            self.assertEqual(getattr(source,name)[:,other].tobytes(),getattr(edited,name)[:,other].tobytes())
        index=38;t=source.time_offsets_s[index];u=(t-.5)/.5;w=3*u*u-2*u**3;wd=(6*u-6*u*u)/.5
        for indices,gain in zip(groups,(1.1,.9)):
            delta=source.joint_position_rad[index,indices]-source.joint_position_rad[0,indices]
            np.testing.assert_allclose(edited.joint_velocity_rad_s[index,indices],(1+(gain-1)*w)*source.joint_velocity_rad_s[index,indices]+(gain-1)*wd*delta,rtol=0,atol=1e-14)
        times=np.array([.731,1.213,2.111]);eps=1e-6
        fd=(sample_reference(edited,times+eps)['dof_pos']-sample_reference(edited,times-eps)['dof_pos'])/(2*eps)
        np.testing.assert_allclose(fd,sample_reference(edited,times)['dof_vel'],atol=1e-7,rtol=0)
    def test_unit_gain_all_order_heading_regression_and_context_lineage(self):
        for order in (['wave','high5'],['high5','wave']):
            for heading in ('recorded_world','align_join_heading'):
                with self.subTest(order=order,heading=heading):
                    p=program(order=order,heading_policy=heading)
                    ref,report=compile_program(self.refs,p,reference_id='fixture_program',sources=[])
                    expected,_=compose_ordered_pair(self.refs,order=order,heading_policy=heading,reference_id='fixture_expected',sources=[])
                    for n in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw'):
                        self.assertEqual(getattr(ref,n).tobytes(),getattr(expected,n).tobytes())
                    self.assertEqual(report['strict_future_queries'],400)
                    self.assertTrue(report['first18_reference_future_feed_bytes_equal'])
                    self.assertFalse(report['execution_authority'])
                    self.assertEqual(ref.metadata['producer']['id'],'phrase-program-reference')
                    self.assertEqual(ref.metadata['context']['program_sha256'],report['program_sha256'])
                    self.assertEqual(set(report['source_reference_ids']),{'wave','high5'})
                    original={(s['path'],s['sha256']) for source in self.refs.values() for s in source.metadata['provenance']['sources']}
                    self.assertTrue(original.issubset({(s['path'],s['sha256']) for s in report['sources']}))
    def test_actual_upper_gain_range_violation_rejected_with_preserved_report(self):
        p=program(**{k:1.25 for k in GAIN_FIELDS})
        with self.assertRaises(ProgramCompileError) as context:compile_program(self.refs,p,reference_id='rejected',sources=[])
        self.assertEqual(context.exception.report['status'],'rejected_joint_ranges')
        self.assertIn('left_elbow_joint',context.exception.report['violations'])
        self.assertEqual(context.exception.report['strict_future_queries'],0)
        self.assertFalse(context.exception.report['execution_authority'])
    def test_reject_unknown_arm_names(self):
        names=list(self.refs['wave'].metadata['body']['joint_names']);names[names.index('left_elbow_joint')]='left_unknown_joint'
        with self.assertRaises(ValueError):_arm_indices(tuple(names))


if __name__=='__main__':unittest.main()
