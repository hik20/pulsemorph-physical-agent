"""Reference-only phrase fixtures: no model inference, pickle or physics."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.phrase_reference import compose_phrase,plain
from pulsemorph.motion.contracts import load_reference,MotionReference
from pulsemorph.motion.g1 import sample_reference,assemble_reference_inputs


class PhraseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'artifacts/runs/agentic/app/jobs'
        cls.wave=load_reference(base/'20261027T023422_8efe4bc4/trials/baseline/common_reference')
        cls.high=load_reference(base/'20261027T022322_76db8965/trials/baseline/common_reference')
        cls.phrase=compose_phrase(cls.wave,cls.high,reference_id='fixture_phrase',sources=[])
    def test_exact_original_prefix_inputs(self):
        for i in (0,1,50,100,166,167):
            a=assemble_reference_inputs(self.wave,i*.02);b=assemble_reference_inputs(self.phrase,i*.02)
            for k in a:self.assertEqual(a[k].tobytes(),b[k].tobytes())
    def test_joint_joins_match_position_and_derivative(self):
        for t,src,clock in ((3.5,self.wave,3.5),(4.5,self.high,.5)):
            a=sample_reference(self.phrase,np.array([t]));b=sample_reference(src,np.array([clock]))
            np.testing.assert_array_equal(a['dof_pos'],b['dof_pos'])
            np.testing.assert_array_equal(a['dof_vel'],b['dof_vel'])
            curve=CubicHermiteSpline(self.phrase.time_offsets_s,self.phrase.joint_position_rad,self.phrase.joint_velocity_rad_s)
            for eps in (1e-6,1e-8):
                np.testing.assert_allclose(curve(t-eps,1),curve(t+eps,1),atol=eps*100,rtol=0)
    def test_exact_bridge_is_endpoint_hermite(self):
        a=sample_reference(self.wave,np.array([3.5]));b=sample_reference(self.high,np.array([.5]))
        expected=CubicHermiteSpline([3.5,4.5],np.vstack([a['dof_pos'],b['dof_pos']]),np.vstack([a['dof_vel'],b['dof_vel']]))
        t=np.array([3.501,3.713,4.123,4.499]);actual=sample_reference(self.phrase,t)
        np.testing.assert_allclose(actual['dof_pos'],expected(t),atol=1e-13,rtol=0)
        np.testing.assert_allclose(actual['dof_vel'],expected(t,1),atol=1e-12,rtol=0)
    def test_high5_tail_and_no_extrapolation(self):
        self.assertEqual(self.phrase.time_offsets_s[-1],8.14)
        a=sample_reference(self.phrase,np.array([5.,6.,8.14]));b=sample_reference(self.high,np.array([1.,2.,4.14]))
        np.testing.assert_allclose(a['dof_pos'],b['dof_pos'],atol=1e-13,rtol=0)
        for t in (-.01,8.141,float('nan')):
            with self.assertRaises(ValueError):sample_reference(self.phrase,np.array([t]))
    def test_reject_mismatched_body_controller_frame(self):
        for group,key,value in (('body','profile_sha256','0'*64),('controller','profile_sha256','0'*64),('frame','id','other_world')):
            metadata=plain(self.high.metadata);metadata[group][key]=value
            altered=MotionReference(metadata,self.high.time_offsets_s,self.high.joint_position_rad,self.high.joint_velocity_rad_s,self.high.anchor_orientation_xyzw)
            with self.assertRaisesRegex(ValueError,'mismatch'):compose_phrase(self.wave,altered,reference_id='bad',sources=[])
    def test_torso_endpoints_and_normalization_only_c0(self):
        rot=sample_reference(self.phrase,np.array([3.5,4.,4.5]))['anchor_rot']
        self.assertTrue(np.allclose(np.linalg.norm(rot,axis=1),1,atol=1e-12))
        expected=np.vstack([sample_reference(self.wave,np.array([3.5]))['anchor_rot'],sample_reference(self.high,np.array([.5]))['anchor_rot']])
        delta=(Rotation.from_quat(rot[[0,2]])*Rotation.from_quat(expected).inv()).magnitude()
        np.testing.assert_allclose(delta,0,atol=1e-12)
        self.assertTrue(any('C0' in s for s in self.phrase.metadata['provenance']['notes']))


if __name__=='__main__':unittest.main()
