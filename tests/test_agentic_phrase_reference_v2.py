"""Ordered-pair/heading reference-only fixtures; no physical or ONNX calls."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/vendor/ProtoMotions')]
from pulsemorph.agentic.phrase_reference_v2 import compose_ordered_pair
from pulsemorph.agentic.phrase_reference import plain
from pulsemorph.motion.contracts import load_reference,MotionReference
from pulsemorph.motion.g1 import sample_reference
from deployment.state_utils import compute_yaw_offset_np,apply_heading_offset_np,_extract_yaw_quat_np


class PhraseV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'artifacts/runs/agentic/app/jobs'
        cls.refs={n:load_reference(base/j/'trials/baseline/common_reference') for n,j in {'wave':'20261027T023422_8efe4bc4','high5':'20261027T022322_76db8965'}.items()}
        cls.aligned,cls.transform=compose_ordered_pair(cls.refs,order=('high5','wave'),heading_policy='align_join_heading',reference_id='fixture_aligned',sources=[])
    def compose(self,order=('wave','high5'),policy='recorded_world'):
        return compose_ordered_pair(self.refs,order=order,heading_policy=policy,reference_id='fixture',sources=[])[0]
    def test_recorded_world_exact_v1_regression(self):
        old=load_reference(ROOT/'artifacts/runs/agentic/phrase_reference_plan_v1/common_reference');new=self.compose()
        for name in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw'):
            self.assertEqual(getattr(old,name).tobytes(),getattr(new,name).tobytes())
    def test_heading_only_preserves_local_q_dq_and_clocks(self):
        recorded=self.compose(('high5','wave'))
        for name in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s'):
            self.assertEqual(getattr(recorded,name).tobytes(),getattr(self.aligned,name).tobytes())
        self.assertFalse(np.array_equal(recorded.anchor_orientation_xyzw,self.aligned.anchor_orientation_xyzw))
        self.assertNotIn('root_position',self.transform)
    def test_exact_pinned_heading_operation_and_entry_yaw(self):
        first=sample_reference(self.refs['high5'],np.array([3.5]))['anchor_rot'][0].astype('float32')
        second=sample_reference(self.refs['wave'],np.array([.5]))['anchor_rot'][0].astype('float32')
        offset=compute_yaw_offset_np(first,second)
        np.testing.assert_array_equal(np.array(self.transform['offset_xyzw'],dtype='float32'),offset)
        actual=sample_reference(self.aligned,np.array([4.5]))['anchor_rot'][0]
        expected=Rotation.from_quat(apply_heading_offset_np(offset,second)).as_quat()
        self.assertLess(float((Rotation.from_quat(actual)*Rotation.from_quat(expected).inv()).magnitude()),1e-7)
        yaw_delta=(Rotation.from_quat(_extract_yaw_quat_np(first))*Rotation.from_quat(_extract_yaw_quat_np(actual.astype('float32'))).inv()).magnitude()
        self.assertLess(float(yaw_delta),1e-7)
    def test_prefix_and_exhaustion(self):
        times=np.array([0.,.8,1.6,3.34,3.5]);a=sample_reference(self.aligned,times);b=sample_reference(self.refs['high5'],times)
        for key in ('dof_pos','dof_vel'):np.testing.assert_array_equal(a[key],b[key])
        # The actual tracker boundary is float32. Repeated Rotation/SLERP unit
        # normalization can differ at float64 machine precision, not that feed.
        np.testing.assert_array_equal(a['anchor_rot'].astype('float32'),b['anchor_rot'].astype('float32'))
        self.assertEqual(self.aligned.time_offsets_s[-1],8.14)
        with self.assertRaises(ValueError):sample_reference(self.aligned,np.array([8.15]))
    def test_invalid_pair_policy_and_frame(self):
        for order in (('wave','wave'),('walk','wave'),('wave',)):
            with self.assertRaises(ValueError):self.compose(order)
        with self.assertRaises(ValueError):self.compose(policy='automatic')
        metadata=plain(self.refs['wave'].metadata);metadata['frame']['id']='different_world'
        ref=self.refs['wave'];bad=MotionReference(metadata,ref.time_offsets_s,ref.joint_position_rad,ref.joint_velocity_rad_s,ref.anchor_orientation_xyzw)
        with self.assertRaisesRegex(ValueError,'frame mismatch'):compose_ordered_pair({**self.refs,'wave':bad},order=('high5','wave'),heading_policy='align_join_heading',reference_id='bad',sources=[])
    def test_c1_joints_but_only_c0_torso_claim(self):
        for execution,source,clock in ((3.5,'high5',3.5),(4.5,'wave',.5)):
            a=sample_reference(self.aligned,np.array([execution]));b=sample_reference(self.refs[source],np.array([clock]))
            for k in ('dof_pos','dof_vel'):np.testing.assert_array_equal(a[k],b[k])
        self.assertTrue(any('C0' in n for n in self.aligned.metadata['provenance']['notes']))


if __name__=='__main__':unittest.main()
