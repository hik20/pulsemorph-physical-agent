"""Synthetic pure reference tests; no vendor assets/controller/model/FK required."""
import copy,hashlib,sys,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.motion.contracts import MotionReference
from pulsemorph.motion.g1 import sample_reference
from pulsemorph.agentic import phrase_expression_v1 as e
from pulsemorph.agentic.phrase_program import _gain_source
from pulsemorph.agentic import phrase_program as legacy
from pulsemorph.agentic.app.exports import JOINT_NAMES
def expression(**changes):return {'schema':e.SCHEMA,'order':['wave','high5'],'heading_policy':'recorded_world',**{k:1. for k in e.GAIN_FIELDS},'rationale':'Synthetic reference fixture, not physics.',**changes}
def reference(profile=None,constant=False):
    p=profile or {'body_sha256':'0'*64,'yaml_sha256':'0'*64};t=np.arange(208,dtype=float)*.02
    q=np.tile((.1*t+.02*np.sin(2*t))[:,None],(1,29));dq=np.tile((.1+.04*np.cos(2*t))[:,None],(1,29))
    if constant:q[:]=0;dq[:]=0
    metadata={'schema_version':'pulsemorph.motion_reference.v1','reference_id':'synthetic_source','producer':{'kind':'baseline','id':'synthetic','version':'1'},'body':{'profile_id':'fixture','profile_sha256':p['body_sha256'],'joint_names':list(JOINT_NAMES),'anchor_body_name':'torso_link'},'controller':{'profile_id':'fixture','profile_sha256':p['yaml_sha256'],'control_dt_s':.02},'frame':{'id':'reference_world','up_axis':'z','handedness':'right','quaternion_order':'xyzw','alignment':'recorded_reference_world'},'context':{'decision_id':'fixture','observation_id':'fixture','task_id':'fixture','runtime_session_id':'fixture','clock_epoch':'fixture','decision_time_s':0.},'validity':{'start_state_id':'fixture','start_time_s':0.,'end_time_s':4.14,'start_mode':'replace_unexecuted','end_behavior':'reject'},'observation':{'mode':'offline_file','observed_through_time_s':0.,'audio_observed_through_sample':None,'video_observed_through_time_s':None},'provenance':{'sources':[],'notes':['Synthetic, not observed motion.']}}
    return MotionReference(metadata,t,q,dq,np.tile([0.,0.,0.,1.],(208,1)))
class ExpressionTests(unittest.TestCase):
    def test_exact_eight_fields_owned_and_bounded(self):
        p=expression(first_left_pose_gain=.75,second_right_motion_gain=1.25);parsed=e.parse_expression(p);self.assertEqual(len(e.GAIN_FIELDS),8);parsed['order'].reverse();self.assertEqual(p['order'],['wave','high5'])
        for bad in (True,None,'1',float('nan'),float('inf'),10**1000,.7499,1.2501):
            with self.subTest(bad=str(bad)[:30]),self.assertRaises(ValueError):e.parse_expression(expression(first_left_motion_gain=bad))
        for bad in (expression(extra=1),expression(order=['wave','wave']),expression(rationale='\ud800'),expression(heading_policy='auto')):
            with self.assertRaises(ValueError):e.parse_expression(bad)
    def test_equal_pair_exact_legacy_numeric_bytes_and_unit_identity(self):
        source=reference();edited,report=e.edit_source_expression(source,.85,.85,1.15,1.15);expected=_gain_source(source,.85,1.15)
        for field in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw'):self.assertEqual(getattr(edited,field).tobytes(),getattr(expected,field).tobytes())
        unit,_=e.edit_source_expression(source,1.,1.,1.,1.);self.assertIs(unit,source)
    def test_product_rule_prefix_nonarm_torso_and_mixed_legacy_route(self):
        source=reference();edited,report=e.edit_source_expression(source,1.2,.8,.9,.9);groups=e._arm_indices(JOINT_NAMES);left,right=groups;t=source.time_offsets_s;mask=t<=.5
        for key in ('joint_position_rad','joint_velocity_rad_s'):self.assertEqual(getattr(edited,key)[mask].tobytes(),getattr(source,key)[mask].tobytes())
        nonarm=np.array([j for j in range(29) if j not in np.r_[left,right]])
        for key in ('joint_position_rad','joint_velocity_rad_s'):self.assertEqual(getattr(edited,key)[:,nonarm].tobytes(),getattr(source,key)[:,nonarm].tobytes())
        self.assertEqual(edited.anchor_orientation_xyzw.tobytes(),source.anchor_orientation_xyzw.tobytes())
        legacy=_gain_source(source,1.,.9);self.assertEqual(edited.joint_position_rad[:,right].tobytes(),legacy.joint_position_rad[:,right].tobytes())
        i=38;u=(t[i]-.5)/.5;w=3*u*u-2*u**3;wd=(6*u-6*u*u)/.5;mu=np.array(report['arms']['left']['mean_rad']);q0=source.joint_position_rad[0,left];delta=.2*(mu-q0)-.2*(source.joint_position_rad[i,left]-mu)
        expected=source.joint_velocity_rad_s[i,left]+w*(-.2)*source.joint_velocity_rad_s[i,left]+wd*delta
        np.testing.assert_allclose(edited.joint_velocity_rad_s[i,left],expected,rtol=0,atol=1e-14)
        times=np.array([.731,1.213,2.111]);eps=1e-6;fd=(sample_reference(edited,times+eps)['dof_pos']-sample_reference(edited,times-eps)['dof_pos'])/(2*eps);np.testing.assert_allclose(fd,sample_reference(edited,times)['dof_vel'],rtol=0,atol=1e-8)
    def test_fully_ramped_pose_and_motion_are_separate(self):
        source=reference();pose,inspection=e.edit_source_expression(source,1.2,1.,1.,1.);motion,_=e.edit_source_expression(source,1.,.8,1.,1.);left=e._arm_indices(JOINT_NAMES)[0];keep=(source.time_offsets_s>=1.2-1e-10)&(source.time_offsets_s<=3.2+1e-10);original=source.joint_position_rad[keep][:,left];mu=original.mean(axis=0)
        np.testing.assert_allclose(pose.joint_position_rad[keep][:,left]-pose.joint_position_rad[keep][:,left].mean(axis=0),original-mu,atol=1e-15,rtol=0)
        np.testing.assert_allclose(motion.joint_position_rad[keep][:,left].mean(axis=0),mu,atol=1e-15,rtol=0)
        np.testing.assert_allclose(motion.joint_position_rad[keep][:,left]-mu,.8*(original-mu),atol=1e-15,rtol=0)
    def test_zero_basis_inspection_and_bad_clock(self):
        source=reference(constant=True);report=e.inspect_source_expression(source);self.assertTrue(report['arms']['left']['pose_basis_exact_zero']);self.assertTrue(report['arms']['left']['motion_basis_exact_zero']);self.assertIsNone(report['arms']['left']['basis_norm_condition_number'])
        bad=reference();times=bad.time_offsets_s.copy();times[40]+=.0001;object.__setattr__(bad,'time_offsets_s',times)
        with self.assertRaises(ValueError):e.inspect_source_expression(bad)
    def test_synthetic_compile_range_guard_and_400_strict_prefix(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);body=root/'body';yaml=root/'yaml';body.write_bytes(b'synthetic profile');yaml.write_bytes(b'synthetic control')
            p={'joint_names':JOINT_NAMES,'anchor_body_name':'torso_link','control_dt_s':.02,'body_sha256':hashlib.sha256(body.read_bytes()).hexdigest(),'yaml_sha256':hashlib.sha256(yaml.read_bytes()).hexdigest(),'body_path':body,'yaml_path':yaml};ref=reference(p);refs={'wave':ref,'high5':ref}
            stack.enter_context(patch.object(e,'profile_from_yaml',return_value=p));stack.enter_context(patch('pulsemorph.motion.g1.profile_from_yaml',return_value=p));ranges=stack.enter_context(patch.object(e,'_continuous_ranges',return_value=[{'joint':'fixture','within_unchanged_range':True}]))
            target,report=e.compile_expression(refs,expression(first_left_pose_gain=1.1,first_left_motion_gain=.9),reference_id='synthetic_expression',sources=[])
            self.assertEqual(report['strict_future_queries'],400);self.assertTrue(report['first18_reference_future_feed_bytes_equal']);self.assertFalse(report['execution_authority']);self.assertEqual(target.metadata['producer']['id'],'phrase-expression-reference');self.assertEqual(report['new_FK_calls'],0)
            stack.enter_context(patch.object(legacy,'profile_from_yaml',return_value=p));stack.enter_context(patch.object(legacy,'_continuous_ranges',return_value=[{'joint':'fixture','within_unchanged_range':True}]))
            equal=expression(first_left_pose_gain=.9,first_left_motion_gain=.9,second_right_pose_gain=1.1,second_right_motion_gain=1.1)
            new_equal,_=e.compile_expression(refs,equal,reference_id='synthetic_equal_expression',sources=[])
            old_program={'schema':legacy.SCHEMA,'order':equal['order'],'heading_policy':equal['heading_policy'],'first_left_gain':.9,'first_right_gain':1.,'second_left_gain':1.,'second_right_gain':1.1,'rationale':'synthetic parity'}
            old_equal,_=legacy.compile_program(refs,old_program,reference_id='synthetic_equal_legacy',sources=[])
            for key in ('time_offsets_s','joint_position_rad','joint_velocity_rad_s','anchor_orientation_xyzw'):self.assertEqual(getattr(new_equal,key).tobytes(),getattr(old_equal,key).tobytes())
            ranges.return_value=[{'joint':'fixture','within_unchanged_range':False}]
            with self.assertRaises(e.ProgramCompileError) as caught:e.compile_expression(refs,expression(),reference_id='synthetic_reject',sources=[])
            self.assertEqual(caught.exception.report['status'],'rejected_joint_ranges');self.assertEqual(caught.exception.report['strict_future_queries'],0)
if __name__=='__main__':unittest.main()
