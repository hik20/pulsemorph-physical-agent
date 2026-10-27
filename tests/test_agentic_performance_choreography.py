"""Pure synthetic waypoints. No body FK, model, controller or physics."""
import copy
import unittest
import numpy as np
from pulsemorph.agentic.performance.choreography_v1 import compile_program,ChoreographyError,MIN_SPANS,group,envelope

NAMES=[f'{side}_{joint}_joint' for side in ('left','right') for joint in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]+['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[f'{side}_{joint}_joint' for side in ('left','right') for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
def program(active=True):
    frames=[]
    for beat,sign in [(0,0),(7,1),(14,-1),(21,1),(28,0)]:
        frames.append({'beat_index':beat,'joint_targets_rad':{n:sign*MIN_SPANS[group(n)]/2 if active else 0. for n in NAMES}})
    return {'schema':'pulsemorph.performance_program.v1','music_sha256':'a'*64,'duration_s':16.8,'tempo_bpm':100,'keyframes':frames,'rationale':'Synthetic compiler fixture, no execution.'}
def compile(p,**kwargs):return compile_program(p,NAMES,[[-3.,3.]]*29,np.zeros(29),**kwargs)

class PerformanceCompilerTests(unittest.TestCase):
    def test_fullbody_reference_samples_and_horizon(self):
        c=compile(program());self.assertEqual(c.q.shape,(849,29));self.assertEqual(c.dq.shape,(849,29));self.assertTrue(c.report['numeric_validity_passed']);self.assertTrue(c.report['activation_qualified'])
        self.assertEqual(c.report['control_rows'],840);self.assertAlmostEqual(c.times[-1],16.96)
        for t in (0.,.02,4.2,8.4,16.8,16.96):
            q,dq=c.sample(t);self.assertTrue(np.isfinite(q).all());self.assertTrue(np.isfinite(dq).all())
        self.assertTrue(np.array_equal(c.q[0],np.zeros(29)));self.assertTrue(np.array_equal(c.dq[0],np.zeros(29)));self.assertTrue(np.array_equal(c.sample(16.96)[1],np.zeros(29)))
        with self.assertRaises(ChoreographyError):c.sample(17.)
    def test_pilot_mode_separates_validity_from_activation(self):
        with self.assertRaises(ChoreographyError) as error:compile(program(False))
        self.assertTrue(error.exception.report['numeric_validity_passed']);self.assertFalse(error.exception.report['activation_qualified'])
        c=compile(program(False),require_fullbody=False);self.assertTrue(c.report['admitted']);self.assertFalse(c.report['activation_qualified'])
    def test_no_clamp_initial_replacement_or_nonzero_dq(self):
        p=program();p['keyframes'][0]['joint_targets_rad'][NAMES[0]]=.001
        with self.assertRaisesRegex(ChoreographyError,'exactly'):compile(p)
        with self.assertRaises(ChoreographyError):compile_program(program(),NAMES,[[-3,3]]*29,np.zeros(29),np.ones(29)*.01)
        p=program();p['keyframes'][1]['joint_targets_rad']['waist_roll_joint']=.3
        with self.assertRaises(ChoreographyError) as error:compile(p)
        self.assertFalse(error.exception.report['numeric_validity_passed'])
    def test_continuous_speed_acceleration_and_xml_range(self):
        p=program();p['keyframes'].insert(1,{'beat_index':.5,'joint_targets_rad':{n:(1.5 if 'shoulder_pitch' in n else 0.) for n in NAMES}})
        with self.assertRaises(ChoreographyError) as error:compile(p)
        self.assertTrue(any(r['acceleration_max_rad_s2']>12 for r in error.exception.report['joint_ranges']))
        with self.assertRaises(ChoreographyError):compile_program(program(),NAMES,[[-.01,.01]]*29,np.zeros(29))
    def test_schema_music_missing_joints_bool_nan_duplicates(self):
        for kind in ('bool','nan','missing','order','music','extra'):
            p=program()
            if kind=='bool':p['keyframes'][1]['joint_targets_rad'][NAMES[0]]=True
            elif kind=='nan':p['keyframes'][1]['joint_targets_rad'][NAMES[0]]=float('nan')
            elif kind=='missing':del p['keyframes'][1]['joint_targets_rad'][NAMES[0]]
            elif kind=='order':p['keyframes'][1]['beat_index']=0
            elif kind=='music':p['music_sha256']='not a hash'
            else:p['execute_code']='forbidden'
            with self.subTest(kind=kind),self.assertRaises(ChoreographyError):compile(p)
    def test_every_root_envelope_named(self):
        expected={'left_shoulder_pitch_joint':1.6,'left_wrist_pitch_joint':.8,'waist_yaw_joint':.35,'waist_roll_joint':.18,'left_hip_pitch_joint':.45,'left_hip_roll_joint':.18,'left_hip_yaw_joint':.16,'left_knee_joint':.7,'left_ankle_pitch_joint':.35,'left_ankle_roll_joint':.15}
        for name,bound in expected.items():self.assertEqual(envelope(name),bound)
    def test_sample_derivative_and_knot_continuity_independently(self):
        c=compile(program());epsilon=1e-6
        for t in (.37,1.31,3.99,4.2,7.13,8.4,12.6,15.99):
            q,dq=c.sample(t);left=c.sample(t-epsilon);right=c.sample(t+epsilon)
            self.assertTrue(np.allclose((right[0]-left[0])/(2*epsilon),dq,rtol=0,atol=1e-7))
            self.assertTrue(np.max(np.abs(right[1]-left[1]))<1e-5)
if __name__=='__main__':unittest.main()
