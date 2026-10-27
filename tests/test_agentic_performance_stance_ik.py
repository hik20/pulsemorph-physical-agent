import unittest
import numpy as np
from pulsemorph.agentic.performance.stance_ik_v1 import damped_step,rotation_error,body_tasks,residual_passed,TOLERANCES

class StanceMathTests(unittest.TestCase):
 def test_zero_residual_zero_update(self):
  np.testing.assert_array_equal(damped_step(np.eye(18),np.zeros(18)),np.zeros(18))
 def test_damping_rank_deficient(self):
  x=damped_step(np.zeros((18,18)),np.ones(18))
  np.testing.assert_array_equal(x,np.zeros(18))
 def test_direction(self):
  x=damped_step(np.eye(3),np.array([1.,-1.,0.]))
  self.assertGreater(x[0],0);self.assertLess(x[1],0)
 def test_rotation_identity_and_small_angle(self):
  np.testing.assert_array_equal(rotation_error(np.eye(3),np.eye(3)),np.zeros(3))
  a=.1;r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
  self.assertAlmostEqual(rotation_error(r,np.eye(3))[2],a)
 def test_pi_orientation_cannot_false_converge(self):
  r=np.diag([1.,-1.,-1.])
  self.assertAlmostEqual(np.linalg.norm(rotation_error(r,np.eye(3))),np.pi)
 def test_body_task_hold_and_bounded_requests(self):
  names=['left_hip_pitch_joint','right_hip_pitch_joint','left_hip_roll_joint','right_hip_roll_joint','left_knee_joint','right_knee_joint','waist_roll_joint']
  xy,z=body_tasks(np.zeros(7),names,[1.,2.,3.],.8)
  np.testing.assert_array_equal(xy,[1.,2.]);self.assertEqual(z,.8)
  xy,z=body_tasks(np.ones(7)*10,names,[1.,2.,3.],.8)
  np.testing.assert_allclose(xy,[1.06,2.06]);self.assertAlmostEqual(z,.7)
 def test_nonfinite_solver_inputs(self):
  with self.assertRaises(ValueError):damped_step(np.full((3,3),np.nan),np.zeros(3))
 def test_submillimeter_continuity_tolerances(self):
  exact={'left_position_m':0.,'right_position_m':0.,'left_orientation_rad':0.,'right_orientation_rad':0.,'com_xy_m':0.,'root_z_m':0.,'root_orientation_rad':0.}
  self.assertTrue(residual_passed(exact))
  for name,limit in [('left_position_m',.0005),('com_xy_m',.0005),('root_z_m',.0005),('left_orientation_rad',.003),('root_orientation_rad',.003)]:
   row=dict(exact);row[name]=limit*1.01;self.assertFalse(residual_passed(row))
  self.assertEqual(TOLERANCES['root_z_m'],.0005)

if __name__=='__main__':unittest.main()
