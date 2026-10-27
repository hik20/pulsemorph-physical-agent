import unittest
import inspect
import numpy as np
from pulsemorph.agentic.performance.backend_v6 import vertical_support,PerformanceBackend

class ContactSupportTests(unittest.TestCase):
 def test_static_weight_removed_from_joint_feedforward(self):
  j=np.zeros((2,35));j[:,2]=1;j[0,6]=.5;j[1,7]=.5
  b=np.zeros(35);b[2]=100;b[6:8]=25
  tau,r=vertical_support(j,b)
  self.assertAlmostEqual(sum(r['vertical_forces_N']),100)
  self.assertAlmostEqual(r['root_wrench_residual_norm'],0)
  # NNLS may choose either contact; verify exact predicted support identity.
  f=np.asarray(r['vertical_forces_N'])
  np.testing.assert_allclose(tau,b[6:]-j[:,6:].T@f)
 def test_negative_force_not_invented(self):
  j=np.zeros((1,35));j[0,2]=1;b=np.zeros(35);b[2]=-10
  tau,r=vertical_support(j,b)
  self.assertEqual(r['vertical_forces_N'],[0.])
  self.assertEqual(r['root_wrench_residual_norm'],10.)
 def test_moment_balance(self):
  j=np.zeros((2,35));j[:,2]=1;j[:,3]=[-.1,.1]
  b=np.zeros(35);b[2]=100;b[3]=2
  _,r=vertical_support(j,b)
  np.testing.assert_allclose(r['vertical_forces_N'],[40,60],atol=1e-10)
 def test_all24_contacts_retained(self):
  j=np.zeros((24,35));j[:,2]=1;j[:14,3]=-.1;j[14:,3]=.1
  b=np.zeros(35);b[2]=100;b[3]=2
  _,r=vertical_support(j,b)
  self.assertEqual(len(r['vertical_forces_N']),24)
  self.assertAlmostEqual(r['root_wrench_residual_norm'],0.,places=8)
 def test_invalid(self):
  for j in (np.zeros((0,35)),np.zeros((65,35)),np.full((2,35),np.nan)):
   with self.assertRaises(ValueError):vertical_support(j,np.zeros(35))
 def test_actual_bias_not_mutated_and_appliedhistory(self):
  src=inspect.getsource(PerformanceBackend._advance_contact)
  self.assertIn('self.prev_targets=applied.copy()',src)
  self.assertIn('direct_targets(raw,q,dq,tau_ff,self.kp,self.kd)',src)
  self.assertNotIn('self.data.qfrc_bias[:]=',src)
  self.assertNotIn('self.data.qpos[:]=',src)

if __name__=='__main__':unittest.main()
