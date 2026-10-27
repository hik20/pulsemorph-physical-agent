"""Pure algebra/ownership fixtures only; no real backend or physical calls."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.hybrid_arm_backend_v1 import applied_targets,root_up_z
class HybridMathTests(unittest.TestCase):
 def setUp(self):
  self.arms=np.arange(15,29);self.raw=np.linspace(-.3,.3,29).astype('float32');self.q=np.linspace(-.2,.2,29);self.dq=np.linspace(-.4,.4,29);self.bias=np.linspace(-2,2,29);self.kp=np.linspace(14,40,29);self.kd=np.linspace(.8,2.5,29)
 def test_positive_bias_velocity_formula_and_nonarm_exact(self):
  raw=self.raw.copy();applied=applied_targets(raw,self.q,self.dq,self.bias,self.kp,self.kd,self.arms)
  expected=self.q+self.kd/self.kp*self.dq+self.bias/self.kp
  np.testing.assert_array_equal(applied[self.arms],expected[self.arms].astype('float32'));self.assertEqual(applied[:15].tobytes(),raw[:15].tobytes());self.assertEqual(raw.tobytes(),self.raw.tobytes())
 def test_actuator_force_law_matches_desired_PD_plus_bias(self):
  actual_q=np.linspace(-.1,.1,29);actual_dq=np.linspace(-.2,.2,29);u=applied_targets(self.raw,self.q,self.dq,self.bias,self.kp,self.kd,self.arms).astype(float)
  force=self.kp*(u-actual_q)-self.kd*actual_dq;wanted=self.kp*(self.q-actual_q)+self.kd*(self.dq-actual_dq)+self.bias
  np.testing.assert_allclose(force[self.arms],wanted[self.arms],rtol=0,atol=1e-6)
 def test_no_hidden_clamp_or_gravity_sign_flip(self):
  q=np.zeros(29);dq=q.copy();bias=q.copy();bias[15]=100.;u=applied_targets(self.raw,q,dq,bias,np.ones(29)*10,np.ones(29),self.arms)
  self.assertEqual(u[15],10.);self.assertGreater(u[15],0.)
 def test_root_upright_formula_and_known_stop_contract(self):
  q=np.zeros(36);q[3]=1.;self.assertEqual(root_up_z(q),1.)
  q[3]=np.sqrt(.5);q[4]=np.sqrt(.5);self.assertAlmostEqual(root_up_z(q),0.)
  source=(ROOT/'src/pulsemorph/agentic/embodied/hybrid_arm_backend_v1.py').read_text()
  self.assertIn("raise RuntimeError('Actual physical finite/rootheight/warning stop')",source)
  self.assertIn('aggregate_joint_force_cap_violation',source)
 def test_invalid_arrays_and_gains_reject(self):
  for bad in (np.zeros(28,np.float32),np.ones(29,dtype='float64'),np.ones(29,np.float32)*float('nan')):
   with self.assertRaises(ValueError):applied_targets(bad,self.q,self.dq,self.bias,self.kp,self.kd,self.arms)
  for kp in (np.zeros(29),np.ones(29)*float('inf')):
   with self.assertRaises(ValueError):applied_targets(self.raw,self.q,self.dq,self.bias,kp,self.kd,self.arms)
  with self.assertRaises(ValueError):applied_targets(self.raw,self.q,self.dq,self.bias,self.kp,self.kd,[15]*14)
if __name__=='__main__':unittest.main()
