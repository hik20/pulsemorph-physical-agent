import unittest
import numpy as np
from pulsemorph.agentic.performance.wholebody_qp_v1 import constrained_ls,wrench_inequalities,solve_wholebody,WBCInfeasible

class WBCTests(unittest.TestCase):
 def test_hard_torque_bound_not_clamp(self):
  x,r=constrained_ls(np.eye(1),[10.],np.zeros((0,1)),[],[[1.],[-1.]],[2.,2.])
  self.assertAlmostEqual(x[0],2.)
  self.assertLess(r['stationarity_residual'],1e-5)
 def test_equality_and_nonnegative_support(self):
  x,r=constrained_ls(np.eye(2),[50.,50.],[[1.,1.]],[100.],[[-1.,0.],[0.,-1.]],[0.,0.])
  np.testing.assert_allclose(x,[50.,50.])
 def test_infeasible_rejects(self):
  with self.assertRaises(WBCInfeasible):
   constrained_ls(np.eye(1),[0.],[[1.]],[3.],[[1.]],[2.])
 def test_nested_bounds_false_infeasible_regression(self):
  x,r=constrained_ls([[1.]],[2.],np.zeros((0,1)),[],[[1.],[.1]],[1.,.05])
  self.assertAlmostEqual(x[0],.5,places=6)
  self.assertEqual(r['solver'],'scipy_SLSQP_analytic_gradient_linear_Jacobian')
 def test_friction_and_cop_shift(self):
  g=wrench_inequalities([np.eye(3),np.eye(3)])
  f=np.zeros(12);f[[2,8]]=100
  self.assertLessEqual(np.max(g@f),0)
  f[0]=61
  self.assertGreater(np.max(g@f),0)
  f[0]=0;f[4]=-14 # xCoP=.14 beyond+.135
  self.assertGreater(np.max(g@f),0)
 def test_static_root_balance(self):
  m=np.eye(35);b=np.zeros(35);b[2]=100
  j=np.zeros((12,35));j[:6,:6]=np.eye(6);j[6:,:6]=np.eye(6)
  prior=np.zeros(12);prior[[2,8]]=50
  tau,r=solve_wholebody(m,b,j,np.zeros(12),np.zeros(35),prior,[np.eye(3),np.eye(3)],np.tile([-100,100],(29,1)))
  self.assertLess(np.max(np.abs(r['root_dynamics_residual'])),1e-5)
  self.assertLess(np.max(np.abs(tau)),1e-5)

if __name__=='__main__':unittest.main()
