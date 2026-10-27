import unittest
import numpy as np
from pulsemorph.agentic.performance.wholebody_qp_v2 import solve_wholebody,MIN_LOCAL_FZ_N,WBCInfeasible

class MinimumLoadTests(unittest.TestCase):
 def solve(self,weight,prior):
  m=np.eye(35);b=np.zeros(35);b[2]=weight
  j=np.zeros((12,35));j[:6,:6]=np.eye(6);j[6:,:6]=np.eye(6)
  return solve_wholebody(m,b,j,np.zeros(12),np.zeros(35),prior,[np.eye(3),np.eye(3)],np.tile([-100,100],(29,1)))
 def test_stand_and_eachfoot_min15(self):
  prior=np.zeros(12);prior[2]=100
  _,r=self.solve(100,prior)
  self.assertEqual(r['minimum_local_normal_force_N'],15.)
  self.assertTrue(all(x>=15.-1e-6 for x in r['local_normal_forces_N']))
  self.assertAlmostEqual(sum(r['local_normal_forces_N']),100.,places=5)
 def test_insufficient_totalweight_rejected(self):
  with self.assertRaises(WBCInfeasible):self.solve(20,np.zeros(12))

if __name__=='__main__':unittest.main()
