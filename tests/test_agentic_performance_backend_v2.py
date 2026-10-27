import ast
import inspect
import unittest
import numpy as np
from pulsemorph.agentic.performance.backend_v2 import direct_targets,PerformanceBackend

class DirectServoTests(unittest.TestCase):
 def test_all29_formula(self):
  q=np.linspace(-.3,.3,29);dq=np.linspace(-1,1,29)
  kp=np.arange(29)+20.;kd=np.arange(29)/10;bias=np.arange(29)-14.
  actual=direct_targets(np.full(29,99,np.float32),q,dq,bias,kp,kd)
  np.testing.assert_array_equal(actual,(q+kd/kp*dq+bias/kp).astype(np.float32))
  self.assertEqual(actual.dtype,np.float32)
 def test_raw_actor_not_actuation(self):
  args=[np.zeros(29),np.ones(29),np.ones(29),np.full(29,10.),np.ones(29)]
  a=direct_targets(np.zeros(29,np.float32),*args)
  b=direct_targets(np.full(29,12,np.float32),*args)
  np.testing.assert_array_equal(a,b)
 def test_no_clamp_positive_bias(self):
  result=direct_targets(np.zeros(29,np.float32),np.ones(29),np.zeros(29),np.full(29,40.),np.full(29,10.),np.zeros(29))
  np.testing.assert_array_equal(result,np.full(29,5,np.float32))
 def test_bad_values(self):
  good=[np.zeros(29,np.float32),np.zeros(29),np.zeros(29),np.zeros(29),np.ones(29),np.ones(29)]
  for index,value in [(0,np.zeros(29,np.float64)),(1,np.zeros(28)),(2,np.full(29,np.nan)),(4,np.zeros(29)),(5,np.full(29,-1.))]:
   args=list(good);args[index]=value
   with self.assertRaises(ValueError):direct_targets(*args)
 def test_applied_history_and_existing_step_screen_plumbing(self):
  source=inspect.getsource(PerformanceBackend.advance)
  self.assertIn('self.prev_targets=applied.copy()',source)
  self.assertIn('self.data.ctrl[:]=applied',source)
  self.assertIn('range(20)',source)
  self.assertIn('self.model.jnt_actfrclimited',source)
  self.assertIn("raise RuntimeError('Actual physical finite/rootheight/warning stop')",source)
  self.assertNotIn('data.qpos[:]',source)

if __name__=='__main__':unittest.main()
