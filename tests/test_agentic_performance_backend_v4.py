import inspect
import unittest
import numpy as np
from pulsemorph.agentic.performance.backend_v4 import mixed_targets,PerformanceBackend
from pulsemorph.agentic.performance.backend_v2 import direct_targets

class Mixed18Tests(unittest.TestCase):
 def arrays(self):
  return [np.arange(29,dtype=np.float32),np.linspace(-.2,.2,29),np.ones(29),np.ones(29)*2,np.ones(29)*10,np.ones(29)]
 def test18_formula11_actor_exact(self):
  args=self.arrays();indices=list(range(18));out=mixed_targets(*args,indices)
  np.testing.assert_array_equal(out[:18],direct_targets(*args)[:18])
  np.testing.assert_array_equal(out[18:],args[0][18:])
 def test_no_alias_no_clamp(self):
  args=self.arrays();args[3][:]=1000
  out=mixed_targets(*args,list(range(18)))
  self.assertGreater(float(out[0]),90)
  self.assertFalse(np.shares_memory(out,args[0]))
 def test_invalid_roles(self):
  for indices in (list(range(17)),[0]*18,list(range(17))+[29],np.arange(18,dtype=float)):
   with self.assertRaises(ValueError):mixed_targets(*self.arrays(),indices)
 def test_history_and_actor_guard(self):
  s=inspect.getsource(PerformanceBackend._advance_mixed)
  self.assertIn('self.prev_targets=applied.copy()',s)
  self.assertIn('applied[self.actor_controlled_indices]',s)
  self.assertIn('self.model.jnt_actfrclimited',s)
  self.assertNotIn('data.qpos[:]',s)
 def test_retains_declared_outerloop(self):
  s=inspect.getsource(PerformanceBackend.advance)
  self.assertIn('tilt_request',s);self.assertIn('self._advance_mixed(q, dq)',s)
  self.assertIn('finally:',s);self.assertIn('self.anchor_target = nominal',s)

if __name__=='__main__':unittest.main()
