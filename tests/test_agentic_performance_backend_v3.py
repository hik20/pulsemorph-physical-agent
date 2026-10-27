import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from pulsemorph.agentic.performance.backend_v3 import (PerformanceBackend, NominalBackend,
    CONFIG,tilt_request,applied_anchor)
from pulsemorph.agentic.embodied.runtime_v1 import canonical

class OuterLoopTests(unittest.TestCase):
 def backend(self):
  b=PerformanceBackend.__new__(PerformanceBackend)
  b.authority={'max_control_steps':100};b._performance_authority_sha=hashlib.sha256(canonical(b.authority)).hexdigest()
  b.data=SimpleNamespace(qpos=np.zeros(36),qvel=np.zeros(35),time=0.)
  b.anchor_target=np.array([0.,0.,0.,1.]);b.root_xy_start=np.zeros(2)
  b.outer_tilt=np.zeros(2);b.outer_config=dict(CONFIG);b.outer_rows=[]
  b.joint_ranges=np.tile([-1.,1.],(29,1));b.stopped=False;b.control_index=0
  return b
 def test_signs_rate_and_saturation(self):
  previous=np.zeros(2)
  for _ in range(100):
   result,raw=tilt_request([10,10],[0,0],previous)
   self.assertLessEqual(np.max(np.abs(result-previous)),.004000000001)
   self.assertLessEqual(np.max(np.abs(result)),.120000000001)
   self.assertGreaterEqual(result[0],0);self.assertLessEqual(result[1],0)
   previous=result
 def test_world_quaternion_order(self):
  a=applied_anchor([0,0,0,1],[.1,-.2])
  self.assertAlmostEqual(np.linalg.norm(a),1)
  self.assertGreater(a[0],0);self.assertLess(a[1],0)
  np.testing.assert_array_equal(applied_anchor([0,0,0,1],[0,0]),[0,0,0,1])
 def test_invalid_data(self):
  with self.assertRaises(ValueError):tilt_request([np.nan,0],[0,0],[0,0])
  with self.assertRaises(ValueError):applied_anchor([0,0,0,2],[0,0])
 def test_nominal_restored_on_failure_joint_refs_unchanged(self):
  b=self.backend();b.data.qpos[0]=1;nominal=b.anchor_target.copy()
  q=np.zeros((4,29));dq=q.copy()
  def fail(obj,a,v):
   self.assertFalse(np.array_equal(obj.anchor_target,nominal))
   np.testing.assert_array_equal(a,q);np.testing.assert_array_equal(v,dq)
   raise RuntimeError('fixture failure')
  with patch.object(NominalBackend,'advance',fail):
   with self.assertRaises(RuntimeError):b.advance(q,dq)
  np.testing.assert_array_equal(b.anchor_target,nominal)
  self.assertEqual(len(b.outer_rows),1)
 def test_filter_state_changes_closure_and_bad_future_no_mutation(self):
  b=self.backend()
  with patch.object(NominalBackend,'closure_sha256',return_value='a'*64):
   before=b.closure_sha256();b.outer_tilt[0]=.01
   self.assertNotEqual(before,b.closure_sha256())
  with self.assertRaises(ValueError):b.advance(np.full((4,29),2.),np.zeros((4,29)))
  self.assertEqual(b.outer_rows,[])

if __name__=='__main__':unittest.main()
