import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from pulsemorph.agentic.performance.backend_v1 import (
    PerformanceBackend, SCHEMA, validate_authority, validate_future)
from pulsemorph.agentic.embodied.hybrid_arm_backend_v1 import HybridArmBackend


class PerformanceBackendTests(unittest.TestCase):
    def authority(self, cap=1200):
        import pulsemorph.agentic.performance.backend_v1 as module
        p = str(Path(module.__file__).resolve())
        return {'schema': SCHEMA, 'session_id': 'fixture', 'max_control_steps': cap,
                'source_pins': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}}

    def test_authority_bounds_and_ownership(self):
        for cap in (True, 0, 1201, 1.5):
            with self.assertRaises(ValueError): validate_authority(self.authority(cap))
        a = self.authority(); b = validate_authority(a)
        a['source_pins'].clear()
        self.assertTrue(b['source_pins'])

    def test_all29_future_values_preserved_and_owned(self):
        q = np.arange(116).reshape(4, 29) / 1000
        dq = q * 2
        out, vel = validate_future(q, dq, np.tile([-1, 1], (29, 1)))
        np.testing.assert_array_equal(out, q)
        np.testing.assert_array_equal(vel, dq)
        self.assertFalse(np.shares_memory(out, q))

    def test_invalid_future_rejected(self):
        for q, dq in [(np.zeros((3,29)),np.zeros((4,29))),
                      (np.full((4,29),np.nan),np.zeros((4,29))),
                      (np.full((4,29),1.01),np.zeros((4,29))),
                      (np.zeros((4,29)),np.full((4,29),2.01))]:
            with self.assertRaises(ValueError):
                validate_future(q,dq,np.tile([-1,1],(29,1)))

    def test_bootstrap_is_bounded_then_final_authority_and_forward(self):
        seen = []
        def fake_init(obj, root, *, authority):
            seen.append(copy.deepcopy(authority))
            obj.control_index=0;obj.step_count=0;obj.calls=[{'role':'warmup'}]
            obj.joint_names=[f'j{i}' for i in range(29)]
            obj.joint_ranges=np.tile([-1,1],(29,1))
        with patch.object(HybridArmBackend,'__init__',fake_init):
            backend=PerformanceBackend('.',authority=self.authority())
        self.assertEqual(seen[0]['max_control_steps'],500)
        self.assertEqual(seen[0]['schema'],'pulsemorph.embodied_execution_authority.v1')
        self.assertEqual(backend.authority['max_control_steps'],1200)
        q=np.zeros((4,29));dq=q.copy();q[:,0]=.2;q[:,14]=.3
        with patch.object(HybridArmBackend,'advance') as advance:
            backend.advance(q,dq)
            np.testing.assert_array_equal(advance.call_args.args[0],q)
            backend.authority['max_control_steps']=1201
            with self.assertRaises(RuntimeError): backend.advance(q,dq)
            self.assertEqual(advance.call_count,1)

    def test_unexpected_bootstrap_work_stops(self):
        def fake_init(obj, root, *, authority):
            obj.control_index=1;obj.step_count=20;obj.calls=[{},{}]
        with patch.object(HybridArmBackend,'__init__',fake_init):
            with self.assertRaises(RuntimeError):
                PerformanceBackend('.',authority=self.authority())


if __name__ == '__main__': unittest.main()
