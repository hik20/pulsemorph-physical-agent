"""No-inference fixtures for the prospective six-call counterfactual primitive."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'experiments/agentic/app/future_joint_probe_v1.py'
spec = importlib.util.spec_from_file_location('future_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def pairs():
    result = []
    for d in probe.DECISIONS:
        a = {n: np.zeros(s, dtype=np.float32) for n,s in zip(probe.INPUTS,probe.SHAPES)}
        b = {n: v.copy() for n,v in a.items()}
        for n in probe.SWAPPED:
            b[n] += .1
        result.append((d,a,b,{n:np.zeros((1,29),np.float32) for n in probe.OUTPUTS}))
    return result


class Session:
    def __init__(self, path, mismatch=False, fail=False):
        self.path, self.calls, self.mismatch, self.fail = path, 0, mismatch, fail
    def run(self, names, feed):
        assert (self.path / f'attempt_{self.calls:02d}.json').is_file()
        assert names == list(probe.OUTPUTS)
        self.calls += 1
        if self.fail:
            raise RuntimeError('fixture inference failure')
        return [np.full((1,29), float(self.mismatch),np.float32) for _ in names]


class ProbeTests(unittest.TestCase):
    def test_exact_six_order_and_durable_attempts(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'calls'; s=Session(p)
            rows=probe.finite_calls(s,pairs(),p)
            self.assertEqual(s.calls,6)
            self.assertEqual([(d,c) for d,c,_ in rows],[(d,c) for d in probe.DECISIONS for c in ('baseline','edited')])
            self.assertEqual(len(list(p.glob('attempt*'))),6)
            with self.assertRaises(FileExistsError): probe.finite_calls(s,pairs(),p)
            self.assertEqual(s.calls,6)
    def test_replay_failure_stops_before_edited(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'calls'; s=Session(p,mismatch=True)
            with self.assertRaisesRegex(ValueError,'replay mismatch'): probe.finite_calls(s,pairs(),p)
            self.assertEqual(s.calls,1)
            self.assertTrue((p/'attempt_00.json').exists())
    def test_runtime_failure_durable_no_retry(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'calls'; s=Session(p,fail=True)
            with self.assertRaises(RuntimeError): probe.finite_calls(s,pairs(),p)
            self.assertEqual(s.calls,1)
            with self.assertRaises(FileExistsError): probe.finite_calls(s,pairs(),p)
            self.assertEqual(s.calls,1)
    def test_wrong_order_or_budget_no_calls(self):
        for rows in (pairs()[::-1],pairs()[:2],pairs()+pairs()):
            with tempfile.TemporaryDirectory() as t:
                p=Path(t)/'calls'; s=Session(p)
                with self.assertRaises(ValueError): probe.finite_calls(s,rows,p)
                self.assertEqual(s.calls,0)
    def test_current_history_and_torso_are_locked(self):
        for name in set(probe.INPUTS)-probe.SWAPPED:
            rows=pairs();rows[0][2][name].flat[0]=1
            with self.assertRaisesRegex(ValueError,'non-joint'): probe.validate_pair(rows[0][1],rows[0][2])
    def test_bad_shape_dtype_nonfinite_rejected(self):
        for replacement in (np.zeros((1,29),np.float32),np.zeros((1,4,29),np.float64),np.full((1,4,29),np.nan,np.float32)):
            rows=pairs();rows[0][2]['mimic_future_dof_pos']=replacement
            with self.assertRaises(ValueError): probe.validate_pair(rows[0][1],rows[0][2])


if __name__ == '__main__': unittest.main()
