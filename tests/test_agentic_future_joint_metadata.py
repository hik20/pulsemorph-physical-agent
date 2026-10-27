"""No-inference fixtures for the one symbolic-batch setup revision."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner_v2',ROOT/'experiments/agentic/app/future_joint_runner_v2.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.declared=runner.declared_metadata()
        self.protocol=json.loads((runner.PLAN/'protocol.json').read_text())
    def check(self,kind,declared=None,concrete=None):
        actual=self.declared[kind] if declared is None else declared
        nodes=[SimpleNamespace(**n) for n in actual]
        runner.validate_metadata(nodes,self.protocol[kind] if concrete is None else concrete,self.declared[kind])
    def test_exact_symbolic_batch_and_concrete_one(self):
        for kind in ('inputs','outputs'):self.check(kind)
    def test_reject_metadata_drift(self):
        for kind in ('inputs','outputs'):
            for key,value in (('name','other'),('type','tensor(double)'),('shape',[1,29]),('shape',['other_batch',29]),('shape',['batch_size',30])):
                with self.subTest(kind=kind,key=key,value=value):
                    bad=copy.deepcopy(self.declared[kind]);bad[0][key]=value
                    with self.assertRaises(ValueError):self.check(kind,bad)
    def test_reject_concrete_batch_or_dimension_drift(self):
        for kind in ('inputs','outputs'):
            for axis in (0,-1):
                bad=copy.deepcopy(self.protocol[kind]);bad[0]['shape'][axis]+=1
                with self.assertRaises(ValueError):self.check(kind,concrete=bad)


if __name__=='__main__':unittest.main()
