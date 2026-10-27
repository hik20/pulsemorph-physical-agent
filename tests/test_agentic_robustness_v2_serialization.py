"""Serialize genuine metric arithmetic for every synthetic outcome branch."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/agentic'))
import evaluate_robustness_v2 as repaired

spec=importlib.util.spec_from_file_location('_frozen_robustness_fixture',ROOT/'tests/test_agentic_robustness_evaluation.py')
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class JsonMetricBranchTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.RobustnessEvaluationTests()
        self.fixture.setUp()

    def evaluate(self):
        f=self.fixture
        with patch.object(repaired._original._base,'_audit',side_effect=[f.candidate,f.nominal,f.condition]), \
             patch.object(repaired._original._base,'quality_diagnostics',return_value={'diagnostic_only':True}):
            return repaired.evaluate_robustness('c','n','p',f.protocol,f.goal)

    def roundtrip(self,result):
        encoded=json.dumps(result,allow_nan=False)
        self.assertEqual(json.loads(encoded),result)
        return encoded

    def test_real_score_arithmetic_produces_native_pass_bits(self):
        score=repaired.score_features([1.,1.,1.,1.,2.5],np.array([1.,1.,1.,1.,2.5]),np.array([.08]*4+[.1]))
        self.assertEqual(score['per_dimension_pass'],[True]*5)
        self.assertTrue(all(type(value) is bool for value in score['per_dimension_pass']))
        self.roundtrip(score)

    def test_complete_primary_failure_secondary_success_payload_roundtrip(self):
        result=self.evaluate()
        self.assertTrue(result['integrity_passed'])
        self.assertFalse(result['primary']['intent_passed'])
        self.assertTrue(result['secondary']['intent_passed'])
        self.roundtrip(result)

    def test_nominal_success_payload_roundtrip_and_preserved_arithmetic(self):
        f=self.fixture
        f.candidate=copy.deepcopy(f.nominal); f.condition=copy.deepcopy(f.nominal)
        f.candidate['arrays']['qpos'][1:,7:]*=1.04
        result=self.evaluate()
        self.roundtrip(result)
        self.assertEqual(result['primary'],result['secondary'])
        self.assertAlmostEqual(result['primary']['features'][0],1.04)
        self.assertTrue(result['primary']['joint_success'])

    def test_missing_accent_and_missing_secondary_denominator_roundtrip(self):
        f=self.fixture
        f.candidate['arrays']['qvel'][:]=0
        f.condition['arrays']['qpos'][:,7:]=.2
        result=self.evaluate()
        self.roundtrip(result)
        self.assertIsNone(result['primary']['features'][4])
        self.assertTrue(all(value is None for value in result['secondary']['features']))
        self.assertFalse(result['secondary']['intent_passed'])

    def test_physical_failure_keeps_serializable_measurements(self):
        f=self.fixture
        f.candidate['sub'][0,2]=.5
        result=self.evaluate()
        self.roundtrip(result)
        self.assertFalse(result['physical_gate_passed'])
        self.assertIsNotNone(result['primary']['features'])

    def test_integrity_failure_payload_roundtrip(self):
        self.fixture.condition['manifest']['motion_sha256']='changed'
        result=self.evaluate()
        self.roundtrip(result)
        self.assertFalse(result['integrity_passed'])
        self.assertIsNone(result['primary'])


if __name__=='__main__': unittest.main()
