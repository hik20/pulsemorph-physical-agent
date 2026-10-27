"""Run preserved driver invariants against the versioned v2 control flow."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/agentic'))
import robustness_driver_v2
import robustness_plan_v2

spec=importlib.util.spec_from_file_location('_preserved_driver_cases',ROOT/'tests/test_agentic_robustness_driver.py')
preserved=importlib.util.module_from_spec(spec)
spec.loader.exec_module(preserved)


class VersionedDriverInvariantTests(unittest.TestCase):
    def run_preserved_case(self, name):
        # Substitute module bindings in memory only; frozen source is untouched.
        with patch.object(preserved,'driver',robustness_driver_v2), patch.object(preserved,'robustness_plan',robustness_plan_v2):
            case=preserved.RobustnessDriverTests(methodName=name)
            getattr(case,name)()

    def test_zero_parity_failure_stops_before_nonzero(self):
        self.run_preserved_case('test_nominal_parity_failure_aborts_before_perturbations')

    def test_nominal_calibration_physical_failure_stops(self):
        self.run_preserved_case('test_nominal_calibration_physical_failure_must_abort')

    def test_nonzero_failures_preserved_and_exact_78_accounting(self):
        self.run_preserved_case('test_measured_nonzero_physical_failure_continues_and_counts_separately')


if __name__=='__main__': unittest.main()
