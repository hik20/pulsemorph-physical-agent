"""Driver control-flow tests with no physics, model or real artifact writes."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/agentic'))
spec = importlib.util.spec_from_file_location('robustness_driver_tested', ROOT/'experiments/agentic/robustness_driver.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
import robustness_plan


class RobustnessDriverTests(unittest.TestCase):
    def fixture(self, parent):
        panel = Path(parent)/'panel'
        roles = [('calibration_wave','wave','wave_a'),('calibration_high5','high5','high5_a')]
        roles += [('selected_'+task,task.split('_')[0],task) for task in ('wave_a','wave_b','high5_a','high5_b')]
        cases = []
        for role,source,task in roles:
            original = Path(parent)/'original'/role
            original.mkdir(parents=True)
            (original/'manifest.json').write_text('{}')
            cases.append(dict(role=role,source_id=source,task_id=task,original_trial=str(original),original_proposal={'path':role,'sha256':'fixed'}))
        protocol = dict(cases=cases, scenarios=[dict(scenario_id='nominal',initial_perturbation={})]+[
            dict(scenario_id='p'+str(i),initial_perturbation={'i':i}) for i in range(12)],
            tasks=[dict(task_id=task,goal={}) for task in ('wave_a','wave_b','high5_a','high5_b')],
            execution_code_sha256={}, physical_gate=dict(min_root_height_m=.6,max_mean_joint_rmse_rad=.35))
        def freeze(path):
            path.mkdir(); (path/'proposals').mkdir()
            (path/'protocol.json').write_text(json.dumps(protocol))
            digest = driver.sha256(path/'protocol.json')
            for scenario in protocol['scenarios']:
                for case in cases:
                    identity=scenario['scenario_id']+'__'+case['role']
                    (path/'proposals'/(identity+'.json')).write_text(json.dumps(dict(candidate_id=identity,
                        initial_perturbation=scenario['initial_perturbation'],parameter_source=case['original_proposal'],protocol_sha256=digest)))
            return protocol
        executed=[]
        def execute(proposal,path):
            executed.append(path.name); path.mkdir(parents=True)
            manifest=dict(status='success',strict_g1_admission_passed=True,strict_g1_preflight_count=2,actual_control_steps=200,actual_physics_steps=4000)
            (path/'manifest.json').write_text(json.dumps(manifest))
            return manifest
        def evaluate(path,*args):
            passed = path.name.startswith('nominal__')
            score=dict(joint_success=passed,intent_passed=passed,features=[1]*5)
            return dict(integrity_passed=True,integrity_errors=[],primary=score.copy(),secondary=score.copy(),
                        physical_gate_passed=passed,min_root_height_m=.8 if passed else .5,mean_joint_rmse_rad=.1)
        return panel,freeze,execute,evaluate,executed

    def run_mocked(self, parent, parity=None, nominal_height=.8):
        panel,freeze,execute,evaluate,executed=self.fixture(parent)
        with patch.object(robustness_plan,'freeze',side_effect=freeze), patch.object(driver,'execute_trial',side_effect=execute), \
             patch.object(driver,'check_input_pins',return_value={}), patch.object(driver,'array_archive_parity',return_value=True), \
             patch.object(driver,'_audit',return_value={}), patch.object(driver,'calibration_measurements',return_value=dict(min_root_height_m=nominal_height,mean_joint_rmse_rad=.1)), \
             patch.object(driver,'evaluate_robustness',side_effect=evaluate), patch.object(driver,'trace_parity',side_effect=parity or (lambda *args:{})), patch('builtins.print'):
            result=driver.run(panel)
        return result,executed,panel

    def test_nominal_parity_failure_aborts_before_perturbations(self):
        count=0
        def parity(*args):
            nonlocal count
            count+=1
            if count==6: raise ValueError('injected nominal trace mismatch')
            return {}
        with tempfile.TemporaryDirectory() as temp:
            result,executed,panel=self.run_mocked(temp,parity)
            self.assertFalse(result['panel_completed'])
            self.assertEqual(len(executed),6)
            self.assertTrue(all(name.startswith('nominal__') for name in executed))
            self.assertFalse((panel/'nominal_parity.json').exists())
            self.assertTrue((panel/'failure_traceback.txt').exists())

    def test_measured_nonzero_physical_failure_continues_and_counts_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            result,executed,panel=self.run_mocked(temp)
            self.assertTrue(result['panel_completed'],result['errors'])
            self.assertEqual(len(executed),78)
            self.assertEqual(len(result['rows']),52)
            self.assertEqual(result['physical_trials'],78)
            self.assertEqual(result['control_steps'],15600)
            self.assertEqual(result['physics_steps'],312000)
            for task in result['tasks']:
                self.assertTrue(task['nominal_primary_passed'])
                self.assertEqual(task['nonzero_cases'],12)
                self.assertEqual(task['primary_passed'],0)
                self.assertEqual(task['physical_passed'],0)
                self.assertEqual(len(task['primary_failures']),12)
            self.assertTrue((panel/'nominal_parity.json').exists())

    def test_nominal_calibration_physical_failure_must_abort(self):
        with tempfile.TemporaryDirectory() as temp:
            result,executed,panel=self.run_mocked(temp,nominal_height=.5)
            self.assertFalse(result['panel_completed'])
            self.assertTrue(all(name.startswith('nominal__') for name in executed))
            self.assertFalse((panel/'nominal_parity.json').exists())


if __name__=='__main__': unittest.main()
