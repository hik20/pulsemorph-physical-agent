import sys
from pathlib import Path
import unittest
import json
import tempfile
import copy
import shutil
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/agentic'))
import evaluate_repair as repair
from evaluate_repair import bridge_pins, aggregate_conditions, selection_key


class RepairEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.old = 'experiments/agentic/execute_robustness.py'
        self.new = 'experiments/agentic/execute_repair.py'
        self.switch = {'old': {'path': self.old, 'sha256': 'a'*64},
                       'new': {'path': self.new, 'sha256': 'b'*64}}

    def test_only_declared_wrapper_can_differ(self):
        baseline = {self.old: 'a'*64, 'model': 'c'*64}
        trial = {self.new: 'b'*64, 'model': 'c'*64}
        self.assertEqual(bridge_pins(trial, baseline, self.switch), baseline)
        for changed in ({**trial, 'model': 'd'*64}, {**trial, 'extra': 'd'*64},
                        {**trial, self.new: 'e'*64}, {**trial, self.old: 'a'*64}):
            with self.assertRaises(ValueError):
                bridge_pins(changed, baseline, self.switch)

    def test_complete_aggregate_and_missing_accent(self):
        def row(utility):
            return dict(integrity_passed=True, physical_gate_passed=True,
                        primary=dict(utility=utility, utility_l2=utility**2 if utility is not None else None, joint_success=True))
        values = {'a': row(.2), 'b': row(1.3)}
        self.assertEqual(aggregate_conditions(values, ['a','b'])['utility'], 1.3)
        aggregate = aggregate_conditions(values, ['a','b'])
        self.assertAlmostEqual(aggregate['utility_l2'], .2**2+1.3**2)
        self.assertEqual(selection_key(aggregate, 'seed'), (False, 1.3, .2**2+1.3**2, 'seed'))
        with self.assertRaises(ValueError): aggregate_conditions(values, ['a'])
        with self.assertRaises(ValueError): aggregate_conditions(values, ['a','a'])
        values['b']['primary']['utility'] = None
        self.assertIsNone(aggregate_conditions(values, ['a','b'])['utility'])
        self.assertEqual(selection_key(aggregate_conditions(values, ['a','b']), 'missing')[1], float('inf'))
        values['b']['integrity_passed'] = False
        self.assertFalse(aggregate_conditions(values, ['a','b'])['joint_success'])
        with self.assertRaises(ValueError):
            selection_key(aggregate_conditions(values, ['a','b']), 'invalid')

    def test_actual_windows_manifest_wrapper_replacement(self):
        root = Path(__file__).resolve().parents[1]
        manifests = sorted((root/'artifacts/runs/agentic/robustness_initial_state_v2/trials').glob('*/manifest.json'))
        baseline = json.loads(manifests[0].read_text())['preexecution_input_hashes']
        trial = dict(baseline)
        old_key = next(key for key in trial if key.replace('\\','/') == self.old)
        old_sha = trial.pop(old_key)
        trial[self.new.replace('/','\\')] = 'b'*64
        switch = {'old': {'path':self.old,'sha256':old_sha}, 'new':self.switch['new']}
        self.assertEqual(bridge_pins(trial,baseline,switch), {key.replace('\\','/'):value for key,value in baseline.items()})
        trial[self.new] = 'b'*64
        with self.assertRaisesRegex(ValueError,'collision'):
            bridge_pins(trial,baseline,switch)

    def test_real_recorded_arrays_keep_legacy_score_under_synthetic_wrapper_bridge(self):
        root=Path(__file__).resolve().parents[1]
        panel=root/'artifacts/runs/agentic/robustness_initial_state_v2'
        old_trial=panel/'trials/nominal__selected_wave_a'
        baseline=panel/'trials/nominal__calibration_wave'
        protocol=json.loads((panel/'protocol.json').read_text())
        goal=next(task['goal'] for task in protocol['tasks'] if task['task_id']=='wave_a')
        base=repair.native._original._base
        # Complete real nested inventories and fresh FK are checked first.
        trial_audit=base._audit(old_trial)
        baseline_audit=base._audit(baseline)
        adapted=copy.deepcopy(trial_audit)
        pins=adapted['manifest']['preexecution_input_hashes']
        old_key=next(key for key in pins if key.replace('\\','/')==self.old)
        old_sha=pins.pop(old_key);pins[self.new]='b'*64
        protocol['repair_wrapper_pin_switch']={'old':dict(path=self.old,sha256=old_sha),'new':self.switch['new']}
        with tempfile.TemporaryDirectory() as temporary:
            trial=Path(temporary)
            seed_bytes=(old_trial/'proposal.json').read_bytes()
            origin=dict(path='artifacts/runs/agentic/seed.json',sha256=repair.hashlib.sha256(seed_bytes).hexdigest())
            proposal=json.loads(seed_bytes)
            proposal.update(execution_kind='robust_initial_state_repair',parameter_source=origin)
            (trial/'proposal.json').write_text(json.dumps(proposal))
            (trial/'source_proposal.json').write_bytes(seed_bytes)
            shutil.copyfile(old_trial/'unwarped_source_root_positions.npy',trial/'unwarped_source_root_positions.npy')
            adapted['manifest'].update(execution_kind='robust_initial_state_repair',parameter_source=origin)
            lookup={str(trial):adapted,str(baseline):baseline_audit}
            with patch.object(base,'_audit',side_effect=lambda path:lookup[str(path)]):
                result=repair.evaluate_repair(trial,baseline,baseline,protocol,goal)
            legacy=json.loads((panel/'evaluations/nominal__selected_wave_a.json').read_text())
            self.assertTrue(result['integrity_passed'])
            self.assertEqual(result['primary'],legacy['primary'])
            self.assertEqual(result['secondary'],legacy['secondary'])


if __name__ == '__main__': unittest.main()
