import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repair_runtime', ROOT/'experiments/agentic/execute_repair.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RepairRuntimeTests(unittest.TestCase):
    def test_seed_task_and_source_fixed_but_method_tool_parameters_can_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root/'artifacts/runs/agentic/seed.json'
            seed.parent.mkdir(parents=True)
            original = dict(task_id='goal', source_motion={'path':'motion', 'sha256':'a'*64},
                            method='llm_feedback', control_tool='global_clock_v1', parameters={})
            seed.write_text(json.dumps(original))
            proposal = dict(original, execution_kind='robust_initial_state_repair',
                            method='surrogate_search', control_tool='local_clock_1p4_2p8_v1',
                            parameters={name: 0. if name=='timing_shift_s' else 1.1 for name in
                                        ('left_gain_a','right_gain_a','left_gain_b','right_gain_b','timing_shift_s')},
                            parameter_source={'path':'artifacts/runs/agentic/seed.json','sha256':runtime.sha256(seed)})
            with patch.object(runtime, 'ROOT', root):
                self.assertEqual(runtime.validate_repair_proposal(proposal)[1], seed.read_bytes())
                for key, value in [('task_id','other'), ('source_motion',{}),
                                   ('execution_kind','initial_state_sensitivity_replay'), ('method','calibration')]:
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        runtime.validate_repair_proposal({**proposal,key:value})
                seed.write_text('{}')
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    runtime.validate_repair_proposal(proposal)

    def test_numeric_runtime_functions_are_frozen_identical(self):
        import ast
        old = ast.parse((ROOT/'experiments/agentic/execute_robustness.py').read_text())
        new = ast.parse((ROOT/'experiments/agentic/execute_repair.py').read_text())
        for name in ('validate_parameters','validate_initial_perturbation','apply_initial_perturbation',
                     'strict_g1_preflight','build_generator'):
            left = next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name==name)
            right = next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name==name)
            self.assertEqual(ast.dump(left),ast.dump(right))
        def numeric_tail(tree):
            execute=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='execute_trial')
            body=next(node for node in execute.body if isinstance(node,ast.Try)).body
            start=next(i for i,node in enumerate(body) if isinstance(node,ast.Expr)
                       and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                       and node.value.func.id=='validate_parameters')
            return [ast.dump(node) for node in body[start:]]
        # The complete generator/controller/observation/reset/physics body is
        # bytecode-structure identical after the permitted provenance branch.
        self.assertEqual(numeric_tail(old),numeric_tail(new))


if __name__ == '__main__': unittest.main()
