"""V2 gate namespace/cache tests; real one-artifact scoring parity, no physics."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/agentic'))
import audit_repair_v2 as v2
import evaluate_repair
import evaluate_generalization
import test_agentic_repair_audit as original_tests


class RepairAuditV2Tests(unittest.TestCase):
    def test_actual_four_relative_gate_seeds_and_wrong_hash_path(self):
        panel=ROOT/'artifacts/runs/agentic/repair_development_v1'
        for proposalpath in sorted((panel/'trial_proposals').glob('gate__*.json')):
            proposal=json.loads(proposalpath.read_text())
            role=proposalpath.stem.removeprefix('gate__')
            expected=ROOT/'artifacts/runs/agentic/robustness_initial_state_v2/trials'/('nominal__'+role)/'proposal.json'
            origin=proposal['parameter_source']
            self.assertFalse(Path(origin['path']).is_absolute())
            self.assertEqual(v2.verify_gate_seed(origin,expected),expected.resolve())
            with self.assertRaisesRegex(ValueError,'SHA256'):
                v2.verify_gate_seed({**origin,'sha256':'0'*64},expected)
            other=next(p for p in (panel/'trial_proposals').glob('gate__*.json') if p!=proposalpath)
            wrong=dict(path=str(other),sha256=hashlib.sha256(other.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError,'differs'):
                v2.verify_gate_seed(wrong,expected)
        self.assertEqual(len(list((panel/'trial_proposals').glob('gate__*.json'))),4)

    def test_cache_deepcopy_manifest_invalidation_and_source_fail_closed(self):
        calls=[]
        def raw(path):
            calls.append(str(path));return dict(array=np.array([len(calls)],dtype=float),nested={'value':[1]})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);artifact=root/'trial';artifact.mkdir()
            manifest=artifact/'manifest.json';manifest.write_text('{}')
            source=root/'raw_source.py';source.write_text('original')
            cache=v2.RawAuditCache(raw,[source])
            first=cache(artifact);first['array'][0]=99;first['nested']['value'].append(9)
            second=cache(artifact)
            self.assertEqual(second['array'].tolist(),[1]);self.assertEqual(second['nested']['value'],[1])
            self.assertEqual((cache.hits,cache.misses),(1,1))
            manifest.write_text('{"changed":true}');cache(artifact)
            self.assertEqual((cache.hits,cache.misses),(1,2))
            source.write_text('mutated')
            with self.assertRaisesRegex(ValueError,'source changed'):cache(artifact)

    def test_real_one_artifact_normal_cached_score_equal(self):
        panel=ROOT/'artifacts/runs/agentic/repair_development_v1'
        protocol=json.loads((panel/'protocol.json').read_text());task=next(t for t in protocol['tasks'] if t['task_id']=='wave_a')
        trial=panel/'trials/candidate__wave_a__surrogate_search__0__nominal'
        baseline=Path(protocol['baseline_panel'])/'trials/nominal__calibration_wave'
        expected=evaluate_repair.evaluate_repair(trial,baseline,baseline,protocol,task['goal'])
        cache=v2.RawAuditCache(evaluate_generalization._audit,[ROOT/'experiments/agentic/evaluate_generalization.py',ROOT/'experiments/agentic/evaluate_trial.py'])
        original=evaluate_generalization._audit
        with v2.install_raw_cache(cache):
            actual=evaluate_repair.evaluate_repair(trial,baseline,baseline,protocol,task['goal'])
            again=evaluate_repair.evaluate_repair(trial,baseline,baseline,protocol,task['goal'])
        self.assertIs(evaluate_generalization._audit,original)
        self.assertEqual(expected,actual);self.assertEqual(actual,again)
        self.assertEqual(cache.misses,2);self.assertEqual(cache.hits,4)
        self.assertTrue(expected['integrity_passed'])

    def test_synthetic_full250_v2_with_cache_and_bound_reports(self):
        helper=original_tests.RepairAuditOrchestrationTests()
        with tempfile.TemporaryDirectory() as temporary:
            panel,priors,observations,raw,results=helper.make_panel(temporary)
            def raw_function(path):return raw.get(str(path),dict(manifest={},arrays={},names=[]))
            for folder,payload in [('repair_development_v1_audit_v1',dict(audit_passed=False,panel_evidence_unchanged=True)),
                ('repair_legacy_fields_review_v1',dict(bounded_impact_checks_passed=True,raw_metadata_semantically_correct=False))]:
                target=Path(temporary)/folder/'audit.json' if folder.endswith('audit_v1') else Path(temporary)/folder/'report.json'
                target.parent.mkdir(parents=True);target.write_text(json.dumps(payload))
            with ExitStack() as stack:
                stack.enter_context(patch('evaluate_generalization._audit',raw_function))
                stack.enter_context(patch.object(evaluate_repair.native._original._base,'_audit',raw_function))
                stack.enter_context(patch('audit_robustness_v2.check_initial_state',return_value={}))
                stack.enter_context(patch('audit_robustness_v2.check_center_definitions',return_value={}))
                stack.enter_context(patch('audit_multifeature.common_parity',return_value={}))
                stack.enter_context(patch('run_tool_compatibility_smoke.trace_parity',return_value={'same':True}))
                stack.enter_context(patch('evaluate_repair.evaluate_repair',side_effect=lambda directory,*args:results[str(directory)]))
                stack.enter_context(patch('repair_plan.prior_data',side_effect=lambda task,*args:priors[task['task_id']]))
                stack.enter_context(patch('repair_driver.observation_data',side_effect=lambda out,task,method,index:observations[f'{task}__{method}__{index}']))
                stack.enter_context(patch('repair_search.next_candidate',side_effect=lambda measured,*args:dict(control_tool='global_clock_v1',parameters=[1.,1.,1.,1.,0.],search_diagnostics={'index':len(measured)})))
                report=v2.audit(panel,Path(temporary)/'v2_audit')
            self.assertTrue(report['audit_passed'],report['errors'])
            self.assertEqual(report['raw_audit_cache']['misses'],276)
            self.assertEqual(report['raw_audit_cache']['unique_artifact_paths'],276)
            self.assertTrue(report['audit_sources_unchanged']);self.assertTrue(report['bound_prior_reports_unchanged'])
            self.assertFalse(report['prior_partial_pass_flags_reused'])
            self.assertFalse(report['raw_metadata_semantically_correct'])

if __name__=='__main__':unittest.main()
