"""Advisory orchestration boundaries. Provider/runtime/tool are fixtures only."""
import copy
import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_agentic_app_phrase_manager import MusicFixture, ProviderFixture, RuntimeFixture, MemoryFixture, renderer, PROGRAM
from pulsemorph.agentic.app.phrase.manager_advice_v5 import PhraseAdviceManager, _fields_for_report
from pulsemorph.agentic.app.phrase.gain_advice_v1 import AdviceTimeout, AdviceCancelled
from pulsemorph.agentic.phrase_program import ProgramCompileError

GOALS = {'schema': 'pulsemorph.phrase_goals.v1', 'ratios': [.75, 1.25, 1.25, .75]}


class RejectingRuntime(RuntimeFixture):
    def __init__(self, baseline=False):
        super().__init__()
        self.assets = object()
        self.baseline = baseline
        self.original_report = {'status': 'rejected_joint_ranges', 'violations': ['left_elbow_joint']}

    def run_trial(self, **kwargs):
        index = kwargs['trial_index']
        if index == (-1 if self.baseline else 0):
            self.calls.append(index)
            raise ProgramCompileError('fixture local rejection', self.original_report)
        return super().run_trial(**kwargs)


class AdviceManagerTests(unittest.TestCase):
    def make(self, runtime=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        initial = {'supported': True, 'intent_summary': 'fixture', 'unsupported_requests': [],
                   'program': copy.deepcopy(PROGRAM), 'goals': copy.deepcopy(GOALS)}
        provider = ProviderFixture(initial=initial)
        runtime = runtime or RejectingRuntime()
        root = Path(temp.name)
        manager = PhraseAdviceManager(root, provider, runtime, MusicFixture(), MemoryFixture(), renderer, root / 'app')
        self.addCleanup(manager.shutdown)
        return manager, provider, runtime

    def create(self, manager):
        return manager.create({'instruction': 'fixture fixed goals', 'music_id': 'fixture_music',
                               'max_iterations': 2, 'goals': copy.deepcopy(GOALS)})

    def done(self, manager, identity):
        for _ in range(300):
            job = manager.get(identity)
            if job['status'] in ('complete', 'failed', 'cancelled'):
                return job
            time.sleep(.01)
        self.fail('fixture job did not finish')

    @staticmethod
    def evidence(assets, program, field, **kwargs):
        return {'schema': 'pulsemorph.conditional_gain_advice.v1', 'field': field,
                'status': 'available', 'lower': .75, 'upper': 1.22,
                'conditioning': {'other_gains': {k: v for k, v in program.items() if k.endswith('_gain') and k != field}},
                'execution_authority': False}

    def test_real_rejection_evidence_is_copied_saved_and_given_to_feedback(self):
        manager, provider, runtime = self.make()
        with patch('pulsemorph.agentic.app.phrase.manager_advice_v5.advise_gain_interval', side_effect=self.evidence) as tool:
            identity = self.create(manager)
            job = self.done(manager, identity)
        self.assertEqual(job['status'], 'complete')
        self.assertEqual(runtime.calls, [-1, 0, 1])
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual([c.args[2] for c in tool.call_args_list], ['second_left_gain', 'first_left_gain'])
        self.assertEqual(job['targets'], GOALS['ratios'])
        self.assertEqual(job['trials'][0]['status'], 'rejected')
        self.assertIsNone(job['trials'][0]['metrics'])
        self.assertNotIn('conditional_gain_advice', runtime.original_report)
        report = job['trials'][0]['rejection']['report']
        batch = report['conditional_gain_advice']
        self.assertLessEqual(len(batch['intervals']), 2)
        path = manager.jobs_dir / identity / batch['saved_receipt']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), batch['saved_receipt_sha256'])
        receipt = json.loads(path.read_bytes())
        self.assertEqual(receipt['original_compile_report_sha256'], hashlib.sha256(json.dumps(runtime.original_report, ensure_ascii=False, allow_nan=False, indent=2).encode()).hexdigest())
        prepared = json.loads((manager.jobs_dir / identity / 'prepared_plan_v4.json').read_bytes())
        self.assertEqual(provider.calls[0][0], prepared['prompt'])
        self.assertIn('conditional_gain_advice', provider.calls[1][0])
        self.assertIn('not a simultaneous feasible region', provider.calls[1][0])
        self.assertFalse(job['live'])
        before = path.read_bytes()
        with patch('pulsemorph.agentic.app.phrase.manager_advice_v5.advise_gain_interval', side_effect=self.evidence):
            with self.assertRaises(FileExistsError):
                manager._run_trial(identity, PROGRAM, 0, MusicFixture().get('fixture_music'))
        self.assertEqual(path.read_bytes(), before)

    def test_baseline_failure_never_asks_for_advice(self):
        manager, provider, runtime = self.make(RejectingRuntime(baseline=True))
        with patch('pulsemorph.agentic.app.phrase.manager_advice_v5.advise_gain_interval') as tool:
            identity = self.create(manager)
            job = self.done(manager, identity)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(runtime.calls, [-1])
        tool.assert_not_called()
        self.assertFalse((manager.jobs_dir / identity / 'advice').exists())

    def test_tool_timeout_is_unknown_and_does_not_change_goals(self):
        manager, provider, runtime = self.make()
        with patch('pulsemorph.agentic.app.phrase.manager_advice_v5.advise_gain_interval', side_effect=AdviceTimeout()):
            job = self.done(manager, self.create(manager))
        self.assertEqual(job['status'], 'complete')
        batch = job['trials'][0]['rejection']['report']['conditional_gain_advice']
        self.assertTrue(all(i['status'] == 'unavailable' for i in batch['intervals']))
        self.assertEqual(job['targets'], GOALS['ratios'])
        self.assertEqual(len(provider.calls), 2)

    def test_source_integrity_error_propagates_before_feedback(self):
        manager, provider, runtime = self.make()
        with patch('pulsemorph.agentic.app.phrase.manager_advice_v5.advise_gain_interval', side_effect=ValueError('fixture integrity')):
            identity = self.create(manager)
            job = self.done(manager, identity)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue((manager.jobs_dir / identity / 'advice/trial_00.attempt.json').exists())
        self.assertFalse((manager.jobs_dir / identity / 'advice/trial_00.json').exists())

    def test_cancel_during_tool_keeps_original_rejection_without_execution_claim(self):
        manager, provider, runtime = self.make()
        def cancel(*args, **kwargs):
            kwargs['cancel_event'].set()
            raise AdviceCancelled()
        with patch('pulsemorph.agentic.app.phrase.manager_advice_v5.advise_gain_interval', side_effect=cancel):
            identity = self.create(manager)
            job = self.done(manager, identity)
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(runtime.calls, [-1, 0])
        self.assertIsNone(job.get('active_candidate'))
        self.assertEqual(job['trials'], [])
        attempt = json.loads((manager.jobs_dir / identity / 'advice/trial_00.attempt.json').read_bytes())
        self.assertEqual(attempt['original_compile_report'], runtime.original_report)
        self.assertTrue(attempt['completion_unconfirmed'])

    def test_runtime_guard_runs_after_timeout_before_feedback(self):
        runtime = RejectingRuntime()
        runtime._unchanged = unittest.mock.Mock(side_effect=RuntimeError('fixture runtime integrity'))
        manager, provider, _ = self.make(runtime)
        with patch('pulsemorph.agentic.app.phrase.manager_advice_v5.advise_gain_interval', side_effect=AdviceTimeout()):
            job = self.done(manager, self.create(manager))
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(len(provider.calls), 1)
        runtime._unchanged.assert_called_once()

    def test_no_arm_or_malformed_constraint_never_infers_an_interval(self):
        self.assertEqual(_fields_for_report({'violations': ['left_hip_roll_joint']}), [])
        self.assertEqual(_fields_for_report({'violations': 'left_elbow_joint'}), [])
        self.assertEqual(_fields_for_report({'violations': ['right_wrist_roll_joint', 'left_elbow_joint']}), ['second_left_gain', 'first_left_gain'])


if __name__ == '__main__':
    unittest.main()
