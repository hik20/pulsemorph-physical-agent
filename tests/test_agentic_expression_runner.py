"""Control-flow fixtures only: fake planner/runtime/renderer, no live calls."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_agentic_expression_goals import trial as fixture_trial
from pulsemorph.agentic.expression import runner_v1 as module
from pulsemorph.agentic.expression.goals_v1 import parse_goals
from pulsemorph.agentic.phrase_program import ProgramCompileError

REQUEST = {'instruction': 'High5 sol kolun iki özelliğini artır; diğer referansları koru.',
           'music_id': 'phrase_original_pulse_v1',
           'goals': {'schema': 'pulsemorph.expression_goals.v1', 'mean_pose_ratio': 1.2, 'temporal_ratio': 1.2}}


class Memory:
    def status(self): return {'meterless': {'state': 'fixture'}, 'zetaris': {'executed': False}}
    def record_job(self, value): pass
    def record_trial(self, identity, value): pass


class Music:
    def get(self, identity): return {'id': identity, 'path': 'fixture.wav', 'sha256': 'c' * 64}


class Runtime:
    _implementation = {}
    environment = {'fixture': True}
    def __init__(self, reject=False, fail=False): self.calls, self.reject, self.fail = [], reject, fail
    def status(self): return {'available': True}
    def _unchanged(self): pass
    def _verify_trial(self, trial): pass
    def run_trial(self, expression, output, job_id, index, music, cancel):
        self.calls.append(index)
        Path(output).mkdir(parents=True)
        if self.fail and index == 0:
            (output / 'worker_started.json').write_text('{}')
            raise RuntimeError('Synthetic partial-worker failure')
        if self.reject and index == 0:
            raise ProgramCompileError('fixture rejection', {'violations': ['left_elbow_joint']})
        t = fixture_trial(index, mean=1. if index == -1 else (1.05 if index == 0 else 1.2),
                          temporal=1. if index == -1 else (1.05 if index == 0 else 1.2))
        t['id'] = job_id + ('__baseline' if index == -1 else '__trial_' + str(index))
        t['expression'] = expression
        t['paths'] = {'directory': str(output), 'common_reference': str(output / 'common_reference')}
        (output / 'worker_started.json').write_text('{}')
        (output / 'result.json').write_text(json.dumps({'status': 'completed', 'integrity_passed': True}))
        return t


class Provider:
    def status(self): return {'enabled': True}


class RunnerTests(unittest.TestCase):
    def run_fixture(self, *, rejected=False, unsupported=False, runtime_fail=False, planner_reject=False):
        temporary = tempfile.TemporaryDirectory(dir=module.CACHE, prefix='expression-flow-fixture-')
        self.addCleanup(temporary.cleanup)
        runtime = Runtime(rejected, runtime_fail)
        manager = module.ExpressionManager(ROOT, runtime, Provider(), Music(),
            jobs_dir=Path(temporary.name) / 'jobs', integrations=Memory(), renderer=lambda *_args, **_kwargs: {})
        self.addCleanup(manager._pool.shutdown, wait=True)
        captured = []
        def plan(provider, request, context, output, **kwargs):
            captured.append(copy.deepcopy(kwargs.get('feedback')))
            output.mkdir(parents=True)
            (output / 'attempt.json').write_text('{}')
            (output / 'receipt.json').write_text('{"live":true}')
            if planner_reject:
                raise ValueError('Synthetic completed response rejected by local contract')
            return {'supported': not unsupported, 'locally_supported': not unsupported,
                'intent_summary': 'Synthetic control-flow fixture',
                'unsupported_requests': ['outside scope'] if unsupported else [],
                'expression': module.unit_expression(), 'goals': request['goals'],
                'receipt': {'live': True}}
        def renderer(root, runtime, baseline, selected, music, output, **kwargs):
            output.mkdir(parents=True)
            (output / 'comparison.mp4').write_bytes(b'synthetic media fixture, not a movie')
            manifest = {'baseline_id': baseline['id'], 'selected_id': selected['id'],
                        'comparison_sha256': module.sha(output / 'comparison.mp4')}
            (output / 'manifest.json').write_text(json.dumps(manifest))
            return {'comparison': str(output / 'comparison.mp4'), 'manifest': str(output / 'manifest.json')}
        manager.renderer = renderer
        patches = [patch.object(module, 'plan_expression', plan),
                   patch.object(module, 'bridge', lambda *_args: {'passed': True}),
                   patch.object(manager, '_context', lambda *_args: {}),
                   patch.object(manager, '_export', lambda identity, selected: (manager.jobs_dir / identity / 'reference.zip').write_bytes(b'fixture'))]
        for item in patches:
            item.start(); self.addCleanup(item.stop)
        identity = manager.create(REQUEST)
        deadline = time.monotonic() + 5
        while manager.get(identity)['status'] not in module.TERMINAL and time.monotonic() < deadline:
            time.sleep(.01)
        manager._pool.shutdown(wait=True)
        return manager, identity, runtime, captured

    def test_measured_feedback_reaches_second_model_and_stops_on_success(self):
        manager, identity, runtime, calls = self.run_fixture()
        job = manager.get(identity)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['result'], 'success')
        self.assertEqual(runtime.calls, [-1, 0, 1])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0])
        self.assertEqual(calls[1]['fixed_goals'], REQUEST['goals'])
        self.assertEqual(calls[1]['trials'][0]['evaluation']['features'], [1.05, 1.05])
        self.assertEqual(job['new_model_calls'], 2)  # Synthetic receipt fixture, not actual inference.
        self.assertEqual(job['new_physical_trials'], 3)  # Synthetic completed markers only.
        self.assertEqual(job['goals'], REQUEST['goals'])

    def test_compiler_rejection_has_no_measurement_and_budget_slot_is_consumed(self):
        manager, identity, runtime, calls = self.run_fixture(rejected=True)
        job = manager.get(identity)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['trials'][0]['status'], 'rejected')
        self.assertIsNone(job['trials'][0]['metrics'])
        self.assertIsNone(calls[1]['trials'][0]['evaluation'])
        self.assertIn('rejection_report', calls[1]['trials'][0])
        self.assertEqual(job['new_physical_trials'], 2)

    def test_unsupported_intent_stops_before_baseline(self):
        manager, identity, runtime, calls = self.run_fixture(unsupported=True)
        self.assertEqual(manager.get(identity)['status'], 'unsupported')
        self.assertEqual(runtime.calls, [])
        self.assertEqual(len(calls), 1)

    def test_partial_worker_failure_is_not_reported_as_zero_physical_work(self):
        manager, identity, runtime, calls = self.run_fixture(runtime_fail=True)
        job = manager.get(identity)
        self.assertEqual(job['status'], 'failed')
        self.assertTrue(job['finalized'])
        self.assertEqual(job['new_physical_trials'], 1)
        self.assertEqual(job['worker_attempts'], 2)
        self.assertEqual(job['physical_completion_unconfirmed'], 1)
        self.assertEqual(runtime.calls, [-1, 0])

    def test_rejected_completed_model_response_remains_counted(self):
        manager, identity, runtime, calls = self.run_fixture(planner_reject=True)
        job = manager.get(identity)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['model_call_attempts'], 1)
        self.assertEqual(job['new_model_calls'], 1)
        self.assertEqual(job['model_completion_unconfirmed'], 0)
        self.assertEqual(runtime.calls, [])

    def test_changed_video_or_export_is_rejected(self):
        manager, identity, runtime, calls = self.run_fixture()
        path = manager.media_path(identity, 'video')
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError): manager.media_path(identity, 'video')
        (manager.jobs_dir / identity / 'reference.zip').write_bytes(b'changed export')
        with self.assertRaises(ValueError): manager.media_path(identity, 'export')

    def test_corrupt_history_is_preserved_without_breaking_listing(self):
        manager, identity, runtime, calls = self.run_fixture()
        path = manager.jobs_dir / identity / 'state.json'
        bad = json.loads(path.read_bytes()); bad['trials'] = 7
        path.write_text(json.dumps(bad))
        manager._jobs.clear(); manager._load_history()
        self.assertEqual(manager.list_jobs(), [])
        self.assertEqual(len(manager.history_warnings), 1)


if __name__ == '__main__': unittest.main()
