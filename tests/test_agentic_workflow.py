"""Proposal authority and feedback isolation, without model or physics calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'agentic'))
import workflow


class AgenticWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.put('protocol.json', {
            'methods': ['simple_search', 'llm_feedback', 'llm_feedback_off'],
            'trial_budget_per_method': 3,
            'tool': {'source_rate': 1., 'source_start_s': 0., 'duration_s': 4.,
                     'edit_window_s': [1., 3.], 'arm_amplitude_bounds': [0., 1.]},
        })

    def put(self, relative, value):
        path = self.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def observation(self, method='llm_feedback', revision=0):
        return self.put(f'observations/{method}_{revision}.json', {
            'observation_id': f'{method}_{revision}', 'method': method,
            'prior_trial_results': [{'candidate_id': f'{method}_{i}'}
                                    for i in range(revision)],
        })

    def response(self, observation, values, label='answer'):
        return self.put(f'responses/{label}.json', {
            'only_observation_path': str(observation.resolve()),
            'arm_amplitudes': values, 'brief_reason': 'Recorded model decision.',
        })

    def submit(self, method, index, amplitude, obs=None, response=None):
        return workflow.submit(self.out, method, index, amplitude, obs,
                               'Test decision.', response)

    def test_rejects_amplitudes_outside_authorized_tool_domain(self):
        for value in [-.01, 1.01, float('nan'), float('inf'), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.submit('simple_search', 0, value)
        self.assertFalse((self.out / 'proposals').exists())

    def test_budget_is_a_proposal_boundary(self):
        for index in [-1, 3, 100, .5]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.submit('simple_search', index, .5)

    def test_proposal_cannot_change_saved_model_response(self):
        obs = self.observation()
        response = self.response(obs, [.5])
        self.submit('llm_feedback', 0, .5, obs, response)
        next_obs = self.observation(revision=1)
        next_response = self.response(next_obs, [.2], 'next')
        original_bytes = next_response.read_bytes()
        with self.assertRaises(ValueError):
            self.submit('llm_feedback', 1, .8, next_obs, next_response)
        self.assertEqual(next_response.read_bytes(), original_bytes)
        self.assertFalse((self.out / 'proposals/llm_feedback_1.json').exists())

    def test_feedback_requires_the_matching_observation_revision(self):
        obs = self.observation()
        response = self.response(obs, [.5])
        self.submit('llm_feedback', 0, .5, obs, response)
        stale_response = self.response(obs, [.2], 'stale')
        with self.assertRaises(ValueError):
            self.submit('llm_feedback', 1, .2, obs, stale_response)
        next_obs = self.observation(revision=1)
        next_response = self.response(next_obs, [.2], 'next')
        proposal = self.submit('llm_feedback', 1, .2, next_obs, next_response)
        self.assertEqual(proposal['observation_id'], 'llm_feedback_1')
        self.assertEqual(proposal['response_artifact']['sha256'], workflow.sha256(next_response))

    def test_response_cannot_substitute_another_observation(self):
        obs = self.observation()
        other = self.observation(revision=1)
        response = self.response(other, [.5])
        with self.assertRaises(ValueError):
            self.submit('llm_feedback', 0, .5, obs, response)
        wrong_method = self.observation('llm_feedback_off')
        wrong_response = self.response(wrong_method, [.5])
        with self.assertRaises(ValueError):
            self.submit('llm_feedback', 0, .5, wrong_method, wrong_response)

    def test_committed_proposal_is_exclusive_and_keeps_its_bytes(self):
        self.submit('simple_search', 0, .5)
        path = self.out / 'proposals/simple_search_0.json'
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.submit('simple_search', 0, .5)
        self.assertEqual(path.read_bytes(), before)

    def test_feedback_off_uses_all_three_precommitted_values(self):
        obs = self.observation('llm_feedback_off')
        response = self.response(obs, [.5, .1, .9])
        self.submit('llm_feedback_off', 0, .5, obs, response)
        with self.assertRaises(ValueError):
            self.submit('llm_feedback_off', 1, .8, obs, response)
        self.submit('llm_feedback_off', 1, .1, obs, response)
        self.submit('llm_feedback_off', 2, .9, obs, response)
        self.assertEqual(workflow.read(self.out / 'proposals/llm_feedback_off_2.json')['arm_amplitude'], .9)

    def test_feedback_off_rejects_commitment_after_candidate_outcome(self):
        obs = self.observation('llm_feedback_off')
        response = self.response(obs, [.5, .1, .9])
        self.submit('llm_feedback_off', 0, .5, obs, response)
        self.put('trials/llm_feedback_0/manifest.json', {'status': 'success'})
        with self.assertRaises(ValueError):
            self.submit('llm_feedback_off', 1, .1, obs, response)

    def test_feedback_cannot_skip_prior_commitment(self):
        obs = self.observation(revision=1)
        response = self.response(obs, [.2])
        with self.assertRaises(ValueError):
            self.submit('llm_feedback', 1, .2, obs, response)

    def test_simple_search_solves_linear_response_and_ignores_invalid_trial(self):
        # Under identity response, calibration (1,1) and trial (.5,.5)
        # identify the target .7 exactly; malformed physics is not evidence.
        good = {'arm_amplitude': .5, 'intent_ratio': .5, 'integrity_passed': True}
        bad = {'arm_amplitude': .9, 'intent_ratio': .7, 'integrity_passed': False}
        self.assertAlmostEqual(workflow.simple_amplitude([good]), .7)
        self.assertAlmostEqual(workflow.simple_amplitude([good, bad]), .7)
        self.assertEqual(workflow.simple_amplitude([]), .5)


if __name__ == '__main__':
    unittest.main()
