"""Descriptor context and read-only route boundaries; no real model or physics."""
import copy
import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_agentic_app_phrase_manager import MusicFixture, ProviderFixture, RuntimeFixture, MemoryFixture, renderer, PROGRAM
from test_agentic_app_phrase_server import FakeManager
from pulsemorph.agentic.app.phrase.manager_descriptor_v6 import PhraseDescriptorManager
from pulsemorph.agentic.app.phrase.manager_advice_v5 import PhraseAdviceManager
from pulsemorph.agentic.app.phrase.server_descriptor_v6 import make_descriptor_server

DESCRIPTOR = {'schema': 'pulsemorph.phrase_source_descriptor.v1', 'definitions': {'q0_displacement_rms_rad': 'source initial pose distance, not temporal variation'}, 'phases': []}
GOALS = {'schema': 'pulsemorph.phrase_goals.v1', 'ratios': [.75, 1.25, 1.25, .75]}


class DescriptorContextTests(unittest.TestCase):
    def test_descriptor_is_saved_before_provider_and_prepared_prompt_is_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            initial = {'supported': False, 'intent_summary': 'fixture unsupported', 'unsupported_requests': ['fixture unsupported guarantee'], 'program': copy.deepcopy(PROGRAM), 'goals': copy.deepcopy(GOALS)}
            provider = ProviderFixture(initial=initial)
            runtime = RuntimeFixture()
            runtime.assets = object()
            manager = PhraseDescriptorManager(Path(temp), provider, runtime, MusicFixture(), MemoryFixture(), renderer, Path(temp) / 'app')
            self.addCleanup(manager.shutdown)
            original = provider.generate_json
            def checked(*args, **kwargs):
                job = next(iter(manager._jobs.values()))
                path = manager.jobs_dir / job['id'] / 'source_descriptor.json'
                self.assertTrue(path.is_file())
                self.assertEqual(job['targets'], GOALS['ratios'])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), job['source_descriptor_receipt']['sha256'])
                self.assertFalse(job['source_descriptor_receipt']['physical_response_or_feasibility'])
                return original(*args, **kwargs)
            provider.generate_json = checked
            with patch('pulsemorph.agentic.app.phrase.manager_descriptor_v6.describe_sources', return_value=copy.deepcopy(DESCRIPTOR)):
                identity = manager.create({'instruction': 'fixture unsupported guarantee', 'music_id': 'fixture_music', 'max_iterations': 1, 'goals': copy.deepcopy(GOALS)})
                for _ in range(300):
                    job = manager.get(identity)
                    if job['status'] in ('complete', 'failed', 'cancelled'):
                        break
                    time.sleep(.01)
            self.assertEqual(job['outcome'], 'unsupported')
            self.assertEqual(runtime.calls, [])
            self.assertEqual(len(provider.calls), 1)
            prepared = json.loads((manager.jobs_dir / identity / 'prepared_plan_v4.json').read_bytes())
            self.assertEqual(prepared['prompt'], provider.calls[0][0])
            data = json.loads(prepared['prompt'].rsplit('\n', 1)[1])
            self.assertEqual(data['historical_context']['source_reference_descriptor'], DESCRIPTOR)
            self.assertEqual(data['explicit_goals'], GOALS)

    def test_context_is_owned_and_descriptor_receipt_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = object.__new__(PhraseDescriptorManager)
            manager.jobs_dir = Path(temp)
            manager._check = Mock()
            manager._update = Mock()
            manager.describe_sources = Mock(return_value=copy.deepcopy(DESCRIPTOR))
            original = {'local_measured_examples': [{'metrics': {'fixture': 1}}]}
            with patch.object(PhraseAdviceManager, '_context', return_value=original):
                context = manager._context('fixture')
                self.assertNotIn('source_reference_descriptor', original)
                context['local_measured_examples'][0]['metrics']['fixture'] = 2
                self.assertEqual(original['local_measured_examples'][0]['metrics']['fixture'], 1)
                path = Path(temp) / 'fixture/source_descriptor.json'
                before = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    manager._context('fixture')
                self.assertEqual(path.read_bytes(), before)

    def test_source_integrity_or_startup_error_propagates_without_fabricated_context(self):
        manager = object.__new__(PhraseDescriptorManager)
        manager.runtime = type('Runtime', (), {'assets': object()})()
        manager._verify_startup = Mock(side_effect=ValueError('fixture source changed'))
        with patch('pulsemorph.agentic.app.phrase.manager_descriptor_v6.describe_sources') as describe:
            with self.assertRaises(ValueError):
                manager.describe_sources()
            describe.assert_not_called()


class DescriptorHTTPTests(unittest.TestCase):
    def setUp(self):
        self.old, self.manager = FakeManager('old_'), FakeManager('phrase_')
        self.manager.describe_sources = Mock(return_value=copy.deepcopy(DESCRIPTOR))
        self.server = make_descriptor_server(('127.0.0.1', 0), self.old, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
    def call(self, path, origin=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        connection.request('GET', path, headers={'Origin': origin} if origin else {})
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result
    def test_readonly_descriptor_and_v6_script_keep_origin_guard(self):
        status, data = self.call('/api/phrase/source-descriptor')
        self.assertEqual((status, json.loads(data)), (200, DESCRIPTOR))
        self.assertEqual(self.call('/api/phrase/source-descriptor', 'https://example.invalid')[0], 403)
        self.assertIn(b'phrase_v6.js', self.call('/phrase')[1])
        self.assertIn(b'source-descriptor', self.call('/static/phrase_v6.js')[1])
        self.assertEqual(self.old.creates + self.manager.creates, 0)
    def test_missing_descriptor_is_safe_and_does_not_change_core_status(self):
        self.manager.describe_sources.side_effect = ValueError('C:/private/fixture-key-like-value')
        status, data = self.call('/api/phrase/source-descriptor')
        self.assertEqual(status, 503)
        self.assertNotIn(b'private', data)
        self.assertEqual(self.call('/api/phrase/status')[0], 200)
        self.assertEqual(self.old.creates + self.manager.creates, 0)


if __name__ == '__main__':
    unittest.main()
