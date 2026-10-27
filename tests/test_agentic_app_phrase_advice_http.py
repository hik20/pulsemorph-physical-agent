"""Versioned advice HTTP dispatch, mocked managers/export payload only."""
import http.client
import json
import sys
import threading
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_agentic_app_phrase_server import FakeManager
from pulsemorph.agentic.app.phrase.server_advice_v5 import make_advice_server, AdviceHandler
from pulsemorph.agentic.app.phrase.server import PhraseHandler


class AdviceHTTPTests(unittest.TestCase):
    def setUp(self):
        self.old, self.manager = FakeManager('old_'), FakeManager('phrase_')
        self.server = make_advice_server(('127.0.0.1', 0), self.old, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def call(self, method, path, body=b'', token=True):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['X-PulseMorph-Token'] = self.server.session_token
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        result = response.status, response.getheader('Content-Type'), response.read()
        connection.close()
        return result

    def test_v5_static_and_explicit_goal_create_keep_existing_auth_and_shared_lock(self):
        status, _, data = self.call('GET', '/phrase')
        self.assertEqual(status, 200)
        self.assertIn(b'phrase_v5.js', data)
        self.assertIn(b'conditional_gain_advice', self.call('GET', '/static/phrase_v5.js')[2])
        body = {'instruction': 'fixture', 'music_id': 'fixture', 'max_iterations': 3,
                'goals': {'schema': 'pulsemorph.phrase_goals.v1', 'ratios': [.75, 1.25, 1.25, .75]}}
        with patch.object(self.manager, 'create', return_value='fixture') as create:
            self.assertEqual(self.call('POST', '/api/phrase/jobs', json.dumps(body).encode(), token=False)[0], 403)
            self.old.jobs = [{'id': 'active', 'status': 'running'}]
            self.assertEqual(self.call('POST', '/api/phrase/jobs', json.dumps(body).encode())[0], 409)
            self.old.jobs = []
            self.assertEqual(self.call('POST', '/api/phrase/jobs', json.dumps(body).encode())[0], 202)
            create.assert_called_once_with(body)

    def test_export_still_dispatches_frozen_goal_aware_builder_without_execution(self):
        module = ModuleType('pulsemorph.agentic.app.phrase.exports_goals_v4')
        calls = []
        def build(manager, identity):
            calls.append((manager, identity))
            return b'PK-fixture'
        module.build_phrase_result_bundle = build
        with patch.dict(sys.modules, {module.__name__: module}):
            status, kind, data = self.call('POST', '/api/phrase/jobs/final_fixture/export', b'{}')
        self.assertEqual((status, kind, data), (200, 'application/zip', b'PK-fixture'))
        self.assertEqual(calls, [(self.manager, 'final_fixture')])
        self.assertEqual(self.old.creates + self.manager.creates, 0)

    def test_closed_browser_does_not_attempt_a_second_json_error_response(self):
        handler = object.__new__(AdviceHandler)
        with patch.object(PhraseHandler, '_json', side_effect=ConnectionAbortedError('fixture closed peer')) as write:
            self.assertIsNone(handler._json(200, {'status': 'fixture'}))
            write.assert_called_once_with(200, {'status': 'fixture'})


if __name__ == '__main__':
    unittest.main()
