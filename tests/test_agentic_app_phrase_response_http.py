"""Read-only evidence HTTP boundary; no provider or physical runtime."""
import http.client
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_agentic_app_phrase_server import FakeManager
from pulsemorph.agentic.app.phrase.server_response_v8 import make_response_server


class ResponseHTTPTests(unittest.TestCase):
    def setUp(self):
        self.old, self.manager = FakeManager('old_'), FakeManager('phrase_')
        self.manager.describe_response_evidence = Mock(return_value={'status': 'unavailable'})
        self.manager.query_response_evidence = Mock(return_value={'status': 'not_applicable', 'points': []})
        self.server = make_response_server(('127.0.0.1', 0), self.old, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.assertEqual(self.old.creates + self.manager.creates, 0)
    def call(self, method, path, raw=None, token=True, origin=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['X-PulseMorph-Token'] = self.server.session_token
        if origin:
            headers['Origin'] = origin
        connection.request(method, path, body=raw, headers=headers)
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data
    def test_catalog_is_readonly_and_origin_guarded(self):
        path = '/api/phrase/response-evidence'
        self.assertEqual(json.loads(self.call('GET', path)[1]), {'status': 'unavailable'})
        self.assertEqual(self.call('GET', path, origin='https://example.invalid')[0], 403)
        self.assertEqual(self.call('GET', path + '?field=first_left_gain')[0], 400)
        self.assertEqual(self.call('GET', '/api/phrase/status')[0], 200)
    def test_query_requires_session_and_origin_but_creates_no_job(self):
        path = '/api/phrase/response-evidence/query'
        self.assertEqual(self.call('POST', path, b'{}', token=False)[0], 403)
        self.assertEqual(self.call('POST', path, b'{}', origin='https://example.invalid')[0], 403)
        self.manager.query_response_evidence.assert_not_called()
        self.assertEqual(self.call('POST', path, b'{}')[0], 200)
        self.manager.query_response_evidence.assert_called_once_with({})
    def test_ambiguous_nonfinite_oversized_or_nonobject_body_never_reaches_query(self):
        path = '/api/phrase/response-evidence/query'
        for raw in (b'{"gain":1,"gain":0.75}', b'{"gain":NaN}', b'[]', b'null', b'{}' + b' ' * 4096):
            self.assertEqual(self.call('POST', path, raw)[0], 400)
        self.manager.query_response_evidence.assert_not_called()
    def test_query_errors_do_not_leak_integrity_paths(self):
        path = '/api/phrase/response-evidence/query'
        for kind, expected in ((ValueError, 400), (RuntimeError, 503)):
            self.manager.query_response_evidence.side_effect = kind('C:/private/fixture-path')
            status, data = self.call('POST', path, b'{}')
            self.assertEqual(status, expected)
            self.assertNotIn(b'private', data)
        self.manager.describe_response_evidence.side_effect = RuntimeError('C:/private/fixture-path')
        status, data = self.call('GET', '/api/phrase/response-evidence')
        self.assertEqual(status, 503)
        self.assertNotIn(b'private', data)


if __name__ == '__main__':
    unittest.main()
