"""Full-record recheck is explicit/authenticated; no inference or simulation."""
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
from pulsemorph.agentic.app.phrase.server_snapshot_v9 import make_snapshot_server


class SnapshotHTTPTests(unittest.TestCase):
    def setUp(self):
        self.old, self.manager = FakeManager('old_'), FakeManager('phrase_')
        self.manager.describe_response_evidence = Mock(return_value={'status': 'unavailable'})
        self.manager.query_response_evidence = Mock(return_value={'status': 'not_applicable', 'points': []})
        self.manager.recheck_response_evidence = Mock(return_value={'schema': 'pulsemorph.response_snapshot_recheck.v1', 'status': 'matched'})
        self.server = make_snapshot_server(('127.0.0.1', 0), self.old, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.assertEqual(self.old.creates + self.manager.creates, 0)
    def call(self, path='/api/phrase/response-evidence/recheck', raw=b'{}', token=True, origin=None, method='POST'):
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
    def test_recheck_requires_explicit_session_origin_and_empty_body(self):
        self.assertEqual(self.call(token=False)[0], 403)
        self.assertEqual(self.call(origin='https://example.invalid')[0], 403)
        for raw in (b'[]', b'{"force":true}', b'{"x":1,"x":2}', b'{}' + b' ' * 1024):
            self.assertEqual(self.call(raw=raw)[0], 400)
        self.manager.recheck_response_evidence.assert_not_called()
        self.assertEqual(self.call()[0], 200)
        self.manager.recheck_response_evidence.assert_called_once_with()
    def test_recheck_errors_are_safe_and_do_not_masquerade_as_matched(self):
        self.manager.recheck_response_evidence.side_effect = RuntimeError('C:/private/fixture-path')
        status, data = self.call()
        self.assertEqual(status, 503)
        self.assertNotIn(b'private', data)
        self.assertNotIn(b'matched', data)
    def test_fast_catalog_and_query_do_not_trigger_recheck(self):
        self.assertEqual(self.call('/api/phrase/response-evidence', raw=None, method='GET')[0], 200)
        self.assertEqual(self.call('/api/phrase/response-evidence/query')[0], 200)
        self.manager.recheck_response_evidence.assert_not_called()
        self.assertEqual(self.call('/api/phrase/response-evidence/recheck', raw=None, method='GET')[0], 404)


if __name__ == '__main__':
    unittest.main()
