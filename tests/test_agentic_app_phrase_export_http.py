"""Saved-export HTTP boundary, fake ZIP only; no model or physical call."""
from pathlib import Path
import http.client
import json
import sys
import threading
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_agentic_app_phrase_server import FakeManager
from pulsemorph.agentic.app.phrase.server_export_v3 import make_export_server


class ExportHTTPTests(unittest.TestCase):
    def setUp(self):
        self.old, self.phrase = FakeManager('old_'), FakeManager('phrase_')
        self.calls = []
        self.failure = None
        self.stub = ModuleType('pulsemorph.agentic.app.phrase.exports_v1')
        def build(manager, identity):
            self.calls.append((manager, identity))
            if self.failure:
                raise self.failure
            return b'PK-fixture-no-real-archive'
        self.stub.build_phrase_result_bundle = build
        self.module_patch = patch.dict(sys.modules, {self.stub.__name__: self.stub})
        self.module_patch.start()
        self.server = make_export_server(('127.0.0.1', 0), self.old, self.phrase)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.module_patch.stop()
    def request(self, *, body=b'{}', token=True, method='POST', origin=None, kind='application/json'):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        headers = {'Content-Type': kind}
        if token:
            headers['X-PulseMorph-Token'] = self.server.session_token
        if origin:
            headers['Origin'] = origin
        conn.request(method, '/api/phrase/jobs/fixture_final/export', body, headers)
        response = conn.getresponse()
        data = response.read()
        result = (response.status, dict(response.getheaders()), data)
        conn.close()
        return result
    def test_authenticated_post_uses_final_job_and_attachment_without_creation(self):
        status, headers, blob = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/zip')
        self.assertIn('PulseMorph-8s-fixture_final-result.zip', headers['Content-Disposition'])
        self.assertEqual(blob, b'PK-fixture-no-real-archive')
        self.assertEqual(self.calls, [(self.phrase, 'fixture_final')])
        self.assertEqual(self.old.creates + self.phrase.creates, 0)
    def test_get_does_not_export_and_cross_origin_or_token_denied(self):
        self.assertEqual(self.request(method='GET')[0], 404)
        self.assertEqual(self.request(token=False)[0], 403)
        self.assertEqual(self.request(origin='https://example.invalid')[0], 403)
        self.assertEqual(self.calls, [])
    def test_only_small_empty_json_object_is_accepted(self):
        for body in (b'', b'[]', b'null', b'{"extra":1}', b'{' * 1025, b'not-json'):
            with self.subTest(body=body[:20]):
                self.assertEqual(self.request(body=body)[0], 400)
        self.assertEqual(self.request(kind='text/plain')[0], 400)
        self.assertEqual(self.calls, [])
    def test_shared_export_lock_rejects_reentry_and_is_released_on_failure(self):
        self.server.export_lock.acquire()
        try:
            self.assertEqual(self.request()[0], 409)
        finally:
            self.server.export_lock.release()
        self.failure = ValueError('private path not shown')
        status, _, data = self.request()
        self.assertEqual(status, 400)
        self.assertNotIn(b'private path', data)
        self.assertTrue(self.server.export_lock.acquire(blocking=False))
        self.server.export_lock.release()
    def test_missing_job_and_internal_failure_are_safe_json(self):
        for failure, status in ((KeyError('private'), 404), (RuntimeError('secret-like-diagnostic'), 503)):
            self.failure = failure
            code, headers, data = self.request()
            self.assertEqual(code, status)
            self.assertIn('application/json', headers['Content-Type'])
            self.assertNotIn(b'secret-like', data)
            self.assertIn('error', json.loads(data))


if __name__ == '__main__':
    unittest.main()
