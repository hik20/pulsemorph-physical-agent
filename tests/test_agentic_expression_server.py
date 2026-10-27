"""Loopback HTTP boundary fixtures; no provider or physics runtime."""
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pulsemorph.agentic.expression.server_v1 import Server


class Manager:
    def status(self): return {'ready': True}
    def create(self, value): self.value = value; return 'fixture'
    def cancel(self, identity): self.cancelled = identity
    def get(self, identity): return {'id': identity}
    def list_jobs(self): return []
    def media_path(self, identity, kind): raise RuntimeError('private raw detail must not reach HTTP')


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.manager = Manager(); self.server = Server(0, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)
    def tearDown(self): self.server.shutdown(); self.server.server_close(); self.thread.join()
    def fetch(self, path, *, body=None, headers=None):
        data = None if body is None else body.encode('utf8')
        request = Request(self.base + path, data=data, headers=headers or {})
        try:
            with urlopen(request, timeout=2) as response: return response.status, response.read(), response.headers
        except HTTPError as response: return response.code, response.read(), response.headers
    def mutation_headers(self):
        return {'Content-Type': 'application/json', 'X-PulseMorph-Token': self.server.csrf}
    def test_host_origin_and_session_guards(self):
        self.assertEqual(self.fetch('/api/status', headers={'Host': 'example.invalid'})[0], 403)
        self.assertEqual(self.fetch('/api/status', headers={'Origin': 'https://example.invalid'})[0], 403)
        self.assertEqual(self.fetch('/api/jobs', body='{}', headers={'Content-Type': 'application/json'})[0], 403)
    def test_valid_post_and_cancel_use_structured_bodies(self):
        status, raw, _ = self.fetch('/api/jobs', body='{"instruction":"fixture"}', headers=self.mutation_headers())
        self.assertEqual(status, 202); self.assertEqual(json.loads(raw)['id'], 'fixture')
        self.assertEqual(self.fetch('/api/jobs/fixture/cancel', body='{}', headers=self.mutation_headers())[0], 202)
        self.assertEqual(self.manager.cancelled, 'fixture')
    def test_duplicate_and_nonfinite_json_are_rejected(self):
        for body in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            self.assertEqual(self.fetch('/api/jobs', body=body, headers=self.mutation_headers())[0], 400)
    def test_runtime_failure_returns_safe_category(self):
        status, raw, _ = self.fetch('/media/jobs/fixture/comparison.mp4')
        self.assertEqual(status, 503); self.assertNotIn(b'private raw', raw)
    def test_static_range_and_traversal(self):
        status, raw, headers = self.fetch('/app.js', headers={'Range': 'bytes=0-7'})
        self.assertEqual(status, 206); self.assertEqual(len(raw), 8)
        self.assertEqual(self.fetch('/../runner_v1.py')[0], 404)


if __name__ == '__main__': unittest.main()
