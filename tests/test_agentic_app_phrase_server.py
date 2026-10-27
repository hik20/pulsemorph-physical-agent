"""HTTP admission across both private studios; no model or physical runtime."""
import http.client
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pulsemorph.agentic.app.phrase.server import PhraseStudioServer


class FakeMusic:
    def __init__(self):
        self.imports = 0
    def catalog(self):
        return []
    def import_wav(self, data, label, start):
        self.imports += 1
        return {'id': 'phrase_fixture', 'path': 'private-unused-path', 'duration_s': 8}


class FakeManager:
    def __init__(self, prefix):
        self.jobs = []
        self.prefix = prefix
        self.music = FakeMusic()
        self.creates = 0
        self.jobs_dir = ROOT / 'artifacts/runs/agentic/not_created_http_fixture'
    def status(self):
        return {'provider': {'enabled': True}, 'music_catalog': [], 'jobs': len(self.jobs)}
    def list_jobs(self):
        return list(self.jobs)
    def create(self, body):
        self.creates += 1
        job = {'id': self.prefix + str(self.creates), 'status': 'queued'}
        self.jobs.append(job)
        return job['id']
    def get(self, identity):
        for job in self.jobs:
            if job['id'] == identity:
                return job
        raise KeyError(identity)
    def cancel(self, identity):
        job = self.get(identity)
        job['status'] = 'cancelled'
        return job


class PhraseHTTPTests(unittest.TestCase):
    def setUp(self):
        self.disk = patch('pulsemorph.agentic.app.phrase.server.shutil.disk_usage', return_value=SimpleNamespace(free=4 * 1024**3))
        self.disk.start()
        self.old, self.phrase = FakeManager('old_'), FakeManager('phrase_')
        self.server = PhraseStudioServer(('127.0.0.1', 0), self.old, self.phrase)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.disk.stop()
    def call(self, method, path, body=None, *, token=True, origin=None, kind='application/json'):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        headers = {'Content-Type': kind}
        if token:
            headers['X-PulseMorph-Token'] = self.server.session_token
        if origin:
            headers['Origin'] = origin
        payload = json.dumps(body).encode() if body is not None else b''
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        value = response.read()
        status, content_type = response.status, response.getheader('Content-Type')
        connection.close()
        return status, json.loads(value) if 'application/json' in content_type else value
    def test_bootstrap_and_token_origin_rejections(self):
        status, body = self.call('GET', '/api/phrase/status')
        self.assertEqual(status, 200)
        self.assertEqual(body['session_token'], self.server.session_token)
        self.assertEqual(body['tool_catalog'][0]['supported_order'], ['wave', 'high5'])
        self.assertEqual(self.call('POST', '/api/phrase/jobs', {}, token=False)[0], 403)
        self.assertEqual(self.call('POST', '/api/phrase/jobs', {}, origin='https://example.invalid')[0], 403)
        self.assertEqual(self.phrase.creates, 0)
    def test_phrase_job_blocks_old_and_cancel_releases_admission(self):
        status, result = self.call('POST', '/api/phrase/jobs', {})
        self.assertEqual(status, 202)
        self.assertEqual(self.call('POST', '/api/jobs', {})[0], 409)
        self.assertEqual(self.old.creates, 0)
        self.assertEqual(self.call('POST', '/api/phrase/jobs/' + result['job_id'] + '/cancel', {})[0], 200)
        self.assertEqual(self.call('POST', '/api/jobs', {})[0], 202)
        self.assertEqual(self.call('POST', '/api/phrase/jobs', {})[0], 409)
    def test_queued_old_blocks_phrase_and_audio_import(self):
        self.old.jobs = [{'id': 'prior', 'status': 'queued'}]
        self.assertEqual(self.call('POST', '/api/phrase/jobs', {})[0], 409)
        self.assertEqual(self.call('POST', '/api/phrase/music/import?start_s=0', {}, kind='audio/wav')[0], 409)
        self.assertEqual(self.phrase.music.imports, 0)
    def test_old_storage_gate_precedes_model_job_creation(self):
        with patch('pulsemorph.agentic.app.phrase.server.shutil.disk_usage', return_value=SimpleNamespace(free=100)):
            self.assertEqual(self.call('POST', '/api/jobs', {})[0], 507)
            self.assertEqual(self.old.creates, 0)
            self.assertFalse(self.call('GET', '/api/status')[1]['runtime_disk']['available'])
            self.assertEqual(self.call('POST', '/api/phrase/jobs', {})[0], 202)
    def test_cancel_payload_extra_and_unknown_routes_rejected(self):
        self.phrase.jobs = [{'id': 'p1', 'status': 'queued'}]
        self.assertEqual(self.call('POST', '/api/phrase/jobs/p1/cancel', {'extra': True})[0], 400)
        self.assertEqual(self.phrase.jobs[0]['status'], 'queued')
        self.assertEqual(self.call('GET', '/api/phrase/jobs/missing')[0], 404)
        self.assertEqual(self.call('GET', '/media/phrase/jobs/../../secret.mp4')[0], 404)
        self.assertEqual(self.call('POST', '/api/phrase/jobs/p1/export', {})[0], 404)
    def test_same_creation_lock_admits_only_one_simultaneous_request(self):
        barrier = threading.Barrier(3)
        results = []
        def submit(path):
            barrier.wait()
            results.append(self.call('POST', path, {})[0])
        workers = [threading.Thread(target=submit, args=(path,)) for path in ('/api/jobs', '/api/phrase/jobs')]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join()
        self.assertEqual(sorted(results), [202, 409])
        self.assertEqual(self.old.creates + self.phrase.creates, 1)


if __name__ == '__main__':
    unittest.main()
