"""Actual ephemeral loopback HTTP tests with a fixture manager; no job execution."""
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from pulsemorph.agentic.app.orchestrator import JobBusy
from pulsemorph.agentic.app.server import StudioServer


class FixtureManager:
    def __init__(self, root):
        self.jobs_dir = Path(root) / "jobs"
        self.jobs_dir.mkdir()
        self.audio = Path(root) / "fixture.wav"
        self.audio.write_bytes(bytes(range(100)))
        self.records, self.created, self.cancelled = {}, [], []
        self.busy = False
        def get_music(identity):
            if identity != "pulse": raise KeyError(identity)
            return {"id": identity, "path": str(self.audio), "accent_time_s": 2.405}
        self.music = SimpleNamespace(get=get_music)
        self.integrations = SimpleNamespace(list_memory=lambda: [{"evidence_class": "fixture"}], status=lambda: {"meterless": {"state": "fixture"}})
    def status(self): return {"live": False, "fixture": True}
    def list_jobs(self): return list(self.records.values())
    def get(self, identity): return self.records[identity]
    def create(self, body):
        if not isinstance(body, dict) or set(body) != {"instruction"}: raise ValueError("fixture invalid fields")
        if self.busy: raise JobBusy("fixture busy")
        identity = "fixture-job"
        self.created.append(body)
        self.records[identity] = {"id": identity, "status": "queued", "live": False}
        return identity
    def cancel(self, identity):
        record = self.get(identity)
        self.cancelled.append(identity)
        record["status"] = "cancelled"
        return record


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.manager = FixtureManager(self.temporary.name)
        self.server = StudioServer(("127.0.0.1", 0), self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        self.thread.start()
        self.token = self.server.session_token

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None, token=True):
        supplied = dict(headers or {})
        if token and method == "POST": supplied.setdefault("X-PulseMorph-Token", self.token)
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            supplied.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.request(method, path, body, supplied)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_loopback_binding_and_status_session_token(self):
        with self.assertRaises(ValueError): StudioServer(("0.0.0.0", 0), self.manager)
        status, headers, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["session_token"], self.token)
        self.assertFalse(json.loads(body)["live"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertIn("media-src 'self' blob:;", headers["Content-Security-Policy"])

    def test_post_requires_correct_token(self):
        for headers in [{}, {"X-PulseMorph-Token": "wrong"}]:
            with self.subTest(headers=headers):
                self.assertEqual(self.request("POST", "/api/jobs", {"instruction": "fixture"}, headers=headers, token=False)[0], 403)
        self.assertEqual(self.manager.created, [])

    def test_small_early_rejections_reach_client_without_connection_reset(self):
        raw = b"fixture-wav-bytes-not-live-audio"
        for index in range(20):
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
            try:
                connection.putrequest("POST", "/api/music/import")
                connection.putheader("Content-Type", "audio/wav")
                connection.putheader("Content-Length", str(len(raw)))
                connection.endheaders()
                # Split headers/body to exercise the early rejection race.
                if index % 2: time.sleep(.01)
                connection.send(raw)
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                self.assertIn("error", json.loads(response.read()))
            finally:
                connection.close()
        self.assertEqual(self.manager.created, [])

    def test_foreign_host_and_origin_cannot_read_token_or_mutate(self):
        for headers in [{"Host": "evil.invalid"}, {"Origin": "https://evil.invalid"}, {"Host": "127.0.0.1.evil.invalid"}, {"Origin": "null"}]:
            with self.subTest(headers=headers):
                status, _, body = self.request("GET", "/api/status", headers=headers)
                self.assertEqual(status, 403)
                self.assertNotIn(self.token.encode(), body)
                self.assertEqual(self.request("POST", "/api/jobs", {"instruction": "fixture"}, headers=headers)[0], 403)
        self.assertEqual(self.manager.created, [])

    def test_same_origin_create_get_cancel_and_memory(self):
        origin = f"http://127.0.0.1:{self.server.server_port}"
        status, _, body = self.request("POST", "/api/jobs", {"instruction": "fixture"}, headers={"Origin": origin})
        self.assertEqual(status, 202)
        identity = json.loads(body)["job_id"]
        self.assertEqual(self.request("GET", f"/api/jobs/{identity}")[0], 200)
        self.assertEqual(self.request("GET", "/api/jobs")[0], 200)
        status, _, body = self.request("POST", f"/api/jobs/{identity}/cancel", {})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "cancelled")
        self.assertEqual(self.manager.cancelled, [identity])
        self.assertEqual(self.request("GET", "/api/memory")[0], 200)
        info = json.loads(self.request("GET", "/api/music/pulse")[2])
        self.assertNotIn("path", info)

    def test_result_export_requires_session_and_delivers_zip_attachment(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("README.txt", "HTTP transport fixture only; no motion or physics")
        payload = buffer.getvalue()
        with patch("pulsemorph.agentic.app.exports.build_result_bundle", return_value=payload) as build:
            self.assertEqual(self.request("GET", "/api/jobs/fixture-job/export")[0], 404)
            self.assertEqual(self.request("POST", "/api/jobs/fixture-job/export", {}, token=False)[0], 403)
            self.assertEqual(self.request("POST", "/api/jobs/fixture-job/export", {"path": "../private"})[0], 400)
            build.assert_not_called()
            status, headers, body = self.request("POST", "/api/jobs/fixture-job/export", {})
            self.assertEqual((status, body), (200, payload))
            self.assertEqual(headers["Content-Type"], "application/zip")
            self.assertEqual(headers["Content-Disposition"], 'attachment; filename="PulseMorph-fixture-job-result.zip"')
            self.assertEqual(headers["Cache-Control"], "no-store")
            build.assert_called_once_with(self.manager, "fixture-job")
        self.assertEqual((self.manager.created, self.manager.cancelled), ([], []))

    def test_result_export_busy_and_failure_release_the_download_slot(self):
        with patch("pulsemorph.agentic.app.exports.build_result_bundle", side_effect=ValueError("Kayıt doğrulanamadı")) as build:
            self.server.export_lock.acquire()
            try:
                self.assertEqual(self.request("POST", "/api/jobs/fixture-job/export", {})[0], 409)
                build.assert_not_called()
            finally:
                self.server.export_lock.release()
            self.assertEqual(self.request("POST", "/api/jobs/fixture-job/export", {})[0], 400)
            self.assertTrue(self.server.export_lock.acquire(blocking=False))
            self.server.export_lock.release()

    def test_invalid_json_fields_type_and_size_rejected(self):
        cases = [({"extra": 1}, {}), ([], {}), (b"{invalid", {"Content-Type": "application/json"}),
                 (b"{}", {"Content-Type": "text/plain"}), (b"", {"Content-Length": "16001"}),
                 (b"", {"Content-Length": "-1"}), (b"", {"Content-Length": "bad"})]
        for body, headers in cases:
            with self.subTest(body=body, headers=headers):
                self.assertEqual(self.request("POST", "/api/jobs", body, headers)[0], 400)
        self.assertEqual(self.manager.created, [])

    def test_busy_and_missing_jobs_have_specific_status(self):
        self.manager.busy = True
        self.assertEqual(self.request("POST", "/api/jobs", {"instruction": "fixture"})[0], 409)
        self.assertEqual(self.request("GET", "/api/jobs/missing")[0], 404)
        self.assertEqual(self.request("POST", "/api/jobs/missing/cancel", {})[0], 404)
        self.assertEqual(self.request("POST", "/missing", {})[0], 404)

    def test_traversal_and_unknown_routes_never_serve_files(self):
        for path in ["/missing", "/api/music/unknown", "/media/music/unknown.wav", "/static/../server.py", "/static/%2e%2e/server.py", "/media/jobs/../state.json", "/media/jobs/%2e%2e/before.mp4", "/media/jobs/missing/before.mp4"]:
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)

    def test_media_byte_ranges_return_exact_bytes(self):
        for request_range, expected_range, expected in [("bytes=2-6", "bytes 2-6/100", bytes(range(2, 7))), ("bytes=95-", "bytes 95-99/100", bytes(range(95, 100))), ("bytes=98-200", "bytes 98-99/100", bytes(range(98, 100)))]:
            status, headers, body = self.request("GET", "/media/music/pulse.wav", headers={"Range": request_range})
            self.assertEqual(status, 206)
            self.assertEqual(body, expected)
            self.assertEqual(headers["Content-Range"], expected_range)
            self.assertEqual(int(headers["Content-Length"]), len(expected))
        status, headers, body = self.request("GET", "/media/music/pulse.wav")
        self.assertEqual((status, body), (200, bytes(range(100))))
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        for invalid in ["bytes=100-", "bytes=9-3", "bytes=0-1,3-4", "garbage"]:
            self.assertEqual(self.request("GET", "/media/music/pulse.wav", headers={"Range": invalid})[0], 416)

    def test_raw_wav_import_token_fields_and_no_private_path(self):
        calls = []
        def imported(data, label, start_s):
            calls.append((data, label, start_s))
            return {"id": "upload_abcdef123456", "label": label, "path": str(self.manager.audio), "duration_s": 4}
        self.manager.music.import_wav = imported
        self.manager.music.catalog = lambda: [{"id": "pulse"}, {"id": "upload_abcdef123456"}]
        raw = b"fixture-wav-bytes-not-live-audio"
        headers = {"Content-Type": "audio/wav"}
        self.assertEqual(self.request("POST", "/api/music/import", raw, headers, token=False)[0], 403)
        status, _, body = self.request("POST", "/api/music/import?start_s=1.5&label=User%20track", raw, headers)
        self.assertEqual(status, 201)
        self.assertEqual(calls, [(raw, "User track", 1.5)])
        self.assertNotIn("path", json.loads(body)["music"])
        for path in ["/api/music/import?start_s=no", "/api/music/import?start_s=1&start_s=2", "/api/music/import?extra=1"]:
            self.assertEqual(self.request("POST", path, raw, headers)[0], 400)
        self.assertEqual(self.request("POST", "/api/music/import", raw, {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/music/import", b"", {**headers, "Content-Length": str(32*1024*1024+1)})[0], 400)
        self.manager.records["active"] = {"status": "running"}
        self.assertEqual(self.request("POST", "/api/music/import", raw, headers)[0], 409)

    def test_uploaded_music_ids_are_readable_without_serving_arbitrary_paths(self):
        self.manager.music.get = lambda identity: {"id": identity, "path": str(self.manager.audio)}
        self.assertEqual(self.request("GET", "/api/music/upload_123abc")[0], 200)
        self.assertEqual(self.request("GET", "/media/music/upload_123abc.wav")[2], bytes(range(100)))


if __name__ == "__main__":
    unittest.main()
