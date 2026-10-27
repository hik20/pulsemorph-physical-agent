"""Loopback-only, single-worker HTTP entry point for the private motion studio."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import parse_qs, unquote, urlsplit

from .orchestrator import JobBusy, JobManager


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager):
        if address[0] != "127.0.0.1":
            raise ValueError("This application is private and must bind to 127.0.0.1")
        self.manager = manager
        self.session_token = secrets.token_urlsafe(32)
        self.export_lock = threading.Lock()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "PulseMorphLocal/0.1"

    def log_message(self, format, *args):
        # Request bodies and tokens never enter access logs.
        pass

    def _trusted_origin(self, *, mutation=False):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in hosts}:
            return False
        if mutation:
            token = self.headers.get("X-PulseMorph-Token", "")
            if not secrets.compare_digest(token, self.server.session_token):
                return False
        return True

    def _headers(self, status, kind, size, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _reject_post(self, status, value):
        # Closing a Windows socket with an unread request body can reset the
        # connection before a tiny JSON error reaches the client. Drain only
        # bounded small bodies; large/unresponsive senders never hold a worker.
        previous_timeout = self.connection.gettimeout()
        try:
            remaining = int(self.headers.get("Content-Length", "0")) - getattr(self, "_body_read", 0)
            if 0 < remaining <= 16000:
                self.connection.settimeout(.25)
                self.rfile.read(remaining)
        except (ValueError, OSError, TimeoutError):
            pass
        finally:
            self.connection.settimeout(previous_timeout)
        self._json(status, value)

    def _file(self, path, *, media=False):
        path = Path(path)
        if not path.is_file():
            return self._json(404, {"error": "Dosya bulunamadı"})
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end, status = 0, size-1, 200
        headers = {"Accept-Ranges": "bytes"} if media else {}
        if media and self.headers.get("Range"):
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers["Range"])
            if not match:
                return self._json(416, {"error": "Geçersiz medya aralığı"})
            start = int(match[1])
            end = min(int(match[2]) if match[2] else end, end)
            if not 0 <= start <= end < size:
                return self._json(416, {"error": "Medya aralığı dosyanın dışında"})
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            status = 206
        self._headers(status, content_type, end-start+1, headers)
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end-start+1
                while remaining:
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        if not self._trusted_origin():
            return self._json(403, {"error": "Yalnız yerel uygulama erişebilir"})
        path = unquote(urlsplit(self.path).path)
        manager = self.server.manager
        try:
            if path == "/api/status":
                status = manager.status()
                music_status = manager.music.status() if hasattr(manager.music, "status") else {}
                return self._json(200, dict(status, music_warnings=music_status.get("warnings", []),
                                            session_token=self.server.session_token))
            if path == "/api/jobs":
                return self._json(200, {"jobs": manager.list_jobs()})
            if re.fullmatch(r"/api/jobs/[A-Za-z0-9_-]+", path):
                return self._json(200, manager.get(path.rsplit("/", 1)[1]))
            if path == "/api/memory":
                return self._json(200, {"records": manager.integrations.list_memory(), "integration": manager.integrations.status()["meterless"]})
            if re.fullmatch(r"/api/music/[a-z0-9_]+", path):
                info = manager.music.get(path.rsplit("/", 1)[1])
                return self._json(200, {k: v for k, v in info.items() if k != "path"})
            if path == "/":
                return self._file(Path(__file__).parent / "static/index.html")
            if path in {"/static/style.css", "/static/app.js"}:
                return self._file(Path(__file__).parent / path.lstrip("/"))
            music = re.fullmatch(r"/media/music/([a-z0-9_]+)\.wav", path)
            if music:
                return self._file(manager.music.get(music[1])["path"], media=True)
            media = re.fullmatch(r"/media/jobs/([A-Za-z0-9_-]+)/(before|after|comparison)\.mp4", path)
            if media:
                manager.get(media[1])
                directory = (manager.jobs_dir / media[1] / "media").resolve()
                if not directory.is_relative_to(manager.jobs_dir):
                    raise ValueError("Geçersiz medya yolu")
                return self._file(directory / (media[2] + ".mp4"), media=True)
            return self._json(404, {"error": "Sayfa bulunamadı"})
        except KeyError:
            return self._json(404, {"error": "Kayıt bulunamadı"})
        except (ValueError, OSError):
            return self._json(400, {"error": "İstek okunamadı veya kaynak hazır değil"})
        except Exception:
            return self._json(503, {"error": "Yerel bileşene erişilemedi; kurulum durumunu kontrol edin"})

    def do_POST(self):
        self._body_read = 0
        if not self._trusted_origin(mutation=True):
            return self._reject_post(403, {"error": "Yerel oturum doğrulanamadı; sayfayı yenileyin"})
        path = unquote(urlsplit(self.path).path)
        manager = self.server.manager
        try:
            length = int(self.headers.get("Content-Length", "0"))
            maximum = 32 * 1024 * 1024 if path == "/api/music/import" else 16000
            if not 0 <= length <= maximum:
                raise ValueError("İstek çok büyük")
            if path == "/api/music/import":
                if self.headers.get_content_type() not in {"audio/wav", "audio/x-wav"}:
                    raise ValueError("PCM16 WAV dosyası gerekli")
                if any(job["status"] not in {"complete", "cancelled", "failed"} for job in manager.list_jobs()):
                    raise JobBusy("Müzik eklemeden önce çalışan denemeyi tamamlayın veya durdurun")
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if set(query) - {"start_s", "label"} or any(len(values) != 1 for values in query.values()):
                    raise ValueError("Geçersiz ses kesiti alanları")
                self.connection.settimeout(30)
                try:
                    data = self.rfile.read(length)
                    self._body_read = len(data)
                except TimeoutError as exc:
                    raise ValueError("Dosya aktarımı zamanında tamamlanmadı") from exc
                if len(data) != length:
                    raise ValueError("Dosya aktarımı eksik")
                music = manager.music.import_wav(data, query.get("label", ["Yüklenen ses"])[0],
                                                 float(query.get("start_s", ["0"])[0]))
                return self._json(201, {"music": {k: v for k, v in music.items() if k != "path"},
                                        "music_catalog": manager.music.catalog()})
            if length and self.headers.get_content_type() != "application/json":
                raise ValueError("JSON isteği gerekli")
            raw = self.rfile.read(length) if length else b""
            self._body_read = len(raw)
            body = json.loads(raw) if length else {}
            if path == "/api/jobs":
                return self._json(202, {"job_id": manager.create(body)})
            export = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/export", path)
            if export:
                if body != {}:
                    raise ValueError("Hareket kaydı indirme isteği ek alan içeremez")
                if not self.server.export_lock.acquire(blocking=False):
                    raise JobBusy("Başka bir hareket kaydı hazırlanıyor; tamamlanınca yeniden deneyin")
                try:
                    from .exports import build_result_bundle
                    data = build_result_bundle(manager, export[1])
                    self._headers(200, "application/zip", len(data), {
                        "Content-Disposition": f'attachment; filename="PulseMorph-{export[1]}-result.zip"'})
                    try:
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                    return
                finally:
                    self.server.export_lock.release()
            cancel = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/cancel", path)
            if cancel:
                return self._json(200, manager.cancel(cancel[1]))
            return self._json(404, {"error": "İşlem bulunamadı"})
        except JobBusy as exc:
            return self._reject_post(409, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            return self._reject_post(400, {"error": str(exc)[:500]})
        except KeyError:
            return self._json(404, {"error": "İş bulunamadı"})
        except Exception:
            return self._json(500, {"error": "İş başlatılamadı; yerel kurulum durumunu kontrol edin"})


def main():
    parser = argparse.ArgumentParser(description="Private local agentic motion studio")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--trusted-local", action="store_true", help="Enable the authenticated Codex provider for this private local session")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--runtime-python", type=Path, help="Use a verified isolated Python 3.11 environment for physical jobs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    from .providers import CodexLocalProvider
    from .integrations import AppIntegrations
    from .runtime_config import make_runtime
    from .uploads import StudioMusic
    from .renderer import render_comparison
    artifacts = args.artifact_dir or root / "artifacts/runs/agentic/app"
    runtime = make_runtime(root, args.runtime_python)
    manager = JobManager(root, CodexLocalProvider(trusted_local=args.trusted_local), runtime, StudioMusic(root),
                         AppIntegrations(root, artifacts), render_comparison, artifacts)
    server = StudioServer(("127.0.0.1", args.port), manager)
    print(f"PulseMorph local studio: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.shutdown()


if __name__ == "__main__":
    main()
