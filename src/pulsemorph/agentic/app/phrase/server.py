"""Two private studios on one loopback server with one active HTTP job.

The original four-second entry point and handler remain unchanged. This opt-in
entry adds bounded eight-second jobs and keeps their artifacts separate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import parse_qs, unquote, urlsplit

from ..server import Handler, StudioServer
from ..orchestrator import JobBusy, TERMINAL
from .catalog import tool_catalog


class PhraseStudioServer(StudioServer):
    def __init__(self, address, manager, phrase_manager):
        super().__init__(address, manager)
        self.phrase_manager = phrase_manager
        self.job_creation_lock = threading.Lock()
        self.RequestHandlerClass = PhraseHandler

    def active_job(self):
        return any(job.get('status') not in TERMINAL
                   for manager in (self.manager, self.phrase_manager)
                   for job in manager.list_jobs())

    def legacy_disk_status(self):
        required = 256 * 1024 * 1024
        try:
            free = shutil.disk_usage(self.manager.jobs_dir.parent).free
        except OSError:
            free = None
        return {'available': free is not None and free >= required,
                'free_bytes': free, 'required_bytes': required}


class PhraseHandler(Handler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        relevant = path in {'/studio', '/phrase', '/static/phrase.js', '/static/phrase.css', '/static/studio.js', '/api/status'} or path.startswith(('/api/phrase/', '/media/phrase/'))
        if not relevant:
            return super().do_GET()
        if not self._trusted_origin():
            return self._json(403, {'error': 'Yalnız yerel uygulama erişebilir'})
        manager = self.server.phrase_manager
        try:
            if path == '/api/status':
                legacy = self.server.manager
                music_status = legacy.music.status() if hasattr(legacy.music, 'status') else {}
                return self._json(200, dict(legacy.status(), session_token=self.server.session_token,
                                           music_warnings=music_status.get('warnings', []),
                                           runtime_disk=self.server.legacy_disk_status()))
            if path == '/api/phrase/status':
                return self._json(200, dict(manager.status(), session_token=self.server.session_token,
                                           tool_catalog=tool_catalog(), active_job_across_studios=self.server.active_job(),
                                           legacy_disk=self.server.legacy_disk_status()))
            if path == '/api/phrase/jobs':
                return self._json(200, {'jobs': manager.list_jobs()})
            match = re.fullmatch(r'/api/phrase/jobs/([A-Za-z0-9_-]+)', path)
            if match:
                return self._json(200, manager.get(match[1]))
            match = re.fullmatch(r'/api/phrase/music/([a-z0-9_]+)', path)
            if match:
                return self._json(200, {k: v for k, v in manager.music.get(match[1]).items() if k != 'path'})
            static = Path(__file__).parents[1] / 'static'
            if path in {'/studio', '/phrase'}:
                return self._file(static / ('studio.html' if path == '/studio' else 'phrase.html'))
            if path in {'/static/phrase.js', '/static/phrase.css', '/static/studio.js'}:
                return self._file(static / path.rsplit('/', 1)[1])
            match = re.fullmatch(r'/media/phrase/music/([a-z0-9_]+)\.wav', path)
            if match:
                return self._file(manager.music.get(match[1])['path'], media=True)
            match = re.fullmatch(r'/media/phrase/jobs/([A-Za-z0-9_-]+)/(before|after|comparison)\.mp4', path)
            if match:
                manager.get(match[1])
                directory = (manager.jobs_dir / match[1] / 'media').resolve()
                if not directory.is_relative_to(manager.jobs_dir.resolve()):
                    raise ValueError('Geçersiz medya yolu')
                return self._file(directory / (match[2] + '.mp4'), media=True)
            return self._json(404, {'error': 'Sayfa bulunamadı'})
        except KeyError:
            return self._json(404, {'error': 'Kayıt bulunamadı'})
        except (ValueError, OSError):
            return self._json(400, {'error': 'Kaynak hazır değil veya istek okunamadı'})
        except Exception:
            return self._json(503, {'error': 'Yerel kompozisyon bileşeni hazır değil'})

    def do_POST(self):
        path = unquote(urlsplit(self.path).path)
        if not path.startswith('/api/phrase/'):
            if path in {'/api/jobs', '/api/music/import'}:
                self._body_read = 0
                if not self._trusted_origin(mutation=True):
                    return self._reject_post(403, {'error': 'Yerel oturum doğrulanamadı; sayfayı yenileyin'})
                with self.server.job_creation_lock:
                    if any(j.get('status') not in TERMINAL for j in self.server.phrase_manager.list_jobs()):
                        return self._reject_post(409, {'error': 'Sekiz saniyelik çalışma sürüyor; önce tamamlayın veya durdurun'})
                    if path == '/api/jobs' and not self.server.legacy_disk_status()['available']:
                        return self._reject_post(507, {'error': 'Dört saniyelik çalışma deposunda en az 256 MiB boş alan gerekli. Model çağrısı başlatılmadı. Sekiz saniyelik stüdyo ayrı depoyu kullanır.'})
                    return super().do_POST()
            return super().do_POST()
        self._body_read = 0
        if not self._trusted_origin(mutation=True):
            return self._reject_post(403, {'error': 'Yerel oturum doğrulanamadı; sayfayı yenileyin'})
        manager = self.server.phrase_manager
        try:
            length = int(self.headers.get('Content-Length', '0'))
            maximum = 32 * 1024 * 1024 if path == '/api/phrase/music/import' else 16000
            if not 0 <= length <= maximum:
                raise ValueError('İstek çok büyük')
            if path == '/api/phrase/music/import':
                if self.headers.get_content_type() not in {'audio/wav', 'audio/x-wav'}:
                    raise ValueError('PCM16 WAV dosyası gerekli')
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if set(query) - {'start_s', 'label'} or any(len(v) != 1 for v in query.values()):
                    raise ValueError('Geçersiz ses kesiti alanları')
                with self.server.job_creation_lock:
                    if self.server.active_job():
                        raise JobBusy('Müzik eklemeden önce çalışan denemeyi tamamlayın veya durdurun')
                    self.connection.settimeout(30)
                    data = self.rfile.read(length)
                    self._body_read = len(data)
                    if len(data) != length:
                        raise ValueError('Dosya aktarımı eksik')
                    record = manager.music.import_wav(data, query.get('label', ['Yüklenen ses'])[0], float(query.get('start_s', ['0'])[0]))
                return self._json(201, {'music': {k: v for k, v in record.items() if k != 'path'}, 'music_catalog': manager.music.catalog()})
            if length and self.headers.get_content_type() != 'application/json':
                raise ValueError('JSON isteği gerekli')
            raw = self.rfile.read(length) if length else b''
            self._body_read = len(raw)
            if len(raw) != length:
                raise ValueError('İstek eksik')
            body = json.loads(raw) if length else {}
            if path == '/api/phrase/jobs':
                with self.server.job_creation_lock:
                    if self.server.active_job():
                        raise JobBusy('Bir çalışma zaten sürüyor; tamamlanmasını bekleyin veya durdurun')
                    return self._json(202, {'job_id': manager.create(body)})
            match = re.fullmatch(r'/api/phrase/jobs/([A-Za-z0-9_-]+)/cancel', path)
            if match:
                if body != {}:
                    raise ValueError('Durdurma isteği ek alan içeremez')
                return self._json(200, manager.cancel(match[1]))
            return self._json(404, {'error': 'İşlem bulunamadı'})
        except JobBusy as exc:
            return self._reject_post(409, {'error': str(exc)})
        except (ValueError, TypeError, TimeoutError):
            return self._reject_post(400, {'error': 'İstek veya sekiz saniyelik ses kesiti doğrulanamadı'})
        except KeyError:
            return self._json(404, {'error': 'İş bulunamadı'})
        except Exception:
            return self._json(500, {'error': 'Kompozisyon başlatılamadı; yerel kurulum durumunu kontrol edin'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--trusted-local', action='store_true')
    parser.add_argument('--runtime-python', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    from ..providers import CodexLocalProvider
    from ..integrations import AppIntegrations
    from ..orchestrator import JobManager
    from ..runtime_config import make_runtime
    from ..uploads import StudioMusic
    from ..renderer import render_comparison
    from .runtime import PhraseRuntime
    from .manager import PhraseManager
    from .music import PhraseMusic
    from .renderer import render_phrase_comparison
    artifact_root = root / 'artifacts/runs/agentic/app'
    phrase_root = Path.home() / '.cache/pulsemorph-agentic/app_phrase_v1'
    provider = CodexLocalProvider(trusted_local=args.trusted_local)
    integrations = AppIntegrations(root, artifact_root)
    old_runtime = make_runtime(root, args.runtime_python)
    old = JobManager(root, provider, old_runtime, StudioMusic(root), integrations, render_comparison, artifact_root)
    phrase = PhraseManager(root, provider, PhraseRuntime(root, old_runtime.python, phrase_root),
                           PhraseMusic(root, phrase_root), integrations, render_phrase_comparison, phrase_root)
    server = PhraseStudioServer(('127.0.0.1', args.port), old, phrase)
    print(f'PulseMorph studio: http://127.0.0.1:{server.server_port}/studio', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        phrase.shutdown()
        old.shutdown()


if __name__ == '__main__':
    main()
