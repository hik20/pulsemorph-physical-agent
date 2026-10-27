"""Versioned goal-aware UI and result dispatch; existing handlers stay frozen."""
from pathlib import Path
import json
import re
from urllib.parse import unquote, urlsplit
from .server import PhraseHandler, PhraseStudioServer


class GoalHandler(PhraseHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path not in {'/phrase', '/static/phrase_v4.js'}:
            return super().do_GET()
        if not self._trusted_origin():
            return self._json(403, {'error': 'Yalnız yerel uygulama erişebilir'})
        static = Path(__file__).parents[1] / 'static'
        return self._file(static / ('phrase_v4.html' if path == '/phrase' else 'phrase_v4.js'))

    def do_POST(self):
        path = unquote(urlsplit(self.path).path)
        match = re.fullmatch(r'/api/phrase/jobs/([A-Za-z0-9_-]+)/export', path)
        if not match:
            return super().do_POST()
        self._body_read = 0
        if not self._trusted_origin(mutation=True):
            return self._reject_post(403, {'error': 'Yerel oturum doğrulanamadı; sayfayı yenileyin'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024 or self.headers.get_content_type() != 'application/json':
                raise ValueError('JSON empty object required')
            self.connection.settimeout(5)
            data = self.rfile.read(length)
            self._body_read = len(data)
            if len(data) != length or json.loads(data) != {}:
                raise ValueError('Exact empty object required')
        except (ValueError, TypeError, TimeoutError):
            return self._reject_post(400, {'error': 'İndirme isteği yalnız boş JSON nesnesi içermeli'})
        if not self.server.export_lock.acquire(blocking=False):
            return self._json(409, {'error': 'Başka bir hareket kaydı hazırlanıyor; tamamlanınca yeniden deneyin'})
        try:
            from .exports_goals_v4 import build_phrase_result_bundle
            archive = build_phrase_result_bundle(self.server.phrase_manager, match[1])
            self._headers(200, 'application/zip', len(archive), {
                'Content-Disposition': f'attachment; filename="PulseMorph-8s-{match[1]}-result.zip"'})
            try:
                self.wfile.write(archive)
            except OSError:
                pass
        except KeyError:
            return self._json(404, {'error': 'Çalışma bulunamadı'})
        except ValueError:
            return self._json(400, {'error': 'Tamamlanmış seçimin dosyaları doğrulanamadı; kayıtlar korundu'})
        except Exception:
            return self._json(503, {'error': 'Hareket kaydı hazırlanamadı; yerel dosyaları kontrol edin'})
        finally:
            self.server.export_lock.release()


def make_goal_server(address, manager, phrase_manager):
    server = PhraseStudioServer(address, manager, phrase_manager)
    server.RequestHandlerClass = GoalHandler
    return server
