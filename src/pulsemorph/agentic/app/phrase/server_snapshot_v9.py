"""Imported-observation UI and an explicit, separate full-record recheck."""
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .server_response_v8 import ResponseHandler, _unique_object, _finite_json
from .server import PhraseStudioServer


class SnapshotHandler(ResponseHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path not in {'/phrase', '/static/phrase_v9.js'}:
            return super().do_GET()
        if not self._trusted_origin():
            return self._json(403, {'error': 'Yalnız yerel uygulama erişebilir'})
        static = Path(__file__).parents[1] / 'static'
        return self._file(static / ('phrase_v9.html' if path == '/phrase' else 'phrase_v9.js'))

    def do_POST(self):
        parts = urlsplit(self.path)
        if unquote(parts.path) != '/api/phrase/response-evidence/recheck':
            return super().do_POST()
        self._body_read = 0
        if not self._trusted_origin(mutation=True):
            return self._reject_post(403, {'error': 'Yerel oturum doğrulanamadı; sayfayı yenileyin'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if parts.query or not 0 < length <= 1024 or self.headers.get_content_type() != 'application/json':
                raise ValueError('exact empty JSON object required')
            self.connection.settimeout(5)
            raw = self.rfile.read(length)
            self._body_read = len(raw)
            if len(raw) != length or json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_finite_json) != {}:
                raise ValueError('exact empty JSON object required')
        except (ValueError, TypeError, OSError):
            return self._reject_post(400, {'error': 'Tam kayıt denetimi yalnız boş JSON nesnesi kabul eder'})
        try:
            report = self.server.phrase_manager.recheck_response_evidence()
        except Exception:
            return self._json(503, {'error': 'Tam kayıtlar bu denetimde doğrulanamadı; geçmiş gözlemler değiştirilmedi'})
        return self._json(200, report)


def make_snapshot_server(address, manager, phrase_manager):
    server = PhraseStudioServer(address, manager, phrase_manager)
    server.RequestHandlerClass = SnapshotHandler
    return server
