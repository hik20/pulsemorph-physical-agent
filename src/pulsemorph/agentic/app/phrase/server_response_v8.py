"""Read-only measured-response catalog/query and versioned studio UI."""
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .server_goal_gap_v7 import GoalGapHandler
from .server import PhraseStudioServer


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON member')
        value[key] = item
    return value


def _finite_json(token):
    raise ValueError('nonfinite JSON number')


class ResponseHandler(GoalGapHandler):
    def do_GET(self):
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        if path not in {'/phrase', '/static/phrase_v8.js', '/api/phrase/response-evidence'}:
            return super().do_GET()
        if not self._trusted_origin():
            return self._json(403, {'error': 'Yalnız yerel uygulama erişebilir'})
        if path == '/api/phrase/response-evidence':
            if parts.query:
                return self._json(400, {'error': 'Koşullu sorgu için JSON sorgu yolunu kullanın'})
            try:
                return self._json(200, self.server.phrase_manager.describe_response_evidence())
            except Exception:
                return self._json(503, {'error': 'Ölçülmüş motor kanıtı doğrulanamadı'})
        static = Path(__file__).parents[1] / 'static'
        return self._file(static / ('phrase_v8.html' if path == '/phrase' else 'phrase_v8.js'))

    def do_POST(self):
        parts = urlsplit(self.path)
        if unquote(parts.path) != '/api/phrase/response-evidence/query':
            return super().do_POST()
        self._body_read = 0
        # Read-only query, with the same session/Origin guard as other POSTs.
        if not self._trusted_origin(mutation=True):
            return self._reject_post(403, {'error': 'Yerel oturum doğrulanamadı; sayfayı yenileyin'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if parts.query or not 0 < length <= 4096 or self.headers.get_content_type() != 'application/json':
                raise ValueError('bounded JSON object required')
            self.connection.settimeout(5)
            raw = self.rfile.read(length)
            self._body_read = len(raw)
            if len(raw) != length:
                raise ValueError('incomplete query')
            query = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_finite_json)
            if type(query) is not dict:
                raise ValueError('query object required')
        except (ValueError, TypeError, TimeoutError):
            return self._reject_post(400, {'error': 'Motor kanıtı sorgusu doğrulanamadı'})
        try:
            answer = self.server.phrase_manager.query_response_evidence(query)
        except ValueError:
            return self._json(400, {'error': 'Motor kanıtı sorgu alanları doğrulanamadı'})
        except Exception:
            return self._json(503, {'error': 'Ölçülmüş motor kanıtı doğrulanamadı'})
        return self._json(200, answer)


def make_response_server(address, manager, phrase_manager):
    server = PhraseStudioServer(address, manager, phrase_manager)
    server.RequestHandlerClass = ResponseHandler
    return server
