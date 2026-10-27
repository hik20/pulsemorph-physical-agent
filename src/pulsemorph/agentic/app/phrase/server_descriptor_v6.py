"""Read-only pinned source descriptors and versioned UI, existing POST guards."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .server_advice_v5 import AdviceHandler
from .server import PhraseStudioServer


class DescriptorHandler(AdviceHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == '/api/phrase/source-descriptor':
            if not self._trusted_origin():
                return self._json(403, {'error': 'Yalnız yerel uygulama erişebilir'})
            try:
                return self._json(200, self.server.phrase_manager.describe_sources())
            except Exception:
                return self._json(503, {'error': 'Kaynak referans özeti şu anda okunamadı'})
        if path not in {'/phrase', '/static/phrase_v6.js'}:
            return super().do_GET()
        if not self._trusted_origin():
            return self._json(403, {'error': 'Yalnız yerel uygulama erişebilir'})
        static = Path(__file__).parents[1] / 'static'
        return self._file(static / ('phrase_v6.html' if path == '/phrase' else 'phrase_v6.js'))


def make_descriptor_server(address, manager, phrase_manager):
    server = PhraseStudioServer(address, manager, phrase_manager)
    server.RequestHandlerClass = DescriptorHandler
    return server
