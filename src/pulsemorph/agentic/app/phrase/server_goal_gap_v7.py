"""Versioned goal-gap UI; execution and descriptor routes remain unchanged."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .server_descriptor_v6 import DescriptorHandler
from .server import PhraseStudioServer


class GoalGapHandler(DescriptorHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path not in {'/phrase', '/static/phrase_v7.js'}:
            return super().do_GET()
        if not self._trusted_origin():
            return self._json(403, {'error': 'Yalnız yerel uygulama erişebilir'})
        static = Path(__file__).parents[1] / 'static'
        return self._file(static / ('phrase_v7.html' if path == '/phrase' else 'phrase_v7.js'))


def make_goal_gap_server(address, manager, phrase_manager):
    server = PhraseStudioServer(address, manager, phrase_manager)
    server.RequestHandlerClass = GoalGapHandler
    return server
