"""Loopback-only Expression Studio. A POST authorizes one bounded private job."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit

from ..app.providers import CodexLocalProvider
from ..app.phrase.assets import PUBLIC_PYTHON
from ..app.phrase.music import PhraseMusic
from .runtime_v1 import ExpressionRuntime
from .runner_v1 import CACHE, JOB_ROOT, ExpressionManager

WEB = Path(__file__).with_name('web')


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, manager):
        super().__init__(('127.0.0.1', port), Handler)
        self.manager, self.csrf = manager, secrets.token_urlsafe(32)


class Handler(BaseHTTPRequestHandler):
    server_version = 'PulseMorphExpression/1'

    def log_message(self, *_args):
        pass  # Neither request bodies nor session values enter access logs.

    def _guard(self, mutation=False):
        allowed = {'127.0.0.1:' + str(self.server.server_port),
                   'localhost:' + str(self.server.server_port)}
        if self.headers.get('Host') not in allowed:
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://' + host for host in allowed}:
            return False
        return not mutation or secrets.compare_digest(
            self.headers.get('X-PulseMorph-Token', ''), self.server.csrf)

    def _bytes(self, status, data, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, value):
        self._bytes(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf8'),
                    'application/json; charset=utf-8')

    def _file(self, path, content_type, attachment=False):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            return self._json(404, {'error': 'FILE_UNAVAILABLE'})
        size, start, end, status = path.stat().st_size, 0, path.stat().st_size - 1, 200
        header = self.headers.get('Range')
        if header:
            match = re.fullmatch(r'bytes=(\d+)-(\d*)', header)
            if not match:
                return self._json(416, {'error': 'INVALID_RANGE'})
            start = int(match[1]); end = min(size - 1, int(match[2])) if match[2] else size - 1
            if start >= size or end < start:
                return self._json(416, {'error': 'INVALID_RANGE'})
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if status == 206:
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
        if attachment:
            self.send_header('Content-Disposition', 'attachment; filename="pulsemorph-reference.zip"')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                data = stream.read(min(65536, remaining))
                if not data:
                    break
                self.wfile.write(data); remaining -= len(data)

    def do_GET(self):
        if not self._guard():
            return self._json(403, {'error': 'LOOPBACK_ORIGIN_REQUIRED'})
        path = urlsplit(self.path).path
        manager = self.server.manager
        try:
            if path in ('/', '/index.html', '/app.js', '/style.css'):
                name = 'index.html' if path == '/' else path[1:]
                kind = {'index.html': 'text/html; charset=utf-8', 'app.js': 'application/javascript; charset=utf-8',
                        'style.css': 'text/css; charset=utf-8'}[name]
                return self._file(WEB / name, kind)
            if path == '/favicon.ico':
                return self._bytes(204, b'', 'image/x-icon')
            if path == '/api/status':
                return self._json(200, {**manager.status(), 'csrf_token': self.server.csrf})
            if path == '/api/jobs':
                return self._json(200, {'jobs': manager.list_jobs()})
            match = re.fullmatch(r'/api/jobs/([A-Za-z0-9_-]{1,80})(/export)?', path)
            if match:
                if match[2]:
                    return self._file(manager.media_path(match[1], 'export'), 'application/zip', True)
                return self._json(200, manager.get(match[1]))
            match = re.fullmatch(r'/media/jobs/([A-Za-z0-9_-]{1,80})/comparison.mp4', path)
            if match:
                return self._file(manager.media_path(match[1], 'video'), 'video/mp4')
            if path == '/media/music/phrase_original_pulse_v1.wav':
                return self._file(manager.music.get('phrase_original_pulse_v1')['path'], 'audio/wav')
            return self._json(404, {'error': 'NOT_FOUND'})
        except (ValueError, KeyError, OSError):
            return self._json(409, {'error': 'LOCAL_RECORD_UNAVAILABLE_OR_CHANGED'})
        except Exception:
            return self._json(503, {'error': 'LOCAL_RECORD_UNAVAILABLE_OR_CHANGED'})

    def do_POST(self):
        if not self._guard(True):
            return self._json(403, {'error': 'LOCAL_SESSION_REQUIRED'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self._json(415, {'error': 'JSON_REQUIRED'})
        try:
            self.connection.settimeout(10)
            length = int(self.headers.get('Content-Length', '0'))
            if not 2 <= length <= 16384:
                raise ValueError('Bounded JSON required')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete JSON body')
            def pairs(items):
                result = {}
                for key, item in items:
                    if key in result:
                        raise ValueError('Duplicate JSON member')
                    result[key] = item
                return result
            def constant(_value):
                raise ValueError('Nonfinite JSON')
            value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
            path = urlsplit(self.path).path
            if path == '/api/jobs':
                return self._json(202, {'id': self.server.manager.create(value)})
            match = re.fullmatch(r'/api/jobs/([A-Za-z0-9_-]{1,80})/cancel', path)
            if match and value == {}:
                self.server.manager.cancel(match[1])
                return self._json(202, {'cancel_requested': True})
            return self._json(404, {'error': 'NOT_FOUND'})
        except (ValueError, KeyError, TypeError, UnicodeError):
            return self._json(400, {'error': 'INVALID_OR_UNAVAILABLE_JOB_REQUEST'})
        except Exception:
            return self._json(503, {'error': 'LOCAL_JOB_UNAVAILABLE'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8774)
    parser.add_argument('--trusted-local', action='store_true')
    args = parser.parse_args()
    if not args.trusted_local or not 1024 <= args.port <= 65535:
        raise ValueError('Explicit private local launch required')
    root = Path(__file__).resolve().parents[4]
    runtime = ExpressionRuntime(root, PUBLIC_PYTHON, JOB_ROOT)
    manager = ExpressionManager(root, runtime, CodexLocalProvider(trusted_local=True),
                                PhraseMusic(root))
    server = Server(args.port, manager)
    print('PulseMorph Expression Studio: http://127.0.0.1:' + str(args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        for identity in list(manager._cancel):
            manager.cancel(identity)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
