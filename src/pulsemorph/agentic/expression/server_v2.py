"""Expression Studio with request-bound goal enums; frozen v1 stays intact."""
import argparse
from pathlib import Path
from ..app.providers import CodexLocalProvider
from ..app.phrase.assets import PUBLIC_PYTHON
from ..app.phrase.music import PhraseMusic
from .runtime_v1 import ExpressionRuntime
from .runner_v2 import ExpressionManager, JOB_ROOT
from .server_v1 import Server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8774)
    parser.add_argument('--trusted-local', action='store_true')
    args = parser.parse_args()
    if not args.trusted_local or not 1024 <= args.port <= 65535:
        raise ValueError('Explicit private local launch required')
    root = Path(__file__).resolve().parents[4]
    runtime = ExpressionRuntime(root, PUBLIC_PYTHON, JOB_ROOT)
    manager = ExpressionManager(root, runtime, CodexLocalProvider(trusted_local=True), PhraseMusic(root))
    server = Server(args.port, manager)
    print('PulseMorph Expression Studio v2: http://127.0.0.1:' + str(args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        for identity in list(manager._cancel):
            manager.cancel(identity)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
