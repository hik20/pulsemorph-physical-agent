"""Private studio with verified, strictly conditioned measured-response evidence."""
import argparse
from pathlib import Path
from .entrypoint_v2 import prepare_parent_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--trusted-local', action='store_true')
    parser.add_argument('--runtime-python', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    prepare_parent_imports(root)
    from ..providers import CodexLocalProvider
    from ..integrations import AppIntegrations
    from ..orchestrator import JobManager
    from ..runtime_config import make_runtime
    from ..uploads import StudioMusic
    from ..renderer import render_comparison
    from .runtime import PhraseRuntime
    from .manager_response_v8 import PhraseResponseManager
    from .music import PhraseMusic
    from .renderer import render_phrase_comparison
    from .server_response_v8 import make_response_server
    artifacts = root / 'artifacts/runs/agentic/app'
    phrase_root = Path.home() / '.cache/pulsemorph-agentic/app_phrase_v1'
    provider = CodexLocalProvider(trusted_local=args.trusted_local)
    integrations = AppIntegrations(root, artifacts)
    runtime = make_runtime(root, args.runtime_python)
    manager = JobManager(root, provider, runtime, StudioMusic(root), integrations, render_comparison, artifacts)
    phrase = PhraseResponseManager(root, provider, PhraseRuntime(root, runtime.python, phrase_root),
                           PhraseMusic(root, phrase_root), integrations, render_phrase_comparison, phrase_root)
    server = make_response_server(('127.0.0.1', args.port), manager, phrase)
    print(f'PulseMorph studio: http://127.0.0.1:{server.server_port}/studio', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        phrase.shutdown()
        manager.shutdown()


if __name__ == '__main__':
    main()
