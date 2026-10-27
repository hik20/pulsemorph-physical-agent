"""V2 composition: frozen HTTP/manager with bounded encoders and render diagnostics."""
import argparse
from pathlib import Path
from .server_v1 import Facade,Handler,Server
from .assets_v1 import ROOT_CACHE,PUBLIC_PYTHON
from .manager_v1 import DanceManager
from .music_v1 import DanceMusic
from .render_v2 import render_dance


def logged_render(*args,**kwargs):
    # render_v2 writes exclusive failure.json with actual partial counters before
    # propagating; the frozen manager keeps its safe generic public error.
    return render_dance(*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8774)
    parser.add_argument('--trusted-local',action='store_true')
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args()
    if not args.trusted_local or not 1024<=args.port<=65535:
        raise ValueError('Explicit private local launch required')
    root=Path(__file__).resolve().parents[4]
    from ..app.phrase.music import PhraseMusic
    from ..app.providers import CodexLocalProvider
    from ..app.integrations import AppIntegrations
    from ..expression.runtime_v1 import ExpressionRuntime
    from ..expression.runner_v3 import ExpressionManager,JOB_ROOT as LEGACY_JOBS
    music=PhraseMusic(root);provider=CodexLocalProvider(trusted_local=True)
    legacy=ExpressionManager(root,ExpressionRuntime(root,PUBLIC_PYTHON,LEGACY_JOBS),provider,music)
    dance=None
    if not args.preview_only:
        from .runtime_v1 import DanceRuntime
        dance=DanceManager(root,DanceRuntime(root),provider,DanceMusic(music),jobs_dir=ROOT_CACHE/'jobs',renderer=logged_render,
            integrations=AppIntegrations(root,root/'artifacts/runs/agentic/dance_studio_v1'))
    server=Server(args.port,Facade(root,dance,legacy,music,preview_only=args.preview_only))
    print('PulseMorph Dance Studio v2: http://127.0.0.1:'+str(args.port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:
        if dance:
            for identity in list(dance._cancel):dance.cancel(identity)
    finally:server.server_close()

if __name__=='__main__':main()
