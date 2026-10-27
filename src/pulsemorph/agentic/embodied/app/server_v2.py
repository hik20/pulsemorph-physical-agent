"""V2 launch composition: frozen HTTP/manager, versioned metadata-publish worker."""
import argparse
from pathlib import Path
from .server_v1 import Handler,Server
from .manager_v1 import BodyAgentManager

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8777)
    parser.add_argument('--trusted-local',action='store_true')
    args=parser.parse_args()
    if not args.trusted_local or not 1024<=args.port<=65535:
        raise ValueError('Explicit private local launch required')
    root=Path(__file__).resolve().parents[5]
    manager=BodyAgentManager(root,worker=Path(__file__).with_name('worker_v2.py'))
    server=Server(args.port,manager)
    print('PulseMorph Body Agent Lab v2: http://127.0.0.1:'+str(args.port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:
        for identity in list(manager.processes):manager.cancel(identity)
    finally:server.server_close()

if __name__=='__main__':main()
