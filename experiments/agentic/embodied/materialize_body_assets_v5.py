"""Materialize already acquired pinned public assets; never downloads/installs."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from pulsemorph.agentic.embodied.app.acquisition_v5 import materialize
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--identity',required=True);a=p.parse_args()
    r=materialize(a.source,a.destination,a.identity)
    print(json.dumps({'files':len(r['files']),'new_downloads':0,'new_NN':0,'new_FK':0,'new_steps':0}))
if __name__=='__main__':main()
