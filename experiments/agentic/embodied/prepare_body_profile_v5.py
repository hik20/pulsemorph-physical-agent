"""Explicit public reference-only preparation, never starts physics/inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from pulsemorph.agentic.embodied.app.config_v5 import load_config,probe_environment
from pulsemorph.agentic.embodied.app.profile_v5 import build_profile,materialize_declared_newlines
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--build',action='store_true');a=p.parse_args();c=load_config(a.config)
    if a.build:print(json.dumps(build_profile(c)))
    else:print(json.dumps({'environment':probe_environment(c['runtime_python']),'required_materialization':materialize_declared_newlines(c),'new_NN':0,'new_FK':0,'new_steps':0}))
if __name__=='__main__':main()
