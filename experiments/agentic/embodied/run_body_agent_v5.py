"""Prepare by default; --execute needs an explicitly approved plan digest."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.app.firstuse_v5 import prepare,execute,write_new
def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--request');p.add_argument('--output');p.add_argument('--plan',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--approved-plan-sha256');a=p.parse_args()
    if a.execute:r=execute(a.plan,approved_plan_sha256=a.approved_plan_sha256 or '')
    else:
        if not all((a.config,a.request,a.output)):p.error('Preparation requiresconfig/request/output')
        r=prepare(ROOT,a.config,json.loads(Path(a.request).read_bytes()),a.output);write_new(a.plan,r)
    print(json.dumps({'status':r.get('status','prepared'),'executed':a.execute,'task_verified_actual':r.get('task_verified_actual',False)}))
    if a.execute and r.get('status')=='failed':raise SystemExit(1)
if __name__=='__main__':main()
