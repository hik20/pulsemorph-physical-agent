"""Run independent committed/numerical streams in isolated processes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import subprocess
import sys
from musical_tasks import ROOT, read, write_new_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();out=a.pilot.resolve()
    protocol=read(out/'protocol.json')
    streams=[(t['task_id'],m) for t in protocol['tasks'] for m in ['surrogate_search','llm_feedback_off']]
    def stream(item):
        task,method=item
        for index in range(8):
            identity=f'{task}__{method}__{index}'
            logfile=out/'logs'/f'{identity}.log';logfile.parent.mkdir(exist_ok=True)
            command=[sys.executable,str(ROOT/'experiments/agentic/generalization_driver.py'),'--pilot',str(out),
                '--action','run','--task',task,'--method',method,'--index',str(index)]
            with logfile.open('x',encoding='utf8') as f:
                r=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            write_new_json(out/'launches'/f'{identity}.json',dict(command=command,returncode=r.returncode,log=str(logfile)))
            if r.returncode: raise RuntimeError(identity+' failed; inspect '+str(logfile))
            print(json.dumps({'completed':identity}),flush=True)
        return item
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(stream,streams): pass

if __name__=='__main__': main()
