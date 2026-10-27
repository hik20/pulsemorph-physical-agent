"""Execute already-generated feedback proposals in separate local processes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import subprocess
import sys
from musical_tasks import ROOT, read, write_new_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--tasks',nargs='*',default=['wave_a','wave_b','high5_a','high5_b'])
    a=p.parse_args();out=a.pilot.resolve()
    def execute(task):
        identity=f'{task}__llm_feedback__{a.index}'
        log=out/'logs'/f'{identity}.log';log.parent.mkdir(exist_ok=True)
        command=[sys.executable,str(ROOT/'experiments/agentic/generalization_driver.py'),'--pilot',str(out),
            '--action','run','--task',task,'--method','llm_feedback','--index',str(a.index)]
        with log.open('x',encoding='utf8') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        write_new_json(out/'launches'/f'{identity}.json',dict(command=command,returncode=result.returncode,log=str(log)))
        if result.returncode: raise RuntimeError(identity+' failed; inspect '+str(log))
        row=read(out/'evaluations'/f'{identity}.json')
        print(json.dumps({k:row.get(k) for k in ['candidate_id','features','physical_gate_passed','intent_passed','utility']}),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(execute,a.tasks))

if __name__=='__main__': main()
