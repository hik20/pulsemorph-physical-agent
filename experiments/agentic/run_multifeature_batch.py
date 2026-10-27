"""Run a fixed requested set of independent proposal slots, retaining console logs."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

from multifeature_driver import run, read, compact


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--method',choices=['jacobian_search','llm_feedback','llm_feedback_off'],required=True)
    p.add_argument('--indices',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--tasks',nargs='+');p.add_argument('--log-label',required=True)
    a=p.parse_args();out=a.pilot.resolve();protocol=read(out/'protocol.json')
    if Path(a.log_label).name!=a.log_label:raise ValueError('single log label required')
    jobs=a.tasks or [t['task_id'] for t in protocol['tasks']]
    log=out/'logs'/f'{a.log_label}.log';log.parent.mkdir(exist_ok=True)
    with log.open('x',encoding='utf-8') as stream:
        for task in jobs:
            for index in a.indices:
                try:
                    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                        result=run(out,task,a.method,index)
                except Exception as error:
                    print(json.dumps({'task':task,'method':a.method,'index':index,'error':repr(error)}),flush=True)
                    raise
                stream.flush()
                print(json.dumps(compact(result),allow_nan=False),flush=True)


if __name__=='__main__':main()
