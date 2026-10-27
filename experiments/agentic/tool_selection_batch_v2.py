"""Serial execution helper; never generates LLM responses or retries a slot."""
from pathlib import Path
import argparse
import json
from tool_selection_driver_v2 import context, record_evaluation, validate_calibrations, run, summarize
from execute_tool_selection import execute_trial
from musical_tasks import read, write_new_json


def batch(out, phase, index=None):
    out=Path(out).resolve(); protocol=read(out/'protocol.json')
    if phase=='calibrate':
        for source in ('wave','high5'):
            task=next(t['task_id'] for t in protocol['tasks'] if t['source_id']==source)
            for prefix in ('calibration_','calibration_local_'):
                candidate=prefix+source
                result=execute_trial(out/'proposals'/f'{candidate}.json',out/'trials'/candidate)
                if result['status']!='success':
                    raise RuntimeError('calibration runtime failed; no retry: '+candidate)
                record_evaluation(out,candidate,task,calibration=True)
        write_new_json(out/'calibration_parity.json',validate_calibrations(out))
    elif phase=='baselines':
        if type(index) is not int or not 0<=index<8: raise ValueError('index0..7 required')
        for task in protocol['tasks']:
            for method in ('surrogate_search','llm_feedback_off'):
                run(out,task['task_id'],method,index)
    elif phase=='summarize':
        result=summarize(out)
        print(json.dumps({'physical_trials':result['physical_trials'],'control_steps':result['control_steps'],
                          'physics_steps':result['physics_steps']}),flush=True)
    else: raise ValueError('unknown phase')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--phase',choices=['calibrate','baselines','summarize'],required=True)
    parser.add_argument('--index',type=int)
    args=parser.parse_args();batch(args.pilot,args.phase,args.index)
