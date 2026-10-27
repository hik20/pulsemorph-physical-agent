"""Record the orchestrator-observed model turn; no provider billing inference."""
from pathlib import Path
import argparse
from generalization_driver import identifier, observation_path, response_path, commit_model, stamp
from musical_tasks import read, sha256, write_new_json


def record(out, task, method, index, agent):
    out=Path(out).resolve()
    identity=identifier(task,method,index)
    obs=observation_path(out,task,method,index);response=response_path(out,task,method,index)
    raw=read(response)
    if Path(raw['only_observation_path']).resolve()!=obs.resolve():
        raise ValueError('response observation mismatch')
    receipt=dict(candidate_id=identity,task_id=task,method=method,trial_index=index,
        requested_model='gpt-6.1-sol',fork_turns='none',agent_name=agent,
        observation_sha256=sha256(obs),response_sha256=sha256(response),completed_at_utc=stamp(),
        receipt_scope='orchestrator-observed collaboration turn; not provider billing or token receipt')
    write_new_json(out/'model_calls'/f'{identity}.json',receipt)
    commit_model(out,task,method,index)
    return identity


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--method',required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--agent',required=True)
    a=p.parse_args();print(record(a.pilot,a.task,a.method,a.index,a.agent))
