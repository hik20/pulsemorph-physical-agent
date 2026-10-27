"""Persist proposer dispatch instructions and observed completion receipts.

Does not call a model; the root agent dispatches each fresh collaboration turn.
No provider token/cost or sandbox-isolation claim is inferred from these files.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from repair_driver import observe,commit,identifier,link,read,write_new_json,context


def prepare(out,task,method,index):
    if method not in ('llm_feedback','llm_feedback_off'):raise ValueError('model arm required')
    identity=identifier(task,method,index);obs=observe(out,task,method,index)
    response=out/'responses'/(identity+'.json')
    prompt=f'''You are the single-candidate proposer in a frozen PulseMorph robust repair experiment.
Use ONLY this observation JSON as task evidence: {obs.resolve()}
Read that file in full. Do not read AGENTS, other files, sibling results, source code, internet or external context. Do not delegate or run physics. Prior provenance paths inside the JSON are evidence labels, not permission to open them. You may perform arithmetic or run standalone calculations on numbers supplied in this observation, but cannot import any project optimizer or simulation code.
Choose exactly one control_tool and one five-number parameter vector within the frozen bounds. Optimize the four declared development conditions under the unchanged primary target basis and physical gates. The shared historical measurements are allowed; feedback-off has no NEW outcomes. The response is a proposal, not a measurement or execution authorization.
Write exactly one NEW UTF-8 JSON response to {response.resolve()} with exactly these keys:
{{"control_tool":"one allowed tool", "parameters":[five finite JSON numbers], "only_observation_path":"{obs.resolve().as_posix()}", "brief_reason":"concise decision summary"}}
No extra keys, NaN, alternate vectors or prose in the file. Do not overwrite an existing response. Finish by stating the saved path and proposed vector. This is one model turn/one candidate slot. Requested model is gpt-6.1-sol; do not spawn agents. No retry or extra physics is allowed.'''
    record=dict(candidate_id=identity,model='gpt-6.1-sol',fork_turns='none',
        dispatch_requested_at_utc=datetime.now(timezone.utc).isoformat(),observation=link(obs),
        expected_response_path=str(response.resolve()),prompt=prompt,
        isolation='Separate fresh model context and file-access instructions; not an OS sandbox',
        provider_tokens=None,provider_billed_cost=None)
    write_new_json(out/'model_requests'/(identity+'.json'),record)
    return record


def record(out,task,method,index,agent_task):
    context(out);identity=identifier(task,method,index)
    request=read(out/'model_requests'/(identity+'.json'))
    response=out/'responses'/(identity+'.json')
    receipt=dict(candidate_id=identity,model='gpt-6.1-sol',fork_turns='none',agent_task=agent_task,
        request=link(out/'model_requests'/(identity+'.json')),observation=request['observation'],response=link(response),
        dispatch_requested_at_utc=request['dispatch_requested_at_utc'],
        completion_observed_at_utc=datetime.now(timezone.utc).isoformat(),
        provider_tokens=None,provider_billed_cost=None,status='response_observed')
    write_new_json(out/'model_calls'/(identity+'.json'),receipt)
    return commit(out,task,method,index)


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--action',choices=['prepare','record'],required=True)
    parser.add_argument('--task',required=True);parser.add_argument('--method',required=True);parser.add_argument('--index',required=True,type=int)
    parser.add_argument('--agent-task')
    args=parser.parse_args();out=args.output.resolve()
    value=prepare(out,args.task,args.method,args.index) if args.action=='prepare' else record(out,args.task,args.method,args.index,args.agent_task)
    print(json.dumps(value,allow_nan=False))
