"""Immutable AND of two actual world-wrist goals; no action selection or FK."""
import copy
from .tasks_v1 import validate_task as validate_wrist_task,evaluate_task as evaluate_wrist_task,_hash

SCHEMA='pulsemorph.embodied_dual_wrist_task.v1'
STATE_SCHEMA='pulsemorph.embodied_dual_wrist_state.v1'

def validate_task(task):
    if type(task) is not dict or set(task)!={'schema','task_id','goals'} or task['schema']!=SCHEMA:
        raise ValueError('Exact immutable two-wrist task required')
    if type(task['task_id']) is not str or not 1<=len(task['task_id'])<=100:raise ValueError('Bounded dual task identity required')
    task['task_id'].encode('utf8')
    if type(task['goals']) is not list or len(task['goals'])!=2:raise ValueError('Exactly two wrist goals required')
    goals=[validate_wrist_task(v) for v in task['goals']]
    if {g['body'] for g in goals}!={'right_rubber_hand','left_rubber_hand'} or len({g['task_id'] for g in goals})!=2:
        raise ValueError('Distinct left/right bodies and goal identities required')
    return {'schema':SCHEMA,'task_id':task['task_id'],'goals':goals}

def evaluate_task(task,observation,prior_state=None):
    task=validate_task(task);task_hash=_hash(task)
    if type(observation) is not dict:raise ValueError('Actual observation object required')
    if prior_state is not None:
        if (type(prior_state) is not dict or set(prior_state)!={'schema','task_sha256','session_id','revision','observation_sha256','goals'} or
            prior_state['schema']!=STATE_SCHEMA or prior_state['task_sha256']!=task_hash or
            prior_state['session_id']!=observation.get('session_id') or type(prior_state['revision']) is not int or
            type(prior_state['goals']) is not dict or set(prior_state['goals'])!={'right_rubber_hand','left_rubber_hand'}):
            raise ValueError('Dual task/session/state assignment changed')
        if any(type(s) is not dict or s.get('session_id')!=prior_state['session_id'] or s.get('revision')!=prior_state['revision'] or s.get('observation_sha256')!=prior_state['observation_sha256'] for s in prior_state['goals'].values()):
            raise ValueError('Per-goal observation binding differs from dual state')
        if observation.get('revision')==prior_state['revision'] and _hash(observation)!=prior_state['observation_sha256']:
            raise ValueError('Same dual revision changed')
    reports=[]
    for goal in task['goals']:
        previous=None if prior_state is None else prior_state['goals'][goal['body']]
        reports.append(evaluate_wrist_task(goal,observation,previous))
    failed=any(r['failed'] for r in reports);done=not failed and all(r['done'] for r in reports)
    worst=max(range(2),key=lambda i:reports[i]['distance_m']);details=[]
    for goal,report in zip(task['goals'],reports):
        details.append(goal['body']+' distance='+format(report['distance_m']*1000,'.6f')+'mm tolerance='+format(goal['tolerance_m']*1000,'.6f')+'mm '+('met' if report['done'] else 'not met')+' physical='+('pass' if report['physical_gate_passed'] else 'fail'))
    state={'schema':STATE_SCHEMA,'task_sha256':task_hash,'session_id':observation['session_id'],'revision':observation['revision'],'observation_sha256':_hash(observation),'goals':{g['body']:copy.deepcopy(r['state']) for g,r in zip(task['goals'],reports)}}
    return {'done':done,'failed':failed,'reason':('physical_gate_failed; ' if failed else 'both_wrist_goals_met; ' if done else 'dual_wrist_goals_pending; ')+'; '.join(details),
        'task_id':task['task_id'],'distance_m':reports[worst]['distance_m'],'signed_error_world_m':copy.deepcopy(reports[worst]['signed_error_world_m']),
        'inside_tolerance':all(r['inside_tolerance'] for r in reports),'sampled_hold_s':min(r['sampled_hold_s'] for r in reports),
        'physical_gate_passed':all(r['physical_gate_passed'] for r in reports),'goal_count':2,'goals_passed':sum(r['done'] for r in reports),
        'goals':[{k:copy.deepcopy(v) for k,v in report.items() if k!='state'}|{'body':goal['body'],'target_world_m':copy.deepcopy(goal['target_world_m']),'tolerance_m':goal['tolerance_m']} for goal,report in zip(task['goals'],reports)],
        'state':state,'qualification':'Both fixed world-wrist goals must hold in the SAME actual observation. Sampled hold only; no inferred continuous occupancy, action selection, FK or goal rebasing.'}

validate_dual_task=validate_task
evaluate_dual_task=evaluate_task
