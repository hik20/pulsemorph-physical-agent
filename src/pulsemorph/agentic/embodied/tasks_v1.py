"""Immutable world-wrist target assessment from actual observations only.

Hold means sampled occupancy at the declared maximum gap, never inferred
continuous occupancy through a controller chunk. This module has no FK/actions.
"""
import copy
import hashlib
import json
import math

SCHEMA = 'pulsemorph.embodied_wrist_task.v1'
BODIES = ('right_rubber_hand', 'left_rubber_hand')


def _number(value, low, high):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_task(task):
    fields = {'schema', 'task_id', 'body', 'frame', 'target_world_m', 'tolerance_m',
              'hold_s', 'max_sample_gap_s', 'min_root_height_m'}
    if type(task) is not dict or set(task) != fields or task['schema'] != SCHEMA:
        raise ValueError('Exact immutable wrist task required')
    if type(task['task_id']) is not str or not 1 <= len(task['task_id']) <= 100:
        raise ValueError('Bounded task identity required')
    if task['body'] not in BODIES or task['frame'] != 'world':
        raise ValueError('Named actual wrist in world frame required')
    target = task['target_world_m']
    if type(target) is not list or len(target) != 3 or not all(_number(v, -10, 10) for v in target):
        raise ValueError('Finite three-coordinate world target required')
    for field, low, high in [('tolerance_m', .0001, .5), ('hold_s', 0, 5),
                             ('max_sample_gap_s', .001, .05), ('min_root_height_m', .6, 2)]:
        if not _number(task[field], low, high): raise ValueError('Invalid task bound: ' + field)
    return copy.deepcopy(task)


def evaluate_task(task, observation, prior_state=None):
    """Orchestrator callback: return done/failed/reason/state plus measurements.

    Repeated identical boundary observations are idempotent. Different data at
    the same revision, regressing clocks or task/session changes fail closed.
    Physical failure is failed=True, never successful done=True.
    """
    task = validate_task(task)
    obs = observation
    if type(obs) is not dict or obs.get('schema') != 'pulsemorph.embodied_observation.v1':
        raise ValueError('Actual embodied observation required')
    session, revision, clock = obs.get('session_id'), obs.get('revision'), obs.get('sim_time_s')
    if type(session) is not str or not session or type(revision) is not int or revision < 0 or not _number(clock, 0, 1e9):
        raise ValueError('Actual session/revision/simulation clock required')
    task_hash, observation_hash = _hash(task), _hash(obs)
    if prior_state is not None:
        if type(prior_state) is not dict or prior_state.get('schema') != 'pulsemorph.wrist_task_state.v1':
            raise ValueError('Typed task state required')
        if prior_state.get('task_sha256') != task_hash or prior_state.get('session_id') != session:
            raise ValueError('Fixed task/session changed')
        if revision == prior_state['revision']:
            if observation_hash != prior_state['observation_sha256']: raise ValueError('Same revision changed')
            cached = copy.deepcopy(prior_state['cached_assessment'])
            cached['state'] = copy.deepcopy(prior_state)
            return cached
        if revision < prior_state['revision'] or clock <= prior_state['sim_time_s']:
            raise ValueError('Stale/regressing observation')
    body = obs.get('body_poses', {}).get(task['body'], {})
    position = body.get('position_world_m')
    if type(position) is not list or len(position) != 3 or not all(_number(v, -1e6, 1e6) for v in position):
        raise ValueError('Actual world wrist position required')
    flags = obs.get('physical_flags')
    if type(flags) is not dict or any(type(flags.get(k)) is not bool for k in ('finite', 'warnings_unchanged', 'stopped')):
        raise ValueError('Actual physical flags required')
    if not all(_number(flags.get(k), -1e6, 1e6) for k in ('root_height_m', 'min_root_height_m')):
        raise ValueError('Actual root height measurements required')
    stable = (flags['finite'] and flags['warnings_unchanged'] and not flags['stopped']
              and min(flags['root_height_m'], flags['min_root_height_m']) >= task['min_root_height_m'])
    error = [target - actual for target, actual in zip(task['target_world_m'], position)]
    distance = math.sqrt(sum(v*v for v in error))
    inside = distance <= task['tolerance_m']
    started = None
    gap = None if prior_state is None else clock - prior_state['sim_time_s']
    if stable and inside:
        started = clock
        if prior_state is not None and gap <= task['max_sample_gap_s'] + 1e-12 and prior_state.get('inside'):
            started = prior_state['hold_started_s']
    held = 0. if started is None else clock - started
    done = stable and inside and held + 1e-12 >= task['hold_s']
    state = {'schema': 'pulsemorph.wrist_task_state.v1', 'task_sha256': task_hash,
             'session_id': session, 'revision': revision, 'sim_time_s': clock,
             'observation_sha256': observation_hash, 'inside': stable and inside,
             'hold_started_s': started}
    report = {'done': done, 'failed': not stable,
              'reason': 'physical_gate_failed' if not stable else 'target_sampled_hold_met' if done else 'target_outside_tolerance' if not inside else 'sampled_hold_pending',
              'task_id': task['task_id'], 'distance_m': distance, 'signed_error_world_m': error,
              'inside_tolerance': inside, 'sampled_hold_s': held, 'observed_gap_s': gap,
              'physical_gate_passed': stable,
              'qualification': 'Actual world wrist observations; sampled hold only, no interpolation or continuous-path/safety claim.',
              'state': state}
    # Store a bounded non-recursive cached report. Repeated reads cannot add time.
    state['cached_assessment'] = copy.deepcopy(report)
    return report
