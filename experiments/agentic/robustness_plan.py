"""Freeze a finite replay panel without executing physics or calling a model."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from robustness_scenarios import build_scenarios

ROOT = Path(__file__).resolve().parents[2]
PILOT = Path('artifacts/runs/agentic/tool_selection_development_v2')
PROFILE = Path('research/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml')
PROTECTED = (PILOT, Path('artifacts/runs/agentic/tool_selection_development_v2_compatible_refs_v1'),
             Path('artifacts/runs/agentic/tool_selection_development_v2_projection_audit_v1'))
NEW_CODE = ('robustness_plan.py', 'robustness_scenarios.py', 'robustness_driver.py',
            'execute_robustness.py', 'evaluate_robustness.py', 'audit_robustness.py',
            'audit_multifeature.py', 'run_tool_compatibility_smoke.py')
COUNTS = dict(physical_trials=78, calibration_trials=26, selected_trials=52,
              perturbed_selected_trials=48, new_llm_calls=0)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def link(relative):
    return {'path': Path(relative).as_posix(), 'sha256': sha(ROOT / relative)}


def write_new(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def protected_inventory(relative):
    directory = ROOT / relative
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    # Old trial snapshots contain hardlinks; hash identical file identities once.
    cache, files = {}, {}
    for path in sorted(directory.rglob('*')):
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity not in cache:
            cache[identity] = sha(path)
        files[path.relative_to(directory).as_posix()] = cache[identity]
    return {'path': relative.as_posix(), 'files': files}


def original_cases(original, comparison):
    """Validate hard-declared selections against the actual preserved result."""
    cases = []
    selected = {'wave_a': 7, 'wave_b': 7, 'high5_a': 6, 'high5_b': 6}
    ids = [('calibration_' + source, source, 'calibration_' + source)
           for source in ('wave', 'high5')]
    for task, index in selected.items():
        row = next(t for t in comparison['tasks'] if t['task_id'] == task)
        method = next(m for m in row['methods'] if m['method'] == 'llm_feedback')
        candidate = task + '__llm_feedback__' + str(index)
        if method['selected']['candidate_id'] != candidate:
            raise ValueError('preserved selected candidate differs: ' + task)
        ids.append(('selected_' + task, row['source_id'], candidate))
    for role, source, candidate in ids:
        relative = PILOT / 'proposals' / (candidate + '.json')
        proposal = read(ROOT / relative)
        evaluation = read(ROOT / PILOT / 'evaluations' / (candidate + '.json'))
        if proposal['source_id'] != source or not evaluation['integrity_passed']:
            raise ValueError('invalid preserved source/evaluation: ' + candidate)
        if role.startswith('selected_') and not (evaluation['intent_passed'] and evaluation['physical_gate_passed']):
            raise ValueError('preserved feedback selection did not pass: ' + candidate)
        cases.append(dict(role=role, source_id=source, task_id=proposal['task_id'],
                          original_proposal=link(relative), original_trial=(PILOT/'trials'/candidate).as_posix(),
                          goal=None if role.startswith('calibration_') else deepcopy(next(t['goal'] for t in original['tasks'] if t['task_id'] == proposal['task_id']))))
    return cases


def build_protocol():
    """Read and pin current inputs; caller publishes once before any physics."""
    import yaml
    original = read(ROOT / PILOT / 'protocol.json')
    comparison = read(ROOT / PILOT / 'comparison.json')
    accepted = PROTECTED[2] / 'audit.json'
    if read(ROOT / accepted).get('audit_passed') is not True:
        raise ValueError('accepted projection audit required')
    profile = yaml.safe_load((ROOT / PROFILE).read_text(encoding='utf8'))
    names = profile['joint_names']
    if len(names) != 29 or profile['robot']['joint_names'] != names:
        raise ValueError('ordered G1 29-joint profile required')
    keys = ('sources', 'parameter_names', 'bounds', 'anchor', 'feature_names',
            'feature_windows_s', 'feature_tolerances', 'physical_gate', 'accent_window_s',
            'accent_smoothing_samples', 'accent_minimum_prominence', 'section_windows_s',
            'duration_s', 'source_start_s', 'tasks')
    result = {key: deepcopy(original[key]) for key in keys}
    cases = original_cases(original, comparison)
    runtime_inputs = {}
    for case in cases:
        manifest = read(ROOT / case['original_trial'] / 'manifest.json')
        for path, digest in manifest['preexecution_input_hashes'].items():
            path = path.replace('\\', '/')
            if path in runtime_inputs and runtime_inputs[path] != digest:
                raise ValueError('inconsistent original runtime input pin: ' + path)
            runtime_inputs[path] = digest
    for path, digest in runtime_inputs.items():
        if sha(ROOT/path) != digest:
            raise ValueError('original runtime input changed: ' + path)
    dependencies = set(original['execution_code_sha256'])
    dependencies.update(path for path in runtime_inputs if path.endswith('.py'))
    dependencies.update('experiments/agentic/' + name for name in NEW_CODE)
    # Imported existing evaluators/contracts and their pinned upstream runtime
    # remain explicit, alongside every new execution/audit module.
    dependencies.add('experiments/agentic/execute_tool_selection_compatible.py')
    result.update(schema='pulsemorph.agentic_initial_state_robustness_protocol.v1',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  original_protocol=link(PILOT/'protocol.json'), original_comparison=link(PILOT/'comparison.json'),
                  accepted_projection_audit=link(accepted), controller_profile=link(PROFILE),
                  structural_preflight=link(Path('artifacts/runs/agentic/robustness_structural_preflight_v1/report.json')),
                  runtime_input_sha256=runtime_inputs,
                  execution_code_sha256={path: sha(ROOT/path) for path in sorted(dependencies)},
                  protected_inventories=[protected_inventory(path) for path in PROTECTED],
                  cases=cases, scenarios=build_scenarios(names),
                  goals={task['task_id']: deepcopy(task['goal']) for task in original['tasks']},
                  expected_counts=COUNTS.copy(), training_admission=False, quality='unassessed',
                  primary='fixed fresh nominal center and denominator; byte-equivalent to original nominal',
                  secondary='matched perturbed reference own center and denominator',
                  nominal_gate='All six nominal full-trace byte parities before any nonzero trial',
                  yaw_interpretation='heading-equivariance under normal recomputed upstream heading alignment',
                  stopping_policy='No retuning, model calls, silent retries, clipping or outcome-dependent panel exclusions.')
    result['protected_evidence'] = [dict(path=item['path'], inventory_file='protected_inventories/'+str(i)+'.json')
                                    for i, item in enumerate(result['protected_inventories'])]
    return result


def replay_proposal(original, case, scenario, protocol_sha256):
    value = deepcopy(original)
    identity = scenario['scenario_id'] + '__' + case['role']
    value.update(candidate_id=identity, execution_kind='initial_state_sensitivity_replay',
                 observation_id=identity + ':replay-observation',
                 observation_artifact=None, response_artifact=None,
                 parameter_source=deepcopy(case['original_proposal']),
                 initial_perturbation=deepcopy(scenario['initial_perturbation']),
                 protocol_sha256=protocol_sha256,
                 runtime_session_id=protocol_sha256 + ':' + scenario['scenario_id'] + ':' + case['role'])
    return value


def freeze(output):
    """Publish exclusive-create protocol + 78 proposals; existing path is refused.

    Failure leaves partial artifacts visible, never retries or overwrites them.
    No simulation/model call occurs. Code/evidence must remain unchanged later.
    """
    output = Path(output).resolve()
    allowed = (ROOT/'artifacts/runs/agentic').resolve()
    if allowed not in output.parents:
        raise ValueError('panel output must be strictly under agentic runs')
    if output.exists():
        raise FileExistsError(output)
    protocol = build_protocol()
    output.mkdir(parents=True, exist_ok=False)
    (output/'proposals').mkdir()
    for relative, digest in protocol.get('execution_code_sha256', {}).items():
        source = ROOT/relative
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('code changed before snapshot: ' + relative)
        target = output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        if sha(target) != digest or sha(source) != digest:
            raise ValueError('code changed during snapshot: ' + relative)
    if protocol.get('protected_inventories'):
        (output/'protected_inventories').mkdir()
        for index, item in enumerate(protocol['protected_inventories']):
            write_new(output/'protected_inventories'/(str(index)+'.json'), item['files'])
    write_new(output/'protocol.json', protocol)
    digest = sha(output/'protocol.json')
    for scenario in protocol['scenarios']:
        for case in protocol['cases']:
            original = read(ROOT / case['original_proposal']['path'])
            if sha(ROOT / case['original_proposal']['path']) != case['original_proposal']['sha256']:
                raise ValueError('original proposal changed during freeze')
            value = replay_proposal(original, case, scenario, digest)
            write_new(output/'proposals'/(value['candidate_id']+'.json'), value)
    return protocol
