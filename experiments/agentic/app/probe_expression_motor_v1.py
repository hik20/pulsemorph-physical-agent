"""Prepare by default; three separately authorized expression motor trials."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'src'))
from pulsemorph.agentic.app.phrase.assets import PhraseAssets, PUBLIC_PYTHON, sha
from pulsemorph.agentic.app.phrase.runtime import PhraseRuntime
from pulsemorph.agentic.app.phrase.music import PhraseMusic
from pulsemorph.agentic.app.phrase.motor_response_v1 import bridge
from pulsemorph.agentic.expression.runtime_v1 import ExpressionRuntime
from pulsemorph.agentic.phrase_expression_v1 import parse_expression, compile_expression, GAIN_FIELDS

BASE = Path.home() / '.cache/pulsemorph-agentic/analysis/expression-motor-v1'
HISTORICAL = Path.home() / '.cache/pulsemorph-agentic/app_phrase_v1/jobs/20261027T112858_b653183b/trials/baseline'


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode('utf8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def safe(path):
    path = Path(path).absolute()
    if not path.resolve().is_relative_to(BASE.resolve()) or path.resolve() == BASE.resolve():
        raise ValueError('Dedicated private expression motor output required')
    for parent in (path, *path.parents):
        if parent.is_symlink() or (parent.exists() and bool(getattr(parent.lstat(), 'st_file_attributes', 0) & 1024)):
            raise ValueError('Linked expression output forbidden')
    return path


def design():
    unit = {'schema': 'pulsemorph.phrase_expression.v1', 'order': ['wave', 'high5'],
            'heading_policy': 'align_join_heading', 'rationale': 'Preregistered High5-left expression component ablation, no model or goal optimization.',
            **{field: 1. for field in GAIN_FIELDS}}
    return [{'label': 'unit', 'index': -1, 'expression': parse_expression(unit)},
            {'label': 'pose_low', 'index': 0, 'expression': parse_expression(dict(unit, second_left_pose_gain=.75))},
            {'label': 'motion_low', 'index': 1, 'expression': parse_expression(dict(unit, second_left_motion_gain=.75))}]


def own_pins():
    names = ('experiments/agentic/app/probe_expression_motor_v1.py',
             'src/pulsemorph/agentic/app/phrase/motor_response_v1.py')
    return {str(PROJECT / name): sha(PROJECT / name) for name in names}


def verify(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Expression motor input changed')


def historical_inventory():
    result = {}
    for path in HISTORICAL.rglob('*'):
        if path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 1024):
            raise ValueError('Linked historical bridge forbidden')
        if path.is_file():
            result[path.relative_to(HISTORICAL).as_posix()] = sha(path)
    return result


def prepare(output):
    output = safe(output)
    assets = PhraseAssets(PROJECT)
    request = json.loads((HISTORICAL / 'request.json').read_bytes())
    saved = json.loads((HISTORICAL / 'result.json').read_bytes())
    old = PhraseRuntime.__new__(PhraseRuntime)
    old.root, old.python, old.assets = PROJECT, PUBLIC_PYTHON.resolve(), assets
    old.environment, old.artifact_root = request['environment'], HISTORICAL.parents[3]
    old._implementation = old.implementation_pins()
    old._verify_trial({'id': saved['id'], 'index': saved['index'], 'program': saved['program'],
                       'metrics': saved['metrics'], 'paths': {'directory': str(HISTORICAL)}})
    runtime = ExpressionRuntime.__new__(ExpressionRuntime)
    runtime.root, runtime.python, runtime.assets = PROJECT, PUBLIC_PYTHON.resolve(), assets
    runtime.environment, runtime.artifact_root = request['environment'], output
    runtime._implementation = runtime.implementation_pins()
    runtime._unchanged()
    music = PhraseMusic.__new__(PhraseMusic)
    music.root = PROJECT
    music.directory = Path.home() / '.cache/pulsemorph-agentic/app_phrase_v1/music'
    music._warnings = []
    measured = music.get('phrase_original_pulse_v1')
    if measured['sha256'] != '0e0d161cbf1f109e65af17d5483c6f5172715494391859d24108438e781a6893':
        raise ValueError('Fixed owned music changed')
    observation = runtime._music({'available': True, 'path': measured['path'], 'sha256': measured['sha256'],
                                 'duration_s': 8, 'sample_rate': 48000, 'observed_through_sample': 384000})
    pins = {**runtime._implementation, **own_pins(), observation['path']: observation['sha256']}
    verify(pins)
    parent = output
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 1024 ** 3:
        raise ValueError('Expression motor pilot requires one GiB free')
    return {'schema': 'pulsemorph.expression_motor_prepared.v1', 'output': str(output), 'job_id': 'expression_motor_v1',
            'design': design(), 'source_pins': pins, 'environment': request['environment'],
            'environment_provenance': {'historical_request_sha256': sha(HISTORICAL / 'request.json'), 'fresh_environment_worker': False},
            'music_observation': observation, 'music_role': 'Identical required runtime provenance; not manipulated and no model sees music.',
            'historical_unit_directory': str(HISTORICAL), 'historical_inventory': historical_inventory(),
            'baseline_bridge': 'Five raw payload bytes, all eight inputs/four outputs/telemetry array bytes, policy calls and environment/counters exact before variants.',
            'budget': {'physical_trials': 3, 'reference_compiles': 6, 'online_ONNX': 1200, 'warmup_ONNX': 3, 'total_ONNX': 1203,
                       'physics_steps': 24000, 'runtime_saved_state_FK': 1200, 'model_calls': 0, 'independent_new_FK': 0,
                       'controller_wall_s': 900, 'harness_wall_s': 1200, 'retries': 0},
            'qualification': 'High5-left component total trajectory response, one deterministic realization per expression. Original own-q0 metric/screens unchanged; temporal/mean diagnostics separate. No new skill/style/goal success or same-state policy derivative claim.'}


def execute(output, plan, authority_path, authority_sha256):
    if not authority_path or not authority_sha256 or sha(authority_path) != authority_sha256:
        raise ValueError('Exact root expression execution authority required')
    authority = json.loads(Path(authority_path).read_bytes())
    required = {'schema': 'pulsemorph.expression_motor_authority.v1', 'approved': True,
                'prepared_sha256': digest(encoded(plan)), 'max_trials': 3, 'max_reference_compiles': 6,
                'max_ONNX_calls': 1203, 'max_physics_steps': 24000, 'max_runtime_saved_state_FK': 1200,
                'max_model_calls': 0, 'max_independent_FK': 0}
    if any(type(authority.get(k)) is not type(v) or authority.get(k) != v for k, v in required.items()):
        raise ValueError('Expression execution budget/request differs')
    if encoded(prepare(output)) != encoded(plan):
        raise ValueError('Expression preparation changed')
    write(output / 'execution_attempt.json', {'attempted_once': True, 'authority_sha256': authority_sha256, 'retry_allowed': False})
    deadline = time.monotonic() + 1200
    rows = []
    try:
        from pulsemorph.agentic.app.phrase.entrypoint_v2 import prepare_parent_imports
        prepare_parent_imports(PROJECT)
        assets = PhraseAssets(PROJECT)
        for item in plan['design']:
            verify(plan['source_pins'])
            ref, report = compile_expression(assets.references(), item['expression'], reference_id='expression_motor_preflight_' + item['label'], sources=assets.source_records())
            write(output / ('preflight_' + item['label'] + '.json'), {'report': report, 'numeric_arrays': {
                name: digest(getattr(ref, name).tobytes()) for name in ('time_offsets_s', 'joint_position_rad', 'joint_velocity_rad_s', 'anchor_orientation_xyzw')}})
        runtime = ExpressionRuntime(PROJECT, PUBLIC_PYTHON, output)
        if runtime.environment != plan['environment']:
            raise ValueError('Fresh expression environment differs from prepared historical proof')
        expected = {p: h for p, h in plan['source_pins'].items() if p not in own_pins() and p != plan['music_observation']['path']}
        if runtime.implementation_pins() != expected:
            raise ValueError('Fresh expression runtime pins differ')
        for item in plan['design']:
            if time.monotonic() >= deadline:
                raise RuntimeError('Expression harness wall exhausted')
            verify(plan['source_pins'])
            if historical_inventory() != plan['historical_inventory']:
                raise ValueError('Historical unit bridge changed')
            label = item['label']
            write(output / (label + '_attempt.json'), item)
            cancel = threading.Event()
            timer = threading.Timer(max(0., deadline - time.monotonic()), cancel.set)
            timer.daemon = True
            timer.start()
            try:
                trial = runtime.run_trial(item['expression'], output / 'trials' / label, plan['job_id'], item['index'], plan['music_observation'], cancel)
            finally:
                timer.cancel()
            rows.append({'label': label, **trial})
            write(output / (label + '_completed.json'), trial)
            from pulsemorph.motion.contracts import load_reference
            actual = load_reference(Path(trial['paths']['directory']) / 'common_reference')
            expected_arrays = json.loads((output / ('preflight_' + label + '.json')).read_bytes())['numeric_arrays']
            if any(digest(getattr(actual, name).tobytes()) != h for name, h in expected_arrays.items()):
                raise ValueError('Actual expression reference differs from preflight')
            if label == 'unit':
                write(output / 'unit_bridge.json', bridge(trial['paths']['directory'], HISTORICAL))
        verify(plan['source_pins'])
        if historical_inventory() != plan['historical_inventory']:
            raise ValueError('Historical bridge changed before completion')
        write(output / 'completed.json', {'schema': 'pulsemorph.expression_motor_completed.v1', 'trials': rows,
              'new_model_calls': 0, 'qualification': plan['qualification']})
        print(json.dumps({'status': 'three_expression_trials_completed', 'new_model_calls': 0}))
    except Exception:
        write(output / 'failed.json', {'status': 'stopped', 'completed_labels': [r['label'] for r in rows],
              'retry_allowed': False, 'new_model_calls': 0, 'partial_attempts': 'Read preserved attempts/results; no inferred zero physical work.'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--authority-sha256')
    args = parser.parse_args()
    output = safe(args.output)
    plan = prepare(output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'prepared.json', plan)
    elif not args.execute:
        raise FileExistsError('Exclusive new preparation required')
    elif encoded(json.loads((output / 'prepared.json').read_bytes())) != encoded(plan):
        raise ValueError('Prepared expression plan changed')
    if not args.execute:
        print(json.dumps({'status': 'prepared_only', 'prepared_sha256': digest(encoded(plan)), 'new_compiles_worker_model_ONNX_FK_steps': 0}))
        return
    execute(output, plan, args.authority, args.authority_sha256)


if __name__ == '__main__':
    main()
