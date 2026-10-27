"""Prepare a measured-response planning request; one explicit model attempt only."""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'src'))
from jsonschema import Draft202012Validator
from pulsemorph.agentic.app.phrase.assets import PhraseAssets, PUBLIC_PYTHON
from pulsemorph.agentic.app.phrase.runtime import PhraseRuntime
from pulsemorph.agentic.app.phrase.music import PhraseMusic
from pulsemorph.agentic.app.phrase.source_descriptor_v1 import describe_sources
from pulsemorph.agentic.app.phrase.response_evidence_v1 import load_response_evidence
from pulsemorph.agentic.app.phrase.manager_goals_v4 import prepare_initial, INITIAL_SCHEMA_V4
from pulsemorph.agentic.app.phrase.goals_v1 import validate_goals
from pulsemorph.agentic.phrase_planner import validate_program, safe_provider_error_category

BASE = Path.home() / '.cache/pulsemorph-agentic/analysis/response-plan-probe-v1'
GOALS = {'schema': 'pulsemorph.phrase_goals.v1', 'ratios': [.8, 1., 1., 1.]}
INSTRUCTION = ('Önce Wave, sonra High5; birleşimde yönü hizala. İlk bölüm sol kol için gerçekleşmiş '
               'own-q0 RMS oranı hedefi0.8, diğer üç fizik hedefi1 olsun. Diğer üç kaynak kazancını1 '
               'tut. Yalnız bu koşulda gerçekten ölçülmüş üç ilk-sol kazanç noktası arasından hedefe '
               'yakın bir aday öner. Noktalar arası tahmin veya başka kazançlara aktarım yapma. '
               'Gerekçende hangi ölçülmüş noktayı kullandığını ve koşulunu belirt. Hedef metriğini '
               'değiştirme; fiziksel başarı/ritim/estetik/yeni beceri garantisi istemiyorum. Bu çağrı '
               'yalnız plan üretir, yeni fizik denemesi değildir.')


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode('utf8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def pins():
    names = ['experiments/agentic/app/probe_response_plan_v1.py',
             'src/pulsemorph/agentic/app/phrase/response_evidence_v1.py',
             'src/pulsemorph/agentic/app/phrase/manager_response_v8.py',
             'src/pulsemorph/agentic/app/phrase/source_descriptor_v1.py',
             'src/pulsemorph/agentic/app/phrase/assets.py',
             'src/pulsemorph/agentic/app/phrase/runtime.py',
             'src/pulsemorph/agentic/app/phrase/music.py',
             'src/pulsemorph/agentic/app/phrase/manager_goals_v4.py',
             'src/pulsemorph/agentic/app/phrase/goals_v1.py',
             'src/pulsemorph/agentic/app/phrase/catalog.py',
             'src/pulsemorph/agentic/app/providers.py',
             'src/pulsemorph/agentic/phrase_planner.py']
    return {name: sha((PROJECT / name).read_bytes()) for name in names}


def prepare():
    before = pins()
    # No constructor: reuse the saved, hash-bound environment description, then
    # let the evidence loader verify its actual current binaries and raw trials.
    saved = Path.home() / '.cache/pulsemorph-agentic/analysis/motor-response-v1/prepared_v1/prepared.json'
    historical_bytes = saved.read_bytes()
    if sha(historical_bytes) != '888bea169d350381c22bbb4b9c53e24c681237d9bd0f8216f6f9360d2ffb4997':
        raise ValueError('Expected immutable motor preparation changed')
    historical = json.loads(historical_bytes)
    runtime = PhraseRuntime.__new__(PhraseRuntime)
    runtime.root, runtime.python = PROJECT, PUBLIC_PYTHON.resolve()
    runtime.assets = PhraseAssets(PROJECT)
    runtime.environment = historical['environment']
    runtime.artifact_root = saved.parent
    runtime._implementation = runtime.implementation_pins()
    evidence = load_response_evidence(runtime)
    if evidence['status'] != 'available':
        raise ValueError('Actual measured response store required for this probe')
    music = PhraseMusic.__new__(PhraseMusic)
    music.root = PROJECT
    music.directory = Path.home() / '.cache/pulsemorph-agentic/app_phrase_v1/music'
    music._warnings = []
    measured = music.get('phrase_original_pulse_v1')
    if measured['sha256'] != '0e0d161cbf1f109e65af17d5483c6f5172715494391859d24108438e781a6893':
        raise ValueError('Expected owned original eight-second music changed')
    keep = ('schema', 'sha256', 'duration_s', 'sample_rate', 'sample_frames', 'channels', 'pcm_bits',
            'rms', 'peak_amplitude', 'accent_time_s', 'accent_measurement', 'onset_times_s',
            'sections', 'bpm', 'bpm_source', 'rights', 'limitations')
    summary = {key: value for key, value in measured.items() if key in keep}
    context = {'source_reference_descriptor': describe_sources(runtime.assets),
               'measured_motor_response_evidence': evidence}
    planned = prepare_initial(summary, INSTRUCTION, GOALS, context, candidate_budget=1)
    if pins() != before:
        raise ValueError('Probe sources changed during preparation')
    return {'schema': 'pulsemorph.response_plan_probe.v1', 'prepared_plan': planned,
            'source_pins': before, 'music_sha256': measured['sha256'],
            'evidence_sha256': sha(encoded(evidence)), 'max_model_attempts': 1,
            'physical_authority': False,
            'qualification': 'Prompted observed-point selection only; no fresh physical/E2E, calibration, causal benefit or novel skill proof.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-model', action='store_true')
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--authority-sha256')
    args = parser.parse_args()
    output = args.output.absolute()
    if not output.resolve().is_relative_to(BASE.resolve()) or output.resolve() == BASE.resolve():
        raise ValueError('Dedicated private response-plan output required')
    for path in (output, *output.parents):
        if path.is_symlink() or (path.exists() and bool(getattr(path.lstat(), 'st_file_attributes', 0) & 1024)):
            raise ValueError('Linked output path forbidden')
    prepared = prepare()
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'prepared_request.json', prepared)
    elif not args.execute_model:
        raise FileExistsError('Exclusive new preparation required')
    elif encoded(json.loads((output / 'prepared_request.json').read_bytes())) != encoded(prepared):
        raise ValueError('Prepared request changed')
    if not args.execute_model:
        print(json.dumps({'status': 'prepared_only', 'request_sha256': sha(encoded(prepared)),
                          'new_model_compile_ONNX_FK_steps': 0}))
        return
    if not args.authority or sha(args.authority.read_bytes()) != args.authority_sha256:
        raise ValueError('Exact root authority file/hash required')
    authority = json.loads(args.authority.read_bytes())
    required = {'schema': 'pulsemorph.response_plan_authority.v1', 'approved': True,
                'prepared_sha256': sha(encoded(prepared)), 'max_model_attempts': 1,
                'max_model_wall_s': 120, 'max_compile_ONNX_FK_steps': 0}
    if any(type(authority.get(k)) is not type(v) or authority.get(k) != v for k, v in required.items()):
        raise ValueError('Root authority budget/request differs')
    write(output / 'model_attempt.json', {'attempted': True, 'completion_unconfirmed': True,
          'at': datetime.now(timezone.utc).isoformat(), 'request_sha256': sha(encoded(prepared)),
          'authority_sha256': args.authority_sha256, 'retry_allowed': False})
    completed = False
    try:
        from pulsemorph.agentic.app.providers import CodexLocalProvider
        result = CodexLocalProvider(trusted_local=True).generate_json(
            prepared['prepared_plan']['prompt'], copy.deepcopy(INITIAL_SCHEMA_V4), timeout=120)
        completed = True
        clean = {k: copy.deepcopy(result.receipt[k]) for k in ('provider', 'model_requested',
                 'model_returned', 'usage', 'live', 'tool_events', 'elapsed_seconds', 'trust_boundary') if k in result.receipt}
        clean.update(completion_unconfirmed=False, request_sha256=sha(encoded(prepared)),
                     response_sha256=sha(encoded(result.data)))
        write(output / 'response.json', result.data)
        write(output / 'receipt.json', clean)
        if clean.get('model_requested') != 'gpt-6.1-sol' or clean.get('tool_events') != 0:
            raise ValueError('Provider boundary differs')
        Draft202012Validator(INITIAL_SCHEMA_V4).validate(result.data)
        validate_program(result.data['program'])
        if validate_goals(result.data['goals']) != GOALS or pins() != prepared['source_pins']:
            raise ValueError('Goal/source binding differs')
        write(output / 'completed_status.json', {'status': 'plan_only_response_validated',
              'completed_model_calls': 1, 'new_compile_ONNX_FK_steps': 0,
              'supported': result.data['supported'], 'unsupported_requests': result.data['unsupported_requests'],
              'qualification': prepared['qualification']})
        print(json.dumps({'status': 'plan_only_response_validated', 'completed_model_calls': 1}))
    except Exception as error:
        write(output / 'failed_status.json', {'status': 'probe_failed', 'retry_allowed': False,
              'completed_provider_return': completed, 'completion_unconfirmed': not completed,
              'error_category': safe_provider_error_category(error), 'new_compile_ONNX_FK_steps': 0})
        raise SystemExit(1)
    finally:
        write(output / 'source_after.json', {'source_pins': pins(),
              'unchanged': pins() == prepared['source_pins'], 'new_compile_ONNX_FK_steps': 0})


if __name__ == '__main__':
    main()
