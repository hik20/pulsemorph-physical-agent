"""Prepare-only reference-expression pilot; explicit authority permits ten compiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT)]
from pulsemorph.agentic.app.phrase.assets import PhraseAssets, sha, PUBLIC_PYTHON
from pulsemorph.agentic.phrase_expression_v1 import parse_expression, compile_expression, describe_bases, GAIN_FIELDS
from pulsemorph.agentic.phrase_program import parse_program, compile_program, ProgramCompileError
from pulsemorph.motion.contracts import save_reference

BASE = Path.home() / '.cache/pulsemorph-agentic/analysis/phrase-expression-reference-v1'


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode('utf8')


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def design():
    unit = {'schema': 'pulsemorph.phrase_expression.v1', 'order': ['wave', 'high5'],
            'heading_policy': 'align_join_heading', 'rationale': 'Preregistered reference-only expression pilot.',
            **{field: 1. for field in GAIN_FIELDS}}
    rows = [{'label': 'unit', 'expression': parse_expression(unit)}]
    mild = dict(unit)
    for arm, gain in zip(('first_left', 'first_right', 'second_left', 'second_right'), (.85, 1.15, 1.15, .85)):
        mild[arm + '_pose_gain'] = mild[arm + '_motion_gain'] = gain
    rows.append({'label': 'equal_mild', 'expression': parse_expression(mild)})
    for kind in ('pose', 'motion'):
        for label, gain in (('low', .75), ('high', 1.25)):
            value = dict(unit, **{'first_left_' + kind + '_gain': gain})
            rows.append({'label': kind + '_' + label, 'expression': parse_expression(value)})
    negative = dict(unit, second_left_pose_gain=1.25, second_left_motion_gain=1.25)
    rows.append({'label': 'negative_equal_second_left_high', 'expression': parse_expression(negative)})
    return rows


def pins(assets):
    values = {str(PROJECT / p): h for p, h in assets.manifest['live_numeric_pins'].items()}
    values.update({str(assets.directory / p): h for p, h in assets.manifest['files'].items()})
    values[str(assets.directory / 'manifest.json')] = sha(assets.directory / 'manifest.json')
    for name in ('src/pulsemorph/agentic/phrase_expression_v1.py',
                 'src/pulsemorph/agentic/phrase_program.py',
                 'src/pulsemorph/agentic/phrase_reference.py',
                 'src/pulsemorph/agentic/phrase_reference_v2.py',
                 'experiments/agentic/app/probe_phrase_expression_v1.py'):
        values[str(PROJECT / name)] = sha(PROJECT / name)
    values[str(PUBLIC_PYTHON)] = sha(PUBLIC_PYTHON)
    return values


def verify(values):
    for path, digest in values.items():
        if sha(path) != digest:
            raise ValueError('Reference pilot input changed')


def prepare():
    assets = PhraseAssets(PROJECT)
    before = pins(assets)
    bases = describe_bases(assets.references())
    rows = design()
    verify(before)
    return {'schema': 'pulsemorph.expression_reference_pilot.v1', 'design': rows,
            'source_pins': before, 'source_bases': bases,
            'legacy_comparisons': ['unit', 'equal_mild', 'negative_equal_second_left_high'],
            'max_expression_compiles': 7, 'max_legacy_compiles': 3,
            'max_total_compiles': 10, 'max_main_strict_future_queries': 4000,
            'new_model_ONNX_FK_physics_authority': False,
            'qualification': 'Source-joint reference expressivity only; no physical admission or Cartesian/style/goal attainment.'}


def legacy(expression):
    return parse_program({'schema': 'pulsemorph.phrase_program.v1', 'order': expression['order'],
            'heading_policy': expression['heading_policy'], 'rationale': expression['rationale'],
            **{arm + '_gain': expression[arm + '_pose_gain'] for arm in ('first_left', 'first_right', 'second_left', 'second_right')}})


def execute(output, plan, authority_path, authority_sha256):
    if not authority_path or not authority_sha256 or sha(authority_path) != authority_sha256:
        raise ValueError('Exact root authority required')
    authority = json.loads(Path(authority_path).read_bytes())
    required = {'schema': 'pulsemorph.expression_reference_authority.v1', 'approved': True,
                'plan_sha256': hashlib.sha256(encoded(plan)).hexdigest(), 'max_reference_compiles': 10,
                'max_model_ONNX_FK_physics': 0}
    if any(type(authority.get(k)) is not type(v) or authority.get(k) != v for k, v in required.items()):
        raise ValueError('Reference authority scope/budget differs')
    if encoded(prepare()) != encoded(plan):
        raise ValueError('Prepared design/source changed')
    write(output / 'execution_attempt.json', {'attempted_once': True, 'authority_sha256': authority_sha256,
          'max_compiles': 10, 'retry_allowed': False})
    from pulsemorph.agentic.app.phrase.entrypoint_v2 import prepare_parent_imports
    prepare_parent_imports(PROJECT)
    assets = PhraseAssets(PROJECT)
    refs = assets.references()
    records = assets.source_records()
    results, parity = [], {}
    count = 0
    try:
        for row in plan['design']:
            verify(plan['source_pins'])
            directory = output / row['label']
            directory.mkdir(exist_ok=False)
            write(directory / 'expression_attempt.json', row)
            expression_ref = None
            try:
                count += 1
                expression_ref, report = compile_expression(refs, row['expression'], reference_id='expression_v1_' + row['label'], sources=records)
                save_reference(expression_ref, directory / 'common_reference')
            except ProgramCompileError as error:
                report = error.report
            write(directory / 'expression_report.json', report)
            results.append({'label': row['label'], 'status': report['status'],
                            'reference': expression_ref is not None, 'report': str(directory / 'expression_report.json')})
            if row['label'] in plan['legacy_comparisons']:
                old_ref = None
                try:
                    count += 1
                    old_ref, old_report = compile_program(refs, legacy(row['expression']), reference_id='legacy_compare_' + row['label'], sources=records)
                    save_reference(old_ref, directory / 'legacy_reference')
                except ProgramCompileError as error:
                    old_report = error.report
                write(directory / 'legacy_report.json', old_report)
                if (expression_ref is None) != (old_ref is None):
                    raise ValueError('Expression/legacy admission differs under equal factors')
                same = {'same_admission': True, 'continuous_ranges_equal': report['continuous_joint_ranges'] == old_report['continuous_joint_ranges']}
                if expression_ref is not None:
                    for name in ('time_offsets_s', 'joint_position_rad', 'joint_velocity_rad_s', 'anchor_orientation_xyzw'):
                        a, b = getattr(expression_ref, name), getattr(old_ref, name)
                        same[name] = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                else:
                    same['violations_equal'] = report['violations'] == old_report['violations']
                if not all(same.values()):
                    raise ValueError('Equal-factor expression failed exact legacy numeric parity')
                parity[row['label']] = same
            verify(plan['source_pins'])
        if count != 10:
            raise ValueError('Exact reference compile budget differs')
        write(output / 'completed.json', {'status': 'completed_reference_only', 'compile_calls': count,
              'results': results, 'legacy_parity': parity, 'source_pins_unchanged': True,
              'new_model_ONNX_FK_physics': 0, 'qualification': plan['qualification']})
        print(json.dumps({'status': 'completed_reference_only', 'compile_calls': count,
                          'reference_admitted': sum(r['reference'] for r in results), 'new_physics': 0}))
    except Exception:
        write(output / 'failed.json', {'status': 'stopped', 'compile_calls_started': count,
              'retry_allowed': False, 'new_model_ONNX_FK_physics': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-reference', action='store_true')
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--authority-sha256')
    args = parser.parse_args()
    output = args.output.absolute()
    if not output.resolve().is_relative_to(BASE.resolve()) or output.resolve() == BASE.resolve():
        raise ValueError('Dedicated private reference pilot directory required')
    for path in (output, *output.parents):
        if path.is_symlink() or (path.exists() and bool(getattr(path.lstat(), 'st_file_attributes', 0) & 1024)):
            raise ValueError('Linked output forbidden')
    plan = prepare()
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'prepared.json', plan)
    elif not args.execute_reference:
        raise FileExistsError('Exclusive preparation required')
    elif encoded(json.loads((output / 'prepared.json').read_bytes())) != encoded(plan):
        raise ValueError('Prepared reference plan changed')
    if not args.execute_reference:
        print(json.dumps({'status': 'prepared_only', 'plan_sha256': hashlib.sha256(encoded(plan)).hexdigest(), 'new_compiles_model_ONNX_FK_physics': 0}))
        return
    execute(output, plan, args.authority, args.authority_sha256)


if __name__ == '__main__':
    main()
