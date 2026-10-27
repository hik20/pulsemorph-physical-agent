"""Bind one real model proposal, music PCM and source packages to a reference.

No model, policy inference, FK or physics. A rejected proposal is preserved.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research/vendor/ProtoMotions'), str(HERE)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.planner = args.planner.resolve()
    args.output = args.output.resolve()
    allowed = ROOT / 'artifacts/runs/agentic'
    if not args.planner.is_relative_to(allowed) or not args.output.is_relative_to(allowed) or args.output == allowed:
        raise ValueError('planner/output must be dedicated local agentic artifact directories')
    import numpy as np
    from pulsemorph.motion.contracts import load_reference, save_reference, MotionReference
    from pulsemorph.agentic.phrase_program import compile_program, ProgramCompileError
    from pulsemorph.agentic.phrase_planner import verify_music_binding
    from pulsemorph.agentic.phrase_reference import plain

    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed_reference_preparation', 'new_llm_calls': 0,
              'new_onnx_calls': 0, 'new_physics_steps': 0, 'execution_authority': False}
    try:
        prior = ROOT / 'artifacts/runs/agentic/phrase_reference_plan_v2/report.json'
        pins = json.loads(prior.read_text())['source_pins']
        for path, digest in pins.items():
            if sha(ROOT / path) != digest:
                raise ValueError('original source pin changed: ' + path)
        pins[str(prior.relative_to(ROOT))] = sha(prior)
        result = json.loads((args.planner / 'result.json').read_text(encoding='utf8'))
        request = json.loads((args.planner / 'request.json').read_text(encoding='utf8'))
        if result['request_sha256'] != request['request_sha256'] or result['receipt'].get('live') is not True or result['receipt'].get('tool_events') != 0:
            raise ValueError('requires a real completed tool-free proposal bound to its request')
        music = ROOT / 'artifacts/runs/agentic/phrase_program_music_v1/two_section_original.wav'
        summary = verify_music_binding(request['music_summary'], music)
        refs = {}
        sources = []
        for name, job in {'wave': '20261027T023422_8efe4bc4', 'high5': '20261027T022322_76db8965'}.items():
            directory = ROOT / 'artifacts/runs/agentic/app/jobs' / job / 'trials/baseline/common_reference'
            refs[name] = load_reference(directory)
            for filename in ('manifest.json', 'arrays.npz'):
                path = directory / filename
                sources.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
        extra = [Path(__file__), HERE / 'adapter_program_v1.py', HERE / 'execute_program_v1.py',
                 ROOT / 'src/pulsemorph/agentic/phrase_program.py',
                 ROOT / 'src/pulsemorph/agentic/phrase_planner.py',
                 ROOT / 'src/pulsemorph/agentic/app/providers.py', music,
                 ROOT / 'artifacts/runs/agentic/phrase_program_music_v1/summary.json',
                 ROOT / 'artifacts/runs/agentic/phrase_program_music_v1/receipt.json',
                 args.planner / 'result.json', args.planner / 'program.json', args.planner / 'request.json']
        for path in extra:
            record = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
            pins[record['path']] = record['sha256']
            sources.append(record)
        reference, compiled = compile_program(refs, result['program'], reference_id='music_instruction_phrase_program_v1', sources=sources)
        metadata = plain(reference.metadata)
        metadata['context'].update(observation_id='phrase-program:offline-music:' + summary['sha256'], task_id='phrase_program_physical_v1')
        metadata['observation'].update(mode='offline_file', observed_through_time_s=8., audio_observed_through_sample=summary['sample_frames'])
        metadata['provenance']['notes'] += ['Offline actual eight-second PCM features and exact user instruction informed one real LLM program. No audio samples were heard by the model; no beat-timing or physical-expression success implied.']
        reference = MotionReference(metadata, reference.time_offsets_s, reference.joint_position_rad, reference.joint_velocity_rad_s, reference.anchor_orientation_xyzw)
        save_reference(reference, args.output / 'common_reference')
        report.update(compiled)
        report.update(source_pins=pins, status='reference_only_passed_awaiting_design_review',
                      planner_request_sha256=result['request_sha256'], music_sha256=summary['sha256'],
                      model_proposal_calls_in_source=1, model_returned=result['receipt'].get('model_returned'),
                      reference_manifest_sha256=sha(args.output / 'common_reference/manifest.json'),
                      reference_arrays_sha256=sha(args.output / 'common_reference/arrays.npz'))
        # Call the concrete executor's actual eight-feed assembly preflight;
        # temporary binding is reference-only and does not grant execution.
        import execute_program_v1 as executor
        executor.REFERENCE_PLAN = args.output.resolve()
        _, checks = executor.preflight()
        write(args.output / 'prefix_actual_assembly_parity.json', checks)
        report['actual_assembly_preflight'] = checks
    except ProgramCompileError as exc:
        report.update(status='rejected_compiler', compiler_report=exc.report, error=str(exc))
    except Exception as exc:
        report.update(status='failed_reference_preparation', error_type=type(exc).__name__, error=str(exc))
    write(args.output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'error', 'error_type') if key in report}))
    return 0 if report['status'] == 'reference_only_passed_awaiting_design_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
