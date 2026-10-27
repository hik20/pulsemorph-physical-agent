"""Explicit projection-audit-gated DEVELOPMENT preview of recorded states, never new physics.

Uses the unchanged generalization_preview renderer (100 frames, 25 fps,
960x720, mj_forward only). Titles show method/tool and physics/model budgets.
The global fresh calibration is the baseline panel; all four global/local
calibration executions remain disclosed in the separate preview manifest.
V2 protocol/LLM observations use schema v2; runtime/comparison remain v1.
A null selected candidate is rejected explicitly, with no fabricated panel.
Completion and technical gates are required here, never presumed guaranteed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import generalization_preview as common

read, sha256, write_new_json = common.read, common.sha256, common.write_new_json
TOOLS = ('global_clock_v1', 'local_clock_1p4_2p8_v1')
METHODS = ('surrogate_search', 'llm_feedback', 'llm_feedback_off')
REQUIRED_AUDIT_SOURCES = {f'experiments/agentic/{name}' for name in (
    'audit_tool_selection_projection.py', 'project_tool_reference.py', 'tool_selection_driver_v2.py',
    'tool_selection_tasks_v2.py', 'tool_selection_batch_v2.py')}


def inventory(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file()}


def verify_audit(pilot, projection, audit_path):
    """Accept only an explicit projection audit bound to both immutable trees."""
    pilot, projection, audit_path = map(lambda p: Path(p).resolve(), (pilot, projection, audit_path))
    report = read(audit_path)
    flags = ('audit_passed', 'original_evidence_unchanged', 'audit_sources_unchanged',
             'compatible_projection_required', 'projection_raw_arrays_unchanged',
             'compatible_export_unchanged', 'failed_raw_audit_unchanged')
    if (report.get('schema') != 'pulsemorph.agentic_tool_selection_projection_audit.v1'
            or any(report.get(k) is not True for k in flags)
            or report.get('original_strict_g1_admission') is not False or report.get('errors') != []):
        raise ValueError('accepted explicit projection audit required; raw strict admission remains false')
    if Path(report['compatible_projection_root']).resolve() != projection:
        raise ValueError('wrong compatible projection root')
    current, projected = inventory(pilot), inventory(projection)
    if (not current or report.get('evidence_inventory_before') != current
            or report.get('evidence_inventory_after') != current):
        raise ValueError('audit does not match actual pilot evidence')
    if (not projected or report.get('projection_inventory_before') != projected
            or report.get('projection_inventory_after') != projected):
        raise ValueError('audit does not match actual projection evidence')
    index_path = projection / 'index.json'
    if sha256(index_path) != report.get('projection_index_sha256'):
        raise ValueError('projection index hash changed')
    index = read(index_path)
    if (index.get('schema') != 'pulsemorph.agentic_tool_reference_projection.v1'
            or Path(index['pilot']).resolve() != pilot
            or index.get('original_strict_g1_admission') is not False
            or index.get('projection_strict_g1_admission') is not True
            or index.get('numerical_only_no_resimulation') is not True
            or index.get('reference_count') != 100 or len(index.get('references', [])) != 100
            or index.get('original_inventory_before') != current or index.get('original_inventory_after') != current):
        raise ValueError('incompatible projection index scope or pilot binding')
    failed = report['failed_raw_audit']
    failed_path = Path(failed['path']).resolve()
    if (sha256(failed_path) != failed['sha256']
            or Path(index['failed_audit']['path']).resolve() != failed_path
            or index['failed_audit']['sha256'] != failed['sha256']):
        raise ValueError('failed raw audit hash binding changed')
    raw_report = read(failed_path)
    if (raw_report.get('schema') != 'pulsemorph.agentic_tool_selection_final_audit.v2'
            or raw_report.get('audit_passed') is not False
            or 'ValueError: unsupported target metadata fields' not in raw_report.get('errors', [])):
        raise ValueError('original raw strict rejection must remain preserved')
    pins = report.get('source_hashes_before')
    if (not pins or pins != report.get('source_hashes_after')
            or not REQUIRED_AUDIT_SOURCES.issubset({str(Path(name).as_posix()) for name in pins})):
        raise ValueError('audit source hashes are not stable')
    for name, digest in pins.items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or not path.is_file() or sha256(path) != digest:
            raise ValueError('audit source changed: ' + name)
    if len(report.get('trials', {})) != 100:
        raise ValueError('all100 numeric parity results required')
    for identity, row in report['trials'].items():
        parity = row.get('common_parity', {})
        if (parity.get('original_strict_g1_admission') is not False
                or any(parity.get(k) is not True for k in ('compatible_projection_admitted', 'raw_array_bytes_unchanged', 'save_load_passed'))
                or parity.get('decisions') != 200 or parity.get('future_samples') != 800):
            raise ValueError('incomplete numeric projection parity: ' + identity)
    return report, current


def render(pilot, projection, task_id, output, audit_path):
    pilot, output, audit_path = Path(pilot).resolve(), Path(output).resolve(), Path(audit_path).resolve()
    projection = Path(projection).resolve()
    if output.exists():
        raise FileExistsError('preview output already exists')
    if (ROOT / 'artifacts/media/agentic').resolve() not in output.parents:
        raise ValueError('preview output must be under artifacts/media/agentic')
    report, evidence = verify_audit(pilot, projection, audit_path)
    audit_digest = sha256(audit_path)
    comparison, protocol = read(pilot / 'comparison.json'), read(pilot / 'protocol.json')
    if (comparison.get('schema') != 'pulsemorph.agentic_tool_selection_comparison.v1'
            or comparison.get('development_only') is not True
            or protocol.get('phase') != 'development_comparison'
            or protocol.get('schema') != 'pulsemorph.agentic_tool_selection_protocol.v2'
            or comparison.get('comparison_valid') is not True
            or comparison.get('physical_trials') != 100
            or comparison.get('planned_budget') != dict(physical_trials=100, candidate_trials=96, calibration_trials=4)
            or protocol.get('imported_prior_trial_records', []) != []
            or comparison['protocol_sha256'] != sha256(pilot / 'protocol.json')):
        raise ValueError('completed development comparison with four calibration costs required')
    tasks = [t for t in comparison['tasks'] if t['task_id'] == task_id]
    if len(tasks) != 1:
        raise ValueError('unknown or duplicate task')
    task = tasks[0]
    if len(task['methods']) != 3 or [m['method'] for m in task['methods']] != list(METHODS):
        raise ValueError('expected exactly three methods')
    labels, selected = ['DEV BASE | GLOBAL | 1P/0M'], []
    for method, title, calls in zip(task['methods'], ('NUMERIC', 'LLM+FB', 'LLM-OFF'), (0, 8, 8)):
        row = method.get('selected')
        if row is None:
            raise ValueError('no physically eligible selected trial: ' + method['method'])
        tool = row.get('control_tool')
        if tool not in TOOLS or method.get('physical_trials') != 8 or method.get('model_proposal_calls') != calls:
            raise ValueError('tool or method budget mismatch')
        audited = [s for s in report['selections'] if s['task_id'] == task_id and s['method'] == method['method']]
        if len(audited) != 1 or audited[0]['selected_candidate'] != row['candidate_id'] or audited[0]['selected_tool'] != tool:
            raise ValueError('selected identity/tool differs from audit')
        labels.append(f"DEV {title} | {'GLOBAL' if tool == TOOLS[0] else 'LOCAL'} | 8P/{calls}M")
        selected.append(dict(method=method['method'], candidate_id=row['candidate_id'], control_tool=tool,
                             physical_trials=8, model_proposal_calls=calls))
    calibration_id = 'calibration_' + task['source_id']
    baseline = [r for r in comparison['calibrations'] if r['candidate_id'] == calibration_id]
    if len(comparison['calibrations']) != 4 or len(baseline) != 1 or baseline[0].get('control_tool') != TOOLS[0]:
        raise ValueError('fresh global calibration baseline required')
    parity = comparison.get('calibration_parity', {})
    if parity.get('parity_passed') is not True or parity.get('calibration_trials') != 4 or parity != report.get('calibration_parity'):
        raise ValueError('four fresh calibration parity proof required')
    # Shared renderer captures exact selected evaluation/manifest/trace hashes,
    # source XML/meshes and original protocol audio; all remain unchanged.
    dependencies = [Path(__file__).resolve(), Path(common.__file__).resolve(), ROOT / 'src/pulsemorph/evidence.py']
    dependencies += [ROOT / name for name in report['source_hashes_before']]
    pins = {str(p.resolve()): sha256(p) for p in dependencies}
    old_labels = common.LABELS
    try:
        common.LABELS = labels
        video = common.render_case(pilot, output, task)
    finally:
        common.LABELS = old_labels
    for path, digest in pins.items():
        if sha256(Path(path)) != digest:
            raise ValueError('preview dependency changed during render')
        dest = output / 'source_snapshot' / Path(path).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    verify_audit(pilot, projection, audit_path)
    if sha256(audit_path) != audit_digest or inventory(pilot) != evidence:
        raise ValueError('audit or evidence changed during render')
    write_new_json(output / 'tool_selection_manifest.json', dict(
        schema='pulsemorph.agentic_tool_selection_projection_preview.v1', development_only=True,
        protocol_schema='pulsemorph.agentic_tool_selection_protocol.v2',
        llm_observation_schema='pulsemorph.agentic_tool_selection_observation.v2',
        original_strict_g1_admission=False, compatible_export_verified=True,
        numerical_only_no_resimulation=True, projection_root=str(projection),
        projection_index_sha256=report['projection_index_sha256'],
        failed_raw_audit=report['failed_raw_audit'],
        corrected_reference_links=[r for r in read(projection / 'index.json')['references']
            if r['candidate_id'] in [calibration_id] + [s['candidate_id'] for s in selected]],
        null_selection_policy='reject; never synthesize unavailable candidate panels',
        task_id=task_id, audit_path=str(audit_path), audit_sha256=audit_digest,
        comparison_sha256=sha256(pilot / 'comparison.json'), protocol_sha256=sha256(pilot / 'protocol.json'),
        render_manifest_sha256=sha256(output / 'manifest.json'), preview_dependencies_sha256=pins,
        baseline=dict(candidate_id=calibration_id, control_tool=TOOLS[0], fresh=True),
        selected=selected, candidate_budget_each_method=8, model_calls_each_method=[0, 8, 8],
        imported_prior_trial_labels=0, calibration_trials=4, calibration_parity=parity,
        panel_budget_key='P = physical trials; M = model proposals; calibration costs separate',
        scope='DEVELOPMENT; actual recorded qpos through unchanged mj_forward renderer; no new physics or held-out quality claim.',
        audio=read(output / 'manifest.json')['audio'],
        source_capture_sha256={str(p.relative_to(output)): sha256(p) for p in (output / 'source_snapshot').rglob('*') if p.is_file()}))
    return video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'projection', 'output', 'audit'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    print(render(args.pilot, args.projection, args.task, args.output, args.audit))
